"""Reproducible, isolated W7A8 latency experiments. Run from native_inference.

No production kernel or previously benchmarked artifact is overwritten.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_int8_variant import once

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'scratch/w7_followup'


def pack32(code):
    start = code.index('    for (int j=0;j<k;j+=8) {', code.index('    __m256 inv='))
    code = code[:start] + code[start:].replace('    for (int j=0;j<k;j+=8) {', '''    int j=0;
    for (;j+31<k;j+=32) {
        __m256i q[4];
        for (int t=0;t<4;++t) {
            __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j+t*8),inv),z);
            q[t]=_mm256_cvtps_epi32(v);
            q[t]=_mm256_min_epi32(_mm256_set1_epi32(254),_mm256_max_epi32(_mm256_setzero_si256(),q[t]));
        }
        __m256i p=_mm256_packus_epi16(_mm256_packs_epi32(q[0],q[1]),_mm256_packs_epi32(q[2],q[3]));
        /* AVX2 packs within 128-bit lanes: reorder four-byte groups. */
        p=_mm256_permutevar8x32_epi32(p,_mm256_setr_epi32(0,4,1,5,2,6,3,7));
        _mm256_storeu_si256((__m256i *)(out+j),p);
    }
    for (;j<k;j+=8) {''', 1)
    return code


def fused_epilogue(code):
    start = code.index('static __attribute__((noinline)) void qdot64(')
    end = code.index('\nstatic void qaffine_row', start)
    block = code[start:end]
    block = once(block, 'const int8_t *packed,int k,int32_t *out)',
                 'const int8_t *packed,int k,const int32_t *wsum,const float *wscale,\n'
                 '        const float *bias,float activation_scale,int zp,float *out)')
    for t in range(8):
        block = once(block, f'    _mm256_storeu_si256((__m256i *)(out+{t*8}),sum{t});', f'''    {{
        __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(wsum+{t*8})),_mm256_set1_epi32(-zp));
        __m256 scale=_mm256_mul_ps(_mm256_set1_ps(activation_scale),_mm256_loadu_ps(wscale+{t*8}));
        __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum{t},correction)),scale,_mm256_loadu_ps(bias+{t*8}));
        _mm256_storeu_ps(out+{t*8},result);
    }}''')
    code = code[:start]+block+code[end:]
    start = code.index('        int32_t integer_sums[64];')
    end = code.index('\n    }\n    for (;c<n;', start)
    return code[:start]+'''        qdot64(activation,q->packed+c*k,k,q->sum+c,q->scale+c,
               bias+c,activation_scale,zp,y+c);'''+code[end:]


def reciprocal(code):
    # exp8 clamps its exponent; use the stable sigmoid form to keep rcp inputs
    # in [1,2]. Direct rcp(1+exp(-x)) can flush tiny reciprocals to zero.
    start = code.index('static inline __m256 sigmoid8(')
    end = code.index('void dpdf_gates_avx2(', start)
    return code[:start]+'''static inline __m256 reciprocal8(__m256 d) {
    __m256 r=_mm256_rcp_ps(d);
    return _mm256_fmadd_ps(r,_mm256_fnmadd_ps(d,r,_mm256_set1_ps(1)),r);
}
static inline __m256 sigmoid8(__m256 x) {
    __m256 absolute=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),x);
    __m256 e=exp8(_mm256_sub_ps(_mm256_setzero_ps(),absolute));
    __m256 inv=reciprocal8(_mm256_add_ps(_mm256_set1_ps(1),e));
    return _mm256_blendv_ps(inv,_mm256_mul_ps(e,inv),_mm256_cmp_ps(x,_mm256_setzero_ps(),_CMP_LT_OQ));
}
static inline __m256 tanh8(__m256 x) {
    __m256 sign=_mm256_and_ps(x,_mm256_set1_ps(-0.0f));
    __m256 absolute=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),x);
    __m256 e=exp8(_mm256_mul_ps(_mm256_set1_ps(-2),absolute));
    return _mm256_xor_ps(sign,_mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1),e),reciprocal8(_mm256_add_ps(_mm256_set1_ps(1),e))));
}
'''+code[end:]


def batch8(code):
    kernel = '''static __attribute__((noinline)) void qdot8(const int8_t *activation,
        const int8_t *packed,int k,__m256i *out) {
    const __m256i ones=_mm256_set1_epi16(1);
'''
    for t in range(8):
        kernel += f'    __m256i sum{t}=_mm256_setzero_si256();\n'
    kernel += '''    for (int j=0;j<k;j+=4) {
        __m256i w=_mm256_loadu_si256((const __m256i *)(packed+j*8));
'''
    for t in range(8):
        kernel += f'''        {{ int32_t bytes; memcpy(&bytes,activation+{t}*k+j,4);
          __m256i a=_mm256_set1_epi32(bytes);
          sum{t}=_mm256_add_epi32(sum{t},_mm256_madd_epi16(_mm256_maddubs_epi16(a,w),ones)); }}
'''
    kernel += '    }\n'
    for t in range(8):
        kernel += f'    out[{t}]=sum{t};\n'
    kernel += '''}
static void qaffine_eight(const dpdf_qmatrix *q,const int8_t *activation,
        const float *scales,const int *zp,const float *bias,float *y,int m) {
    const int k=q->k,n=q->n;
    for (int r=0;r<m;r+=8) for (int c=0;c<n;c+=8) {
        __m256i sums[8];
        qdot8(activation+r*k,q->packed+c*k,k,sums);
        for (int t=0;t<8;++t) {
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+c)),_mm256_set1_epi32(-zp[r+t]));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r+t]),_mm256_loadu_ps(q->scale+c));
            _mm256_storeu_ps(y+(r+t)*n+c,_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t],correction)),scale,_mm256_loadu_ps(bias+c)));
        }
    }
}

'''
    at = code.index('static void qaffine_batch_tiled(')
    code = code[:at]+kernel+code[at:]
    start = code.index('static void qaffine_batch_tiled(')
    end = code.index('static void qaffine_batch(', start)
    block = once(code[start:end], '    for (int r=0;r<m;r+=4)',
                 '    if (m%8==0) { qaffine_eight(q,activation,scales,zp,bias,y,m); return; }\n    for (int r=0;r<m;r+=4)')
    code = code[:start]+block+code[end:]
    start = code.index('void dpdf_qaffine_pair(')
    block = once(code[start:], '    for (int r=0;r<m;r+=4)', '''    if (m%8==0) {
        qaffine_eight(q0,activation,scales,zp,bias0,y0,m);
        qaffine_eight(q1,activation,scales,zp,bias1,y1,m); return;
    }
    for (int r=0;r<m;r+=4)''')
    return code[:start]+block


def prepare(name):
    destination = WORK/name
    subprocess.run([sys.executable, str(Path(__file__).with_name('prepare_int8_variant.py')),
                    'w7', str(destination)], check=True)
    ip = destination/'int8.c'
    ap = destination/'avx2.c'
    code = ip.read_text()
    if name in ('pack32', 'combined', 'combined_lto', 'exact_all', 'pgo', 'pack_gate', 'pack_gate_lto', 'pack_poly5', 'pack_fit5'):
        code = pack32(code)
    if name in ('epilogue', 'combined', 'combined_lto', 'exact_all', 'pgo'):
        code = fused_epilogue(code)
    if name == 'batch8':
        code = batch8(code)
    if name == 'column_first':
        for stride in (8, 16):
            old = f'for (int r=0;r<m;r+=4) for (int c=0;c<n;c+={stride})'
            code = once(code, old, f'for (int c=0;c<n;c+={stride}) for (int r=0;r<m;r+=4)')
    if name in ('unroll2', 'unroll4'):
        unroll = name[-1]
        for function, following in [('qdot64(', '\nstatic void qaffine_row'),
                                    ('qdot4pair(', 'static void qaffine_batch_tiled')]:
            start = code.index('static __attribute__((noinline)) void '+function)
            end = code.index(following, start)
            block = once(code[start:end], '    for (int j=0;',
                         f'    #pragma GCC unroll {unroll}\n    for (int j=0;')
            code = code[:start]+block+code[end:]
    ip.write_text(code)
    if name == 'reciprocal':
        ap.write_text(reciprocal(ap.read_text()))
    if name in ('gate4', 'exact_all', 'pgo', 'pack_gate', 'pack_gate_lto'):
        ap.write_text(once(ap.read_text(), '#pragma GCC unroll 2', '#pragma GCC unroll 4'))
    if name in ('poly5', 'pack_poly5'):
        ac = ap.read_text()
        ac = once(ac, '    __m256 p=_mm256_set1_ps(1.0f/5040.0f);\n'
                     '    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/720.0f));\n'
                     '    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/120.0f));',
                     '    __m256 p=_mm256_set1_ps(1.0f/120.0f);')
        ap.write_text(ac.replace('degree-7 Taylor', 'degree-5 Taylor (research approximation)'))
    if name == 'pack_fit5':
        import numpy as np
        half = np.log(2)/2
        coefficients = np.polynomial.Chebyshev.interpolate(np.exp, 5, domain=[-half, half]).convert(kind=np.polynomial.Polynomial).coef.astype(np.float32)
        ac = ap.read_text()
        start = ac.index('    __m256 p=_mm256_set1_ps(1.0f/5040.0f);')
        end = ac.index('    __m256i e=', start)
        poly = f'    __m256 p=_mm256_set1_ps({float(coefficients[5]).hex()}f);\n'
        for coefficient in coefficients[4::-1]:
            poly += f'    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps({float(coefficient).hex()}f));\n'
        ap.write_text((ac[:start]+poly+ac[end:]).replace('degree-7 Taylor', 'degree-5 Chebyshev interpolation (research)'))
        (WORK/'pack_fit5_coefficients.json').write_text(json.dumps({'domain': [-half, half], 'ascending_coefficients_f32': coefficients.tolist()}, indent=2)+'\n')
    if name == 'profile':
        # Reuse the existing diagnostic instrumentation, restricted to model 8.
        old = (ROOT/'scratch/latency_followup/profile.py').read_text()
        old = old.replace("root=Path(__file__).resolve().parents[2]", f'root=Path({str(ROOT)!r})')
        old = old.replace("folder=root/'scratch/latency_followup'", "folder=root/'scratch/w7_followup'")
        old = old.replace("    shutil.copytree(root/'native',folder/'preserved',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n", '')
        old = old.replace("    shutil.copytree(root/'native',folder/'profile_src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))\n", '')
        old = old.replace("profile_src", "profile").replace("[(8,'rework8'),(2,'rework2')]", "[(8,'rework8')]")
        old = old.replace('build/followup_profile_{model}', 'build/w7_followup_profile')
        old = old.replace('results/dpdfnet{model}_followup_profile.json', 'results/w7_followup_profile.json')
        (WORK/'profile.py').write_text(old)
        subprocess.run([sys.executable, str(WORK/'profile.py')], check=True)
    return destination


def run_logged(args, log):
    with log.open('w') as out:
        result = subprocess.run(args, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-12000:], flush=True)
        raise RuntimeError(f'{args[0]} failed: {log}')


def build(name, source, extra=()):
    target = ROOT/'build'/f'w7_followup_{name}'
    generated = WORK/'profile_8.c' if name == 'profile' else ROOT/'models/rework8/generated_model.c'
    run_logged(['cmake', '-S', str(source), '-B', str(target), '-DDPDF_EXTENDED_MODEL=ON',
                f'-DDPDF_GENERATED_MODEL={generated}', f'-DDPDF_TEST_WEIGHTS={ROOT}/models/rework8/weights.f32',
                *extra], WORK/f'{name}_configure.log')
    run_logged(['cmake', '--build', str(target), '-j4'], WORK/f'{name}_build.log')
    run_logged(['ctest', '--test-dir', str(target), '--output-on-failure'], WORK/f'{name}_contracts.log')
    print(f'{name}: built, all contracts passed', flush=True)
    return target


def screen(name, target):
    probe = 'experiments/range_latency_probe.py' if name in ('reciprocal', 'poly5', 'pack_poly5', 'pack_fit5') else 'latency_probe.py'
    output = ROOT/'results'/f'w7_followup_{name}_screen.json'
    run_logged([sys.executable, str(ROOT/'native'/probe), '--model', str(ROOT/'models/dpdfnet8_48khz_hr.onnx'),
                '--weights', str(ROOT/'models/rework8/weights.f32'), '--baseline-build', str(ROOT/'build/range_w78'),
                '--candidate-build', str(target), '--output', str(output),
                '--frames', '600', '--repeats', '3', '--paced-repeats', '0'], WORK/f'{name}_screen.log')
    data = json.loads(output.read_text())
    if name in ('reciprocal', 'poly5', 'pack_poly5', 'pack_fit5'):
        data['parity']['note'] = 'Changed FP32 activation approximation on the same W7A8 grid; numerical differences are not a perceptual acceptance test.'
        output.write_text(json.dumps(data, indent=2)+'\n')
    means = {n: sum(r['implementations'][n]['wall']['mean_ms'] for r in data['continuous'])/3
             for n in ('baseline', 'candidate')}
    print(json.dumps({'variant': name, 'wall_mean_ms': means,
                      'reduction_percent': 100*(1-means['candidate']/means['baseline']),
                      'parity': data['parity']}), flush=True)


def train(target):
    sys.path.insert(0, str(ROOT/'native'))
    from extended_probe import ExtendedModel, CONFIGS
    from probe import session, spectra, initial_state, audio_spectra
    ref = session(ROOT/'models/dpdfnet8_48khz_hr.onnx')
    model = ExtendedModel(ref, *CONFIGS['fc_and_1x1_8'], build=target,
                          weights=ROOT/'models/rework8/weights.f32')
    sources = [ROOT/'scratch/fullband/robustness/audio'/case/'input.wav'
               for case in ('clean_00033', 'low_00525')]
    sources.append(ROOT/'scratch/fullband/long_noise/pink.wav')
    try:
        for frames in [spectra(1000), *[audio_spectra(p)[:500] for p in sources]]:
            state = initial_state(model)
            for frame in frames:
                _, state = model.run(None, {'spec': frame, 'state_in': state})
    finally:
        model.close()
    print('PGO training: 1000 synthetic + 500 frames each clean, quiet, pink-noise', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variants', nargs='+', choices=['profile', 'pack32', 'epilogue', 'reciprocal',
                        'unroll2', 'unroll4', 'gate4', 'lto', 'combined', 'combined_lto',
                        'batch8', 'column_first', 'poly5', 'exact_all', 'pgo', 'pack_gate', 'pack_gate_lto', 'pack_poly5', 'pack_fit5'])
    parser.add_argument('--train', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.train:
        train(args.train)
        return
    for name in args.variants:
        source = prepare(name)
        extra = ['-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON'] if 'lto' in name or name in ('exact_all', 'pgo') else []
        if name == 'pgo':
            profile_dir = WORK/'pgo_counters'
            target = build(name, source, [*extra, f'-DCMAKE_C_FLAGS=-fprofile-generate={profile_dir}'])
            # A separate process must exit to flush GCC's counters.
            subprocess.run([sys.executable, __file__, 'pgo', '--train', str(target)], check=True)
            extra += [f'-DCMAKE_C_FLAGS=-fprofile-use={profile_dir} -fprofile-correction']
        target = build(name, source, extra)
        if name == 'pack_fit5':
            sys.path.insert(0, str(ROOT/'native'))
            from probe import library, check_activations
            checks = {}
            for other in ('pack32', 'pack_poly5', 'pack_fit5'):
                try:
                    values = check_activations(library(ROOT/'build'/f'w7_followup_{other}'/'libdpdf_dprnn.so'))
                    checks[other] = {'existing_activation_contract_passed': True, **values}
                except AssertionError as error:
                    checks[other] = {'existing_activation_contract_passed': False, 'failure': str(error)[:1000]}
            (ROOT/'results/w7_followup_activation_contracts.json').write_text(json.dumps(checks, indent=2)+'\n')
            print(json.dumps(checks), flush=True)
            if not checks['pack_fit5']['existing_activation_contract_passed']:
                raise RuntimeError('Fitted approximation failed existing activation tolerance')
        if name == 'profile':
            subprocess.run([sys.executable, str(WORK/'profile.py'), 'run'], check=True)
        else:
            screen(name, target)


if __name__ == '__main__':
    main()
