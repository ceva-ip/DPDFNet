"""Isolated single-thread experiments against the accepted Oct3 W7A8 builds.

Run from native_inference in the existing offline development container.
All timing stages run sequentially, after builds and correctness checks finish.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'native'))
COMBINATIONS = {
    'combo_asm': ('gate_staged','quant_asm64'),
    'combo_asm_io': ('gate_staged','quant_asm64','block_io'),
    'combo_asm_conv': ('gate_staged','quant_asm64','block_io','depthwise_deinterleave'),
    'combo_asm_norm': ('gate_staged','quant_asm64','block_io','depthwise_deinterleave','norm8'),
    'combo_layout': ('gate_staged','quant_recurrent_layout','block_io','depthwise_deinterleave'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_variant(name, source):
    path = source/'avx2.c'
    code = path.read_text()
    start = code.index('void dpdf_gates_avx2(')
    end = code.index('void dpdf_activations_avx2(', start)
    block = code[start:end]
    if name in ('gate1', 'gate4', 'gate8'):
        assert block.count('#pragma GCC unroll 2') == 1
        block = block.replace('#pragma GCC unroll 2', '#pragma GCC unroll '+name[-1])
    elif name == 'gate_staged':
        block = '''void dpdf_gates_avx2(const float *a, const float *b, const float *old, float *out, int rows) {
    for (int r=0; r<rows; ++r) {
        float resets[64],updates[64];
        for (int c=0; c<64; c+=8) {
            _mm256_storeu_ps(resets+c,sigmoid8(_mm256_add_ps(_mm256_loadu_ps(a+r*192+c),_mm256_loadu_ps(b+r*192+c))));
            _mm256_storeu_ps(updates+c,sigmoid8(_mm256_add_ps(_mm256_loadu_ps(a+r*192+64+c),_mm256_loadu_ps(b+r*192+64+c))));
        }
        for (int c=0; c<64; c+=8) {
            __m256 reset=_mm256_loadu_ps(resets+c),update=_mm256_loadu_ps(updates+c);
            __m256 candidate=tanh8(_mm256_fmadd_ps(reset,_mm256_loadu_ps(b+r*192+128+c),_mm256_loadu_ps(a+r*192+128+c)));
            __m256 result=_mm256_fmadd_ps(update,_mm256_sub_ps(_mm256_loadu_ps(old+r*64+c),candidate),candidate);
            _mm256_storeu_ps(out+r*64+c,result);
        }
    }
}
'''
    else:
        raise ValueError(name)
    path.write_text(code[:start]+block+code[end:])


def apply(name, source):
    if name.startswith('gate'):
        gate_variant(name, source)
        return
    for module in ('oct7_quant.transforms', 'oct7_graph.transforms'):
        mod = __import__(module, fromlist=['VARIANTS', 'apply'])
        if name in mod.VARIANTS:
            mod.apply(name, source)
            return
    raise ValueError('Unknown variant '+name)


def manifest(source):
    return {p.name: sha(p) for p in sorted(source.iterdir()) if p.is_file()
            and (p.suffix in ('.c', '.h', '.S') or p.name=='CMakeLists.txt')}


def build(size, variant, suffix=''):
    work = ROOT/'scratch/oct7'/str(size)
    source = work/variant
    if not source.exists():
        baseline = ROOT/('scratch/further_optimization2' if size==2 else 'scratch/further_optimization')/'best_single'
        source.mkdir(parents=True)
        for p in baseline.iterdir():
            if p.is_file() and (p.suffix in ('.c', '.h', '.S') or p.name=='CMakeLists.txt'):
                shutil.copyfile(p, source/p.name)
        for name in COMBINATIONS.get(variant, variant.split('+')):
            apply(name, source)
        cmake = (source/'CMakeLists.txt').read_text()
        # Reproduction does not rely on legacy absolute scratch paths.
        cmake = cmake.replace('${CMAKE_CURRENT_SOURCE_DIR}/../models/native_blocks/erb_0.f32',
                              str(ROOT/'models/native_blocks/erb_0.f32'))
        (source/'CMakeLists.txt').write_text(cmake)
        (source/'source_manifest.json').write_text(json.dumps(manifest(source), indent=2)+'\n')
    expected = json.loads((source/'source_manifest.json').read_text())
    assert expected == manifest(source), 'Preserved candidate sources changed'
    target = ROOT/'build'/f'oct7_{size}_{variant}{suffix}'
    options = ['-DDPDF_EXTENDED_MODEL=ON', '-DCMAKE_BUILD_TYPE=Release',
        f'-DDPDF_GENERATED_MODEL={source}/generated_model.c',
        f'-DDPDF_TEST_WEIGHTS={ROOT}/models/rework{size}/weights.f32']
    if suffix == '_asan':
        options += ['-DDPDF_SANITIZE=ON', '-DCMAKE_EXE_LINKER_FLAGS=-no-pie']
    if suffix == '_scalar':
        options += ['-DDPDF_ENABLE_AVX2=OFF']
    for name, command in (
        ('configure', ['cmake','-S',str(source),'-B',str(target),*options]),
        ('build', ['cmake','--build',str(target),'-j4']),
        ('contracts', ['ctest','--test-dir',str(target),'--output-on-failure'])):
        log = work/f'{variant}{suffix}_{name}.log'
        with log.open('w') as f:
            done = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
        if done.returncode:
            print(log.read_text()[-6000:], flush=True)
            raise RuntimeError(str(log))
    print(f'{size} {variant}{suffix}: built and contracts passed', flush=True)


def measure(args, variant):
    from extended_probe import ExtendedModel
    from probe import cpu_name, session, spectra
    from optimization_probe import exact_recurrent_parity
    from further_streaming_timing import run
    import os
    import platform
    reference = session(ROOT/f'models/dpdfnet{args.size}_48khz_hr.onnx')
    baseline = ROOT/'build'/('further2_best_single' if args.size==2 else 'further_best_single')
    candidate = ROOT/'build'/f'oct7_{args.size}_{variant}'
    builds = {'baseline': baseline, 'candidate': candidate}
    weights = ROOT/f'models/rework{args.size}/weights.f32'
    models = {k: ExtendedModel(reference,4,8,7,build=v,weights=weights) for k,v in builds.items()}
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'driver_sha256': sha(__file__),
        'model_size': args.size, 'variant': variant, 'baseline': 'Oct3 best_single',
        'method': 'Single-thread preallocated C calls, in-place state, balanced order, no samples filtered.',
        'environment': {'cpu':cpu_name(),'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0))},
        'model_sha256': sha(ROOT/f'models/dpdfnet{args.size}_48khz_hr.onnx'),
        'weights_sha256': sha(weights), 'artifacts':{k:sha(v/'libdpdf_full.so') for k,v in builds.items()},
        'candidate_source_manifest':json.loads((ROOT/f'scratch/oct7/{args.size}/{variant}/source_manifest.json').read_text()),
        'owned_bytes':{k:v.owned_bytes for k,v in models.items()}}
    try:
        report['parity'] = exact_recurrent_parity(models['baseline'],models['candidate'],spectra(1000))
        frames = spectra(args.frames+100)
        report['timed_frames_per_run'] = args.frames
        report['warmup'] = 100
        report['continuous'] = run(models,frames,args.repeats,False,100)
        report['paced'] = []
        report['standalone_paced'] = []
        if args.phase == 'final':
            report['paced'] = run(models,frames,args.repeats,True,100)
            for repeat in range(args.repeats):
                names = list(models)
                if repeat%2:
                    names.reverse()
                for name in names:
                    item = run({name:models[name]},frames,1,True,100)[0]
                    item['repeat'] = repeat
                    report['standalone_paced'].append(item)
        report['summary'] = {}
        for mode in ('continuous','paced','standalone_paced'):
            if not report[mode]:
                continue
            records = {}
            for name in models:
                rows = [v['implementations'][name] for v in report[mode] if name in v['implementations']]
                records[name] = {k:median(v['wall'][k] for v in rows) for k in ('mean_ms','p50_ms','p99_ms')}
                records[name]['max_ms'] = max(v['wall']['max_ms'] for v in rows)
                records[name]['over_10ms'] = sum(v['wall']['over_10ms'] for v in rows)
                records[name]['process_cpu_ms'] = median(v['process_cpu']['mean_ms'] for v in rows)
                if mode != 'continuous':
                    records[name]['late_completions'] = sum(v['completion_after_release']['over_10ms'] for v in rows)
            records['mean_reduction_percent'] = 100*(1-records['candidate']['mean_ms']/records['baseline']['mean_ms'])
            report['summary'][mode] = records
        output = ROOT/'results'/f'oct7_{args.size}_{variant}_{args.phase}.json'
        output.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'model':args.size,'variant':variant,'summary':report['summary']}),flush=True)
    finally:
        for model in models.values():
            model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['build','screen','final','asan','scalar'])
    p.add_argument('variants',nargs='+')
    p.add_argument('--size',type=int,choices=[2,8],default=8)
    p.add_argument('--frames',type=int,default=600)
    p.add_argument('--repeats',type=int,default=3)
    a = p.parse_args()
    for variant in a.variants:
        if a.phase in ('build','asan','scalar'):
            build(a.size,variant,{'asan':'_asan','scalar':'_scalar'}.get(a.phase,''))
        else:
            measure(a,variant)


if __name__ == '__main__':
    main()
