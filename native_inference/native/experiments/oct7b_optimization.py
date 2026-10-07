"""Next-round experiments against committed Oct7 combo_asm_norm, both models.

Keep builds/validation/profiling separate from latency and memory jobs. Sources
and existing Oct7 results are immutable. Approximate candidates are explicitly
labelled and require new perceptual scoring before recommendation.
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
MODULES = ('oct7b_quant.transforms', 'oct7b_graph.transforms', 'oct7b_compiler.transforms',
           'oct7b_dense.transforms', 'oct7b_conv.transforms', 'oct7b_pair.transforms',
           'oct7b_wide.transforms')
COMBINATIONS = {
    'combo_core': ('quant_fused192_init_inline','dense_int8_pad8','graph_gru_fuse'),
    'combo_core_pgo': ('compiler_pgo','quant_fused192_init_inline','dense_int8_pad8','graph_gru_fuse'),
    'combo_core_ipo': ('compiler_ipo_nointerpose','quant_fused192_init_inline','dense_int8_pad8','graph_gru_fuse'),
    'combo_core192': ('quant_fused192','dense_int8_pad8','graph_gru_fuse'),
    'combo_core192_bias': ('quant_fused192','dense_int8_pad8','graph_gru_fuse','graph_bias_relu'),
    'combo_core192_pgo': ('compiler_pgo','quant_fused192','dense_int8_pad8','graph_gru_fuse','graph_bias_relu'),
    'combo_exact8': ('compiler_pgo','quant_fused192_init_inline','dense_int8_pad8','graph_gru_fuse','conv_row_pair','quant_pair64'),
    'combo_exact2': ('quant_fused192','dense_int8_pad8','graph_gru_fuse','graph_bias_relu','conv_row_pair','quant_pair64'),
    'combo_min2': ('quant_fused192','dense_int8_pad8','conv_row_pair','quant_pair64'),
    'combo_wide8': ('compiler_pgo','quant_fused192_init_inline','dense_int8_pad8','graph_gru_fuse','conv_row_pair','quant_fused256','quant_pair64'),
    'combo_wide2': ('quant_fused192','dense_int8_pad8','graph_gru_fuse','graph_bias_relu','conv_row_pair','quant_fused256','quant_pair64'),
    'combo_approx2': ('quant_fused192','dense_int8_pad8','graph_gru_approx','graph_bias_relu','conv_row_pair','quant_pair64'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest(source):
    return {p.name: sha(p) for p in sorted(source.iterdir()) if p.is_file()
            and (p.suffix in ('.c','.h','.S') or p.name=='CMakeLists.txt')}


def module_for(name):
    for module in MODULES:
        if not (Path(__file__).parent/Path(*module.split('.')).with_suffix('.py')).is_file():
            continue
        mod = __import__(module, fromlist=['VARIANTS'])
        if name in mod.VARIANTS:
            return mod
    raise ValueError('Unknown variant '+name)


def names_for(variant):
    return COMBINATIONS.get(variant, tuple(variant.split('+')))


def approximate(variant):
    return any(name in getattr(module_for(name), 'APPROXIMATE_VARIANTS',
                               getattr(module_for(name), 'APPROX_VARIANTS', ()))
               for name in names_for(variant) if name!='profile')


def baseline_identity(size):
    source = ROOT/f'scratch/oct7/{size}/combo_asm_norm'
    saved = json.loads((source/'source_manifest.json').read_text())
    assert manifest(source) == saved, 'Committed baseline snapshot changed'
    library = ROOT/f'build/oct7_{size}_combo_asm_norm/libdpdf_full.so'
    summary = json.loads((ROOT/'results/oct7_summary.json').read_text())['models'][str(size)]
    assert sha(library) == summary['library_sha256']['candidate'], 'Baseline binary changed'
    assert saved == summary['source_manifest']
    return source, library


def profile_transform(source):
    import re
    path = source/'generated_model.c'
    code = path.read_text()
    start = code.index('int dpdf_model_process(')
    body = code[start:]
    nodes = list(re.finditer(r'/\* (\d+): (\w+) ([^\n]*?) \*/', body))
    assert nodes and int(nodes[0][1]) == 0
    count = max(int(m[1]) for m in nodes)+1
    head = '#include <time.h>\n'+code[:start]
    head += (f'static double oct7b_node_times[{count}];\n'
             'static double oct7b_stamp(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }\n'
             'void oct7b_profile_reset(void) { memset(oct7b_node_times,0,sizeof(oct7b_node_times)); }\n'
             'void oct7b_profile_get(double *out) { memcpy(out,oct7b_node_times,sizeof(oct7b_node_times)); }\n')
    for index in range(len(nodes)-1,-1,-1):
        match = nodes[index]
        number = int(match[1])
        statement = ('double oct7b_begin=oct7b_stamp();' if index==0 else
                     f'oct7b_node_times[{int(nodes[index-1][1])}]+=oct7b_stamp()-oct7b_begin; oct7b_begin=oct7b_stamp();')
        body = body[:match.end()]+'\n'+statement+body[match.end():]
    assert body.count('return 0;\n}') == 1
    body = body.replace('return 0;\n}', f'oct7b_node_times[{int(nodes[-1][1])}]+=oct7b_stamp()-oct7b_begin; return 0;\n}}')
    path.write_text(head+body)
    (source/'profile_nodes.json').write_text(json.dumps([
        {'index':int(m[1]),'operation':m[2],'name':m[3]} for m in nodes],indent=2)+'\n')


def prepare(size, variant):
    base, _ = baseline_identity(size)
    source = ROOT/f'scratch/oct7b/{size}/{variant}'
    if not source.exists():
        source.mkdir(parents=True)
        for path in base.iterdir():
            if path.name in manifest(base):
                shutil.copyfile(path,source/path.name)
        for name in names_for(variant):
            profile_transform(source) if name=='profile' else module_for(name).apply(name,source)
        (source/'source_manifest.json').write_text(json.dumps(manifest(source),indent=2)+'\n')
    assert manifest(source) == json.loads((source/'source_manifest.json').read_text())
    return source


def build(size, variant, suffix=''):
    source = prepare(size,variant)
    target = ROOT/f'build/oct7b_{size}_{variant}{suffix}'
    work = ROOT/f'scratch/oct7b/{size}'
    common = ['-DDPDF_EXTENDED_MODEL=ON','-DCMAKE_BUILD_TYPE=Release',
              f'-DDPDF_GENERATED_MODEL={source}/generated_model.c',
              f'-DDPDF_TEST_WEIGHTS={ROOT}/models/rework{size}/weights.f32']
    if suffix=='_asan':
        common += ['-DDPDF_SANITIZE=ON','-DCMAKE_EXE_LINKER_FLAGS=-no-pie']
    if suffix=='_scalar':
        common += ['-DDPDF_ENABLE_AVX2=OFF']
    modules = [module_for(name) for name in names_for(variant) if name!='profile']
    pgo = [(module,name) for module,name in zip(modules,[n for n in names_for(variant) if n!='profile'])
           if name in getattr(module,'PGO_VARIANTS',())]
    assert len(pgo)<=1
    def step(label,command):
        log = work/f'{variant}{suffix}_{label}.log'
        with log.open('w') as output:
            done = subprocess.run(command,stdout=output,stderr=subprocess.STDOUT)
        if done.returncode:
            print(log.read_text()[-7000:],flush=True)
            raise RuntimeError(str(log))
    def configure(phase):
        options = list(common)
        for name in names_for(variant):
            if name=='profile':
                continue
            mod = module_for(name)
            if hasattr(mod,'build_options'):
                options += mod.build_options(name,source,target,phase=phase)
        step('configure_'+phase,['cmake','-S',str(source),'-B',str(target),*options])
        step('build_'+phase,['cmake','--build',str(target),'-j4'])
    if pgo and not suffix:
        training = ROOT/f'results/oct7b_{size}_{variant}_training.json'
        counters = pgo[0][0].profile_dir(target)
        if training.exists() or any(counters.rglob('*.gcda')):
            raise RuntimeError('PGO training is immutable; use a fresh variant name rather than overwriting counters or a measured binary')
        configure('generate')
        step('train',[sys.executable,str(Path(pgo[0][0].__file__).with_name('train.py')),
                      '--model',str(ROOT/f'models/dpdfnet{size}_48khz_hr.onnx'),
                      '--weights',str(ROOT/f'models/rework{size}/weights.f32'),
                      '--build',str(target),'--source',str(source),'--output',str(training)])
    configure('use' if not suffix else 'plain')
    step('contracts',['ctest','--test-dir',str(target),'--output-on-failure'])
    print(f'{size} {variant}{suffix}: built and contracts passed',flush=True)


def parity(left,right,frames,is_approx):
    if not is_approx:
        from optimization_probe import exact_recurrent_parity
        return exact_recurrent_parity(left,right,frames)
    import numpy as np
    from probe import initial_state
    states = [initial_state(left),initial_state(right)]
    stats = [{'max_abs':0.,'sum_squared_error':0.,'reference_energy':0.,'samples':0} for _ in range(2)]
    equal = True
    for frame in frames:
        outputs = []
        for index, model in enumerate((left,right)):
            output,states[index] = model.run(None,{'spec':frame,'state_in':states[index]})
            outputs.append(output)
            assert np.isfinite(output).all() and np.isfinite(states[index]).all()
        for part,a,b in zip(stats,(outputs[0],states[0]),(outputs[1],states[1])):
            equal &= a.tobytes()==b.tobytes()
            diff = a.astype(np.float64)-b
            part['max_abs'] = max(part['max_abs'],float(np.abs(diff).max()))
            part['sum_squared_error'] += float(np.sum(diff*diff))
            part['reference_energy'] += float(np.sum(a.astype(np.float64)**2))
            part['samples'] += a.size
    for part in stats:
        part['relative_rms'] = float(np.sqrt(part['sum_squared_error']/max(part['reference_energy'],1e-30)))
        part['rmse'] = float(np.sqrt(part['sum_squared_error']/part['samples']))
    return {'frames':len(frames),'output_and_state_bit_identical':bool(equal),'finite':True,
            'spectrum_error':stats[0],'state_error':stats[1],'requires_new_quality_evaluation':True}


def measure(args,variant):
    from extended_probe import ExtendedModel
    from probe import cpu_name,session,spectra
    from further_streaming_timing import run
    import os
    import platform
    driver_hash = sha(__file__)
    harness = ROOT/'scratch/oct7b/harness'/f'{driver_hash}.py'
    harness.parent.mkdir(parents=True,exist_ok=True)
    if not harness.exists():
        shutil.copyfile(__file__,harness)
    assert sha(harness)==driver_hash
    source = prepare(args.size,variant)
    pgo_identity = None
    for name in names_for(variant):
        mod = module_for(name)
        if name in getattr(mod,'PGO_VARIANTS',()):
            training = ROOT/f'results/oct7b_{args.size}_{variant}_training.json'
            trained = json.loads(training.read_text())
            assert trained['source_manifest']==manifest(source)
            counters = mod.profile_dir(ROOT/f'build/oct7b_{args.size}_{variant}')
            actual = {str(path.relative_to(counters)):sha(path) for path in sorted(counters.rglob('*.gcda'))}
            assert actual==trained['profiles'], 'PGO counters changed'
            cache = (ROOT/f'build/oct7b_{args.size}_{variant}/CMakeCache.txt').read_text()
            assert 'DPDF_OCT7B_PGO_MODE:STRING=USE' in cache
            pgo_identity = {'training_report':str(training),'training_sha256':sha(training),
                            'profile_directory':str(counters),'counter_sha256':actual}
    _, library = baseline_identity(args.size)
    reference = session(ROOT/f'models/dpdfnet{args.size}_48khz_hr.onnx')
    weights = ROOT/f'models/rework{args.size}/weights.f32'
    builds = {'baseline':library.parent,'candidate':ROOT/f'build/oct7b_{args.size}_{variant}'}
    models = {name:ExtendedModel(reference,4,8,7,build=build,weights=weights) for name,build in builds.items()}
    report = {'generated_at':datetime.now(timezone.utc).isoformat(),'driver_sha256':driver_hash,
              'driver_snapshot':str(harness),
              'pgo_identity':pgo_identity,
              'model_size':args.size,'variant':variant,'baseline':'Committed Oct7 combo_asm_norm',
              'approximate':approximate(variant),'method':'Preallocated single-thread C calls; balanced order; all samples retained.',
              'environment':{'cpu':cpu_name(),'platform':platform.platform(),'affinity':sorted(os.sched_getaffinity(0))},
              'model_sha256':sha(ROOT/f'models/dpdfnet{args.size}_48khz_hr.onnx'),'weights_sha256':sha(weights),
              'artifacts':{name:sha(build/'libdpdf_full.so') for name,build in builds.items()},
              'candidate_source_manifest':manifest(source),
              'owned_bytes':{name:model.owned_bytes for name,model in models.items()}}
    try:
        report['parity'] = parity(models['baseline'],models['candidate'],spectra(1000),report['approximate'])
        frames = spectra(args.frames+100)
        report['timed_frames_per_run'] = args.frames
        report['warmup'] = 100
        report['continuous'] = run(models,frames,args.repeats,False,100)
        report['paced'] = []
        report['standalone_paced'] = []
        if args.phase=='final':
            report['paced'] = run(models,frames,args.repeats,True,100)
            for repeat in range(args.repeats):
                order = list(models) if repeat%2==0 else list(reversed(models))
                for name in order:
                    item = run({name:models[name]},frames,1,True,100)[0]
                    item['repeat'] = repeat
                    report['standalone_paced'].append(item)
        report['summary'] = {}
        for mode in ('continuous','paced','standalone_paced'):
            if not report[mode]:
                continue
            values = {}
            for name in models:
                rows = [item['implementations'][name] for item in report[mode] if name in item['implementations']]
                values[name] = {key:median(item['wall'][key] for item in rows) for key in ('mean_ms','p50_ms','p99_ms')}
                values[name]['max_ms'] = max(item['wall']['max_ms'] for item in rows)
                values[name]['over_10ms'] = sum(item['wall']['over_10ms'] for item in rows)
                values[name]['process_cpu_ms'] = median(item['process_cpu']['mean_ms'] for item in rows)
                if mode!='continuous':
                    values[name]['late_completions'] = sum(item['completion_after_release']['over_10ms'] for item in rows)
            values['mean_reduction_percent'] = 100*(1-values['candidate']['mean_ms']/values['baseline']['mean_ms'])
            report['summary'][mode] = values
        assert sha(__file__)==driver_hash, 'Driver changed during measurement'
        assert report['artifacts']=={name:sha(build/'libdpdf_full.so') for name,build in builds.items()}
        assert manifest(source)==report['candidate_source_manifest']
        if pgo_identity:
            assert sha(pgo_identity['training_report'])==pgo_identity['training_sha256']
            folder = Path(pgo_identity['profile_directory'])
            assert pgo_identity['counter_sha256']=={str(path.relative_to(folder)):sha(path) for path in sorted(folder.rglob('*.gcda'))}
        destination = ROOT/f'results/oct7b_{args.size}_{variant}_{args.phase}.json'
        assert not destination.exists(), 'Saved timing results are immutable; use a fresh variant name'
        destination.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'model':args.size,'variant':variant,'parity':report['parity'],'summary':report['summary']}),flush=True)
    finally:
        for model in models.values():
            model.close()


def profile(size,frames):
    import ctypes as ct
    import numpy as np
    from extended_probe import ExtendedModel
    from probe import initial_state,session,spectra
    build(size,'profile')
    _, baseline = baseline_identity(size)
    source = ROOT/f'scratch/oct7b/{size}/profile'
    target = ROOT/f'build/oct7b_{size}_profile'
    reference = session(ROOT/f'models/dpdfnet{size}_48khz_hr.onnx')
    weights = ROOT/f'models/rework{size}/weights.f32'
    models = [ExtendedModel(reference,4,8,7,build=folder,weights=weights) for folder in (baseline.parent,target)]
    try:
        match = parity(*models,spectra(1000),False)
        model = models[1]
        state = initial_state(model)
        stream = spectra(frames+100)
        for frame in stream[:100]:
            _,state = model.run(None,{'spec':frame,'state_in':state})
        model.lib.oct7b_profile_reset()
        for frame in stream[100:]:
            _,state = model.run(None,{'spec':frame,'state_in':state})
        nodes = json.loads((source/'profile_nodes.json').read_text())
        times = np.zeros(max(n['index'] for n in nodes)+1,np.float64)
        model.lib.oct7b_profile_get.argtypes = [ct.POINTER(ct.c_double)]
        model.lib.oct7b_profile_get(times.ctypes.data_as(ct.POINTER(ct.c_double)))
        categories = {}
        for node in nodes:
            node['diagnostic_ms_per_hop'] = float(times[node['index']]/frames/1e6)
            op = node['operation']
            category = ('DPRNN' if 'dprnn' in op.lower() else 'dense' if op in ('Gemm','MatMul') else
                        'convolution' if op=='Conv' else 'scalar_nonlinear' if op in ('Sigmoid','Tanh') else 'other')
            categories[category] = categories.get(category,0)+node['diagnostic_ms_per_hop']
        report = {'model_size':size,'frames':frames,'parity':match,
                  'warning':'Instrumented diagnostic, not a latency benchmark; timers and compiler scheduling affect attribution.',
                  'baseline_sha256':sha(baseline),'profile_sha256':sha(target/'libdpdf_full.so'),
                  'source_manifest':manifest(source),'categories_ms_per_hop':categories,
                  'nodes':sorted(nodes,key=lambda n:-n['diagnostic_ms_per_hop'])}
        (ROOT/f'results/oct7b_{size}_profile.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'model':size,'categories':categories,'top_nodes':report['nodes'][:10]}),flush=True)
    finally:
        for model in models:
            model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('build','screen','final','asan','scalar','profile'))
    p.add_argument('variants',nargs='*')
    p.add_argument('--size',type=int,choices=(2,8),default=8)
    p.add_argument('--frames',type=int,default=600)
    p.add_argument('--repeats',type=int,default=3)
    args = p.parse_args()
    if args.frames<=0 or args.repeats<=0:
        p.error('Positive frames/repeats required')
    if args.phase=='profile':
        profile(args.size,args.frames)
        return
    if not args.variants:
        p.error('At least one variant required')
    for variant in args.variants:
        if args.phase in ('build','asan','scalar'):
            build(args.size,variant,{'asan':'_asan','scalar':'_scalar'}.get(args.phase,''))
        else:
            assert variant!='profile', 'Never benchmark an instrumented profile'
            measure(args,variant)


if __name__=='__main__':
    main()
