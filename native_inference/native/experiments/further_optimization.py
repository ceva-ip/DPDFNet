"""Reproduce the accepted single-thread, exact-output W7A8 optimization.

Run from native_inference in the existing Linux development environment.
Production kernels, generated distribution files and old binaries are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'scratch/further_optimization'
VARIANTS = ('baseline', 'best_single')


def source_manifest(source, name):
    return {'variant': name, 'baseline_profile': 'checked-in fitted W7A8',
            'inference_threads': 1, 'scope': 'C/header/CMake source files',
            'files': {str(path.relative_to(source)).replace('\\', '/'):
                      hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in source.rglob('*') if path.is_file() and
                      (path.suffix in ('.c', '.h', '.cc') or path.name == 'CMakeLists.txt')}}


def logged(command, name):
    WORK.mkdir(parents=True, exist_ok=True)
    log = WORK / (name + '.log')
    with log.open('w') as output:
        result = subprocess.run(list(map(str, command)), stdout=output,
                                stderr=subprocess.STDOUT, cwd=ROOT)
    if result.returncode:
        print(log.read_text()[-6000:], flush=True)
        raise RuntimeError(f'{command[0]} failed; see {log}')
    print(f'{name}: complete', flush=True)


def prepare(name):
    source = WORK / name
    if (source / 'source_manifest.json').exists():
        raise RuntimeError(f'Refusing to overwrite preserved source: {source}')
    shutil.copytree(ROOT / 'native', source,
                    ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=False)
    for filename in ('int8.c', 'avx2.c'):
        shutil.copyfile(ROOT / 'native/integration/w7a8' / filename, source / filename)
    generated = (ROOT / 'models/rework8/generated_model.c').read_text()
    if name == 'best_single':
        from further_exact_kernels import optimize
        optimize(source)
    (source / 'generated_model.c').write_text(generated)
    manifest = source_manifest(source, name)
    (source / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return source


def build(name, sanitizer=False, scalar=False):
    source = WORK / name
    if not source.exists():
        source = prepare(name)
    # The stock scalar oracle describes W8A8. Use its established W7A8 grid
    # adaptation, retaining independent scalar arithmetic and batch checks.
    from prepare_int8_variant import oracle
    (source / 'int8_contract.c').write_text(oracle((ROOT / 'native/int8_contract.c').read_text(), 'w7'))
    cmake = (source / 'CMakeLists.txt').read_text().replace(
        '${CMAKE_CURRENT_SOURCE_DIR}/../models/native_blocks/erb_0.f32',
        (ROOT / 'models/native_blocks/erb_0.f32').as_posix())
    (source / 'CMakeLists.txt').write_text(cmake)
    manifest = source_manifest(source, name)
    (source / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    suffix = '_asan' if sanitizer else '_scalar' if scalar else ''
    target = ROOT / 'build' / ('further_' + name + suffix)
    options = ['-DDPDF_EXTENDED_MODEL=ON', '-DCMAKE_BUILD_TYPE=Release',
               f'-DDPDF_GENERATED_MODEL={source}/generated_model.c',
               f'-DDPDF_TEST_WEIGHTS={ROOT}/models/rework8/weights.f32']
    if sanitizer:
        options += ['-DDPDF_SANITIZE=ON', '-DCMAKE_EXE_LINKER_FLAGS=-no-pie']
    if scalar:
        options += ['-DDPDF_ENABLE_AVX2=OFF']
    logged(['cmake', '-S', source, '-B', target, *options], name + suffix + '_configure')
    logged(['cmake', '--build', target, '-j4'], name + suffix + '_build')
    logged(['ctest', '--test-dir', target, '--output-on-failure'], name + suffix + '_contracts')
    return target


def timing(name, final=False):
    output = ROOT / 'results' / ('further_' + name + ('_final' if final else '_screen') + '.json')
    probe = ROOT / ('native/experiments/further_streaming_timing.py' if final else 'native/latency_probe.py')
    logged([sys.executable, probe,
            '--model', ROOT / 'models/dpdfnet8_48khz_hr.onnx',
            '--weights', ROOT / 'models/rework8/weights.f32',
            '--baseline-build', ROOT / 'build/further_baseline',
            '--candidate-build', ROOT / 'build' / ('further_' + name),
            '--frames', '1000' if final else '600', '--repeats', '4' if final else '3',
            '--paced-repeats', '4' if final else '0',
            '--standalone-paced-repeats', '4' if final else '0', '--output', output],
           name + ('_final' if final else '_screen'))
    report = json.loads(output.read_text())
    for mode in ('continuous', 'paced', 'standalone_paced'):
        rows = report.get(mode, [])
        if not rows:
            continue
        means = {implementation: median(row['implementations'][implementation]['wall']['mean_ms']
                                         for row in rows if implementation in row['implementations'])
                 for implementation in ('baseline', 'candidate')}
        print(json.dumps({'variant': name, 'mode': mode, 'median_run_mean_ms': means,
                          'reduction_percent': 100 * (1 - means['candidate'] / means['baseline'])}), flush=True)


def memory(names):
    report = {'method': 'Four fresh processes per variant; median warmed incremental RSS; source weights unmapped; 120 warmup hops.',
              'variants': {}}
    for name in names:
        target = ROOT / 'build' / ('further_' + name)
        runs = []
        for _ in range(4):
            result = subprocess.check_output([sys.executable, str(ROOT / 'native/onnx_memory_probe.py'),
                      '--model', str(ROOT / 'models/dpdfnet8_48khz_hr.onnx'),
                      '--weights', str(ROOT / 'models/rework8/weights.f32'), '--build', str(target),
                      '--variant', 'selective_int8'], text=True, cwd=ROOT)
            runs.append(json.loads(result))
        item = {'runs': runs, 'median_incremental_rss_bytes': median(run['rss_delta_bytes'] for run in runs),
                'owned_bytes': runs[0]['owned_bytes'],
                'library_sha256': hashlib.sha256((target / 'libdpdf_full.so').read_bytes()).hexdigest()}
        report['variants'][name] = item
        print(json.dumps({'variant': name, **{key: value for key, value in item.items() if key != 'runs'}}), flush=True)
    (ROOT / 'results/further_memory.json').write_text(json.dumps(report, indent=2) + '\n')


def summary(names):
    memory_report = json.loads((ROOT / 'results/further_memory.json').read_text())
    fp_report = json.loads((ROOT / 'results/further_selected_fp_environment.json').read_text())
    if not fp_report['passed'] or fp_report.get('observed_process_thread_counts') != [1]:
        raise AssertionError('W7A8 FP controls and single-thread verification have not passed')
    report = {'reference': 'Current fitted W7A8/pack32 dpdfnet8_48khz_hr; no additional quantization or activation approximation.',
              'inference_threads': 1,
              'fp_controls': {'report': 'results/further_selected_fp_environment.json',
                              'frames_per_mode': fp_report['frames_per_mode'],
                              'modes': len(fp_report['modes']), 'observed_process_threads': [1], 'passed': True},
              'candidates': {}, 'screening': {}}
    for path in [ROOT / 'results/further_best_single_screen.json']:
        data = json.loads(path.read_text())
        means = {name: median(row['implementations'][name]['wall']['mean_ms'] for row in data['continuous'])
                 for name in ('baseline', 'candidate')}
        report['screening'][path.stem.removeprefix('further_').removesuffix('_screen')] = {
            'median_run_mean_ms': means, 'mean_reduction_percent': 100 * (1 - means['candidate'] / means['baseline']),
            'recurrent_parity': data['parity'], 'library_sha256': data['artifacts']}
    for name in names:
        quality_path = ROOT / 'results' / ('further_' + name + '_audio.json')
        timing_path = ROOT / 'results' / ('further_' + name + '_final.json')
        quality = json.loads(quality_path.read_text())
        timing = json.loads(timing_path.read_text())
        if quality['library_sha256'] != timing['artifacts']:
            raise AssertionError(f'{name}: quality/timing libraries differ')
        if (fp_report['builds']['candidate0']['sha256'] != timing['artifacts']['candidate'] or
                fp_report['builds']['baseline']['sha256'] != timing['artifacts']['baseline']):
            raise AssertionError(f'{name}: FP controls and timing libraries differ')
        if not quality['all_selected_cases_bit_identical'] or not quality['full_available_suite']:
            raise AssertionError(f'{name}: full quality suite has not passed')
        contracts = {}
        for suffix in ('', '_asan', '_scalar'):
            path = WORK / (name + suffix + '_contracts.log')
            if '100% tests passed' not in path.read_text():
                raise AssertionError(f'{name}{suffix}: C contracts have not passed')
            contracts[suffix or 'release'] = {'log': str(path.relative_to(ROOT)),
                'log_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'output': path.read_text()}
        modes = {}
        for mode in ('continuous', 'paced', 'standalone_paced'):
            modes[mode] = {}
            for implementation in ('baseline', 'candidate'):
                rows = [row['implementations'][implementation] for row in timing[mode]
                        if implementation in row['implementations']]
                values = {metric: median(row['wall'][metric] for row in rows)
                          for metric in ('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms')}
                values.update(max_ms=max(row['wall']['max_ms'] for row in rows),
                    over_10ms=sum(row['wall']['over_10ms'] for row in rows),
                    median_process_cpu_ms=median(row['process_cpu']['mean_ms'] for row in rows))
                if mode != 'continuous':
                    values['completion_over_10ms'] = sum(row['completion_after_release']['over_10ms'] for row in rows)
                    values['completion_max_ms'] = max(row['completion_after_release']['max_ms'] for row in rows)
                modes[mode][implementation] = values
            modes[mode]['mean_reduction_percent'] = 100 * (1 - modes[mode]['candidate']['mean_ms'] /
                                                        modes[mode]['baseline']['mean_ms'])
        report['candidates'][name] = {'quality_cases': quality['selected_cases'],
            'quality_audio_seconds': quality['total_audio_seconds'],
            'quality_audio_hops': quality['total_audio_hops_including_flush'],
            'byte_identical': True, 'timing': modes, 'memory': memory_report['variants'][name],
            'baseline_memory': memory_report['variants']['baseline'], 'contracts': contracts,
            'library_sha256': timing['artifacts'], 'environment': timing['environment'],
            'quality_report': str(quality_path.relative_to(ROOT)), 'timing_report': str(timing_path.relative_to(ROOT))}
    (ROOT / 'results/further_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({name: candidate['timing'] for name, candidate in report['candidates'].items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('build', 'screen', 'final', 'asan', 'scalar', 'memory', 'summary'))
    parser.add_argument('variants', nargs='+', choices=VARIANTS)
    args = parser.parse_args()
    if args.phase == 'memory':
        memory(args.variants)
        return
    if args.phase == 'summary':
        summary(args.variants)
        return
    for name in args.variants:
        if args.phase in ('build', 'asan', 'scalar'):
            build(name, args.phase == 'asan', args.phase == 'scalar')
        else:
            timing(name, args.phase == 'final')


if __name__ == '__main__':
    main()
