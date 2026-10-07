"""Matched fresh-process RSS for Oct3 versus the selected Oct7 exact builds.

Run from native_inference in the existing Linux development container, after
all timing jobs finish. No build, correctness test, or latency benchmark is
performed here. Each of eight sequential fresh children uses the unchanged
onnx_memory_probe.py protocol: common imported-runtime baseline, native model
creation, source-weight unmapping, 120 warmup hops, then process RSS sampling.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / 'native/onnx_memory_probe.py'
SELECTED = 'combo_asm_norm'
REPEATS = 4


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def local_dependencies(entry):
    """Hash the probe's statically imported repository Python dependency closure."""
    pending = [Path(entry)]
    visited = set()
    while pending:
        path = pending.pop().resolve()
        if path in visited:
            continue
        visited.add(path)
        syntax = ast.parse(path.read_text())
        for node in ast.walk(syntax):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                module_path = Path(*module.split('.')).with_suffix('.py')
                for directory in (path.parent, ROOT/'native', ROOT):
                    dependency = directory/module_path
                    if dependency.is_file() and dependency.resolve().is_relative_to(ROOT):
                        pending.append(dependency)
                        break
    return {relative(path): sha(path) for path in sorted(visited)}


def source_identity(source):
    manifest_path = source/'source_manifest.json'
    saved = json.loads(manifest_path.read_text())
    declared = saved.get('files', saved)
    if not isinstance(declared, dict) or not declared:
        raise ValueError(f'Invalid source manifest: {manifest_path}')
    for name, expected in declared.items():
        path = source/name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f'Preserved source differs from manifest: {path}')
    # Top-level C/header/assembly/CMake files are the compiled snapshot. The
    # legacy manifest may additionally cover nested preserved research files.
    compiled = {path.name: sha(path) for path in sorted(source.iterdir())
                if path.is_file() and
                (path.suffix in ('.c', '.h', '.S') or path.name=='CMakeLists.txt')}
    return {'directory': relative(source), 'source_manifest_sha256': sha(manifest_path),
            'compiled_source_sha256': compiled,
            'validated_declared_source_sha256': declared}


def package_identity(name):
    """Record versions, package entry points and native binary dependencies."""
    spec = util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f'Missing memory-probe dependency: {name}')
    entry = Path(spec.origin).resolve()
    native_files = sorted(path for path in entry.parent.rglob('*') if path.is_file()
                          and (path.suffix in ('.so', '.pyd', '.dll') or '.so.' in path.name))
    return {'version': metadata.version(name), 'entry_point': str(entry),
            'entry_point_sha256': sha(entry),
            'native_binary_sha256': {str(path.relative_to(entry.parent)).replace('\\', '/'):
                                     sha(path) for path in native_files}}


def artifacts(size):
    baseline_name = 'further2_best_single' if size==2 else 'further_best_single'
    baseline_source = ROOT/('scratch/further_optimization2' if size==2
                            else 'scratch/further_optimization')/'best_single'
    candidate_source = ROOT/f'scratch/oct7/{size}/{SELECTED}'
    model = ROOT/f'models/dpdfnet{size}_48khz_hr.onnx'
    weights = ROOT/f'models/rework{size}/weights.f32'
    builds = {'baseline': ROOT/'build'/baseline_name,
              'candidate': ROOT/f'build/oct7_{size}_{SELECTED}'}
    sources = {'baseline': baseline_source, 'candidate': candidate_source}
    return {
        'model': {'path': relative(model), 'sha256': sha(model)},
        'weights': {'path': relative(weights), 'sha256': sha(weights)},
        'libraries': {name: {'path': relative(build/'libdpdf_full.so'),
                            'sha256': sha(build/'libdpdf_full.so')}
                      for name, build in builds.items()},
        'sources': {name: source_identity(source) for name, source in sources.items()},
    }, model, weights, builds


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(2, 8), default=8)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('The unchanged memory probe reads Linux /proc/self/status')
    output = args.output or ROOT/f'results/oct7_{args.size}_memory.json'
    if output.exists() and not args.overwrite:
        parser.error(f'Refusing to replace saved result: {output}; use --output or --overwrite')
    identity, model, weights, builds = artifacts(args.size)
    dependencies = local_dependencies(PROBE)
    child_env = dict(os.environ)
    thread_controls = {name: '1' for name in
                       ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'NUMEXPR_NUM_THREADS')}
    child_env.update(thread_controls)
    report = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'model_size': args.size,
        'model': f'dpdfnet{args.size}_48khz_hr',
        'baseline': 'Accepted Oct3 best_single',
        'candidate': SELECTED,
        'method': ('Four fresh processes per variant; median warmed incremental RSS; '
                   'source weights unmapped; 120 warmup hops. Balanced alternating '
                   'baseline/candidate order; every child completes before the next starts.'),
        'protocol': {
            'fresh_processes_per_variant': REPEATS, 'warmup_hops': 120,
            'native_configuration': {'variant': 'selective_int8', 'tier': 4,
                                     'precision': 8, 'mask': 7},
            'rss_source': 'Linux /proc/self/status VmRSS, kB converted to bytes',
            'incremental_rss': ('Warmed RSS minus RSS after identical Python/NumPy/ORT '
                                'imports and gc.collect(), before native model loading.'),
            'included': ['native model', 'native library', 'stream buffers',
                         'allocator retention above the common imported baseline'],
            'source_weights': 'Read-only mmap closed before warmup and final RSS sample',
            'onnx_reference_session_created': False,
            'owned_bytes': 'Exact native allocation accounting; distinct from process RSS',
            'thread_environment': thread_controls,
            'order': [['baseline', 'candidate'] if repeat%2==0
                      else ['candidate', 'baseline'] for repeat in range(REPEATS)],
            'no_outliers_removed': True,
            'comparability': 'Same unchanged memory worker and metric definition as Oct3.',
        },
        'driver': {'path': relative(__file__), 'sha256': sha(__file__)},
        'dependency_sha256': dependencies,
        'dependency_packages': {name: package_identity(name)
                                for name in ('numpy', 'onnxruntime')},
        'environment': {'platform': platform.platform(), 'python_version': sys.version,
                        'python_executable': sys.executable,
                        'python_executable_sha256': sha(sys.executable),
                        'affinity': sorted(os.sched_getaffinity(0))},
        'artifacts': identity,
        'variants': {name: {'runs': [],
                           'library_sha256': identity['libraries'][name]['sha256']}
                     for name in builds},
        'raw_processes': [], 'completed': False,
    }
    try:
        for repeat in range(REPEATS):
            order = ['baseline', 'candidate'] if repeat%2==0 else ['candidate', 'baseline']
            for name in order:
                command = [sys.executable, str(PROBE), '--model', str(model),
                           '--weights', str(weights), '--build', str(builds[name]),
                           '--variant', 'selective_int8']
                raw = {'sequence': len(report['raw_processes']), 'repeat': repeat,
                       'variant': name, 'command': command,
                       'started_at': datetime.now(timezone.utc).isoformat()}
                report['raw_processes'].append(raw)
                try:
                    process = subprocess.run(command, cwd=ROOT, env=child_env,
                                             capture_output=True, text=True, timeout=60)
                except subprocess.TimeoutExpired as error:
                    raw.update(timeout_seconds=60, stdout=str(error.stdout or ''),
                               stderr=str(error.stderr or ''))
                    raise RuntimeError(f'{name} RSS worker timed out') from error
                raw.update(returncode=process.returncode, stdout=process.stdout,
                           stderr=process.stderr,
                           finished_at=datetime.now(timezone.utc).isoformat())
                if process.returncode:
                    raise RuntimeError(f'{name} RSS worker returned {process.returncode}')
                measurement = json.loads(process.stdout)
                required = ('rss_before_bytes', 'rss_after_bytes', 'rss_delta_bytes',
                            'peak_rss_bytes', 'owned_bytes')
                if any(type(measurement.get(key)) is not int for key in required):
                    raise ValueError(f'{name} RSS worker returned invalid byte measurements')
                if measurement['rss_after_bytes']-measurement['rss_before_bytes'] != measurement['rss_delta_bytes']:
                    raise ValueError(f'{name} inconsistent incremental RSS')
                raw['measurement'] = measurement
                report['variants'][name]['runs'].append(measurement)
                print(json.dumps({'repeat': repeat, 'variant': name, **measurement}), flush=True)
        final_identity, _, _, _ = artifacts(args.size)
        if final_identity != identity or local_dependencies(PROBE) != dependencies or sha(__file__) != report['driver']['sha256']:
            raise RuntimeError('Sources, dependencies or measured artifacts changed during memory sampling')
        for name, item in report['variants'].items():
            if len(item['runs']) != REPEATS:
                raise ValueError(f'{name}: incomplete fresh-process repetitions')
            owned = {run['owned_bytes'] for run in item['runs']}
            if len(owned) != 1:
                raise ValueError(f'{name}: native allocation accounting changed across processes')
            # Match Oct3 naming: runs, median_incremental_rss_bytes,
            # owned_bytes and library_sha256 are preserved in each variant.
            item['median_incremental_rss_bytes'] = median(run['rss_delta_bytes'] for run in item['runs'])
            item['owned_bytes'] = owned.pop()
        report['completed'] = True
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        write_report(output, report)
        raise
    write_report(output, report)
    print(json.dumps({'output': str(output), 'variants': {
        name: {key: value for key, value in item.items() if key != 'runs'}
        for name, item in report['variants'].items()}}), flush=True)


if __name__ == '__main__':
    main()
