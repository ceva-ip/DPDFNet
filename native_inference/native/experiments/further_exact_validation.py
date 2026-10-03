"""Byte-exact live-baseline regression on the saved 48 kHz evaluation suite.

Run from native_inference in the Linux benchmark image. This uses the existing
causal FFT, inverse FFT, overlap-add and 2,400-sample alignment unchanged. Every
spectral output and every recurrent state is compared before PCM synthesis.
Quality scorers are unnecessary when the output waveform is byte-identical.
Correctness jobs must not overlap latency measurements.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native'))
from extended_probe import ExtendedModel
from probe import audio_spectra, session, spectra, synthesize
from streaming_runner import StreamingRunner

CONFIG = (4, 8, 7)
RATE = 48000
DELAY = 2400


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def byte_view(array):
    return memoryview(np.ascontiguousarray(array)).cast('B')


def pcm_sha(array):
    return hashlib.sha256(byte_view(array.astype('<f4', copy=False))).hexdigest()


def exact(left, right, label):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise AssertionError(f'{label}: shape or dtype differs')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise AssertionError(f'{label}: nonfinite value')
    # The integer view distinguishes positive/negative zero, unlike float ==.
    if not np.array_equal(left.view(np.uint32), right.view(np.uint32)):
        raise AssertionError(f'{label}: bytes differ')


def source_identity(build, override=None):
    values = {}
    cache = build / 'CMakeCache.txt'
    if cache.exists():
        for line in cache.read_text().splitlines():
            if ':' in line and '=' in line and not line.startswith('//'):
                key, value = line.split('=', 1)
                values[key.split(':', 1)[0]] = value
    source = Path(override) if override else Path(values.get('CMAKE_HOME_DIRECTORY', ''))
    result = {'source_directory': str(source), 'source_files_sha256': {}}
    if (override or values.get('CMAKE_HOME_DIRECTORY')) and source.is_dir():
        for path in sorted(source.rglob('*')):
            if path.is_file() and (path.suffix in ('.c', '.h', '.cc') or path.name == 'CMakeLists.txt'):
                result['source_files_sha256'][str(path.relative_to(source))] = sha(path)
    generated = values.get('DPDF_GENERATED_MODEL')
    if generated and Path(generated).is_file():
        result['generated_model'] = {'path': generated, 'sha256': sha(generated)}
    return result


def initialize(model_path, weights, baseline, candidate):
    global REFERENCE, MODELS, RUNNERS
    REFERENCE = session(model_path)
    MODELS = {name: ExtendedModel(REFERENCE, *CONFIG, build=Path(build), weights=Path(weights))
              for name, build in [('baseline', baseline), ('candidate', candidate)]}
    RUNNERS = {name: StreamingRunner(model) for name, model in MODELS.items()}


def compare_frames(frames, label, retain_pcm=False):
    for runner in RUNNERS.values():
        runner.reset()
    exact(RUNNERS['baseline'].state, RUNNERS['candidate'].state, f'{label}: initial state')
    digests = {name: {'spectrum': hashlib.sha256(), 'state': hashlib.sha256()}
               for name in RUNNERS}
    outputs = {name: np.empty_like(frames) for name in RUNNERS} if retain_pcm else None
    prefix = {name: hashlib.sha256() for name in RUNNERS}
    replay_frames = min(8, len(frames))
    for index, frame in enumerate(frames):
        left = RUNNERS['baseline'].process(frame)
        right = RUNNERS['candidate'].process(frame)
        exact(left, right, f'{label}: spectrum at hop {index}')
        exact(RUNNERS['baseline'].state, RUNNERS['candidate'].state,
              f'{label}: recurrent state at hop {index}')
        for name, runner in RUNNERS.items():
            digests[name]['spectrum'].update(byte_view(runner.output))
            digests[name]['state'].update(byte_view(runner.state))
            if index < replay_frames:
                prefix[name].update(byte_view(runner.output))
                prefix[name].update(byte_view(runner.state))
            if outputs is not None:
                outputs[name][index] = runner.output
    for name, runner in RUNNERS.items():
        runner.reset()
        replay = hashlib.sha256()
        for frame in frames[:replay_frames]:
            replay.update(byte_view(runner.process(frame)))
            replay.update(byte_view(runner.state))
        if replay.digest() != prefix[name].digest():
            raise AssertionError(f'{label}: {name} reset replay differs')
    result = {'hops_including_flush': len(frames),
              'every_output_and_state_bit_identical': True, 'all_values_finite': True,
              'reset_replay_hops': replay_frames, 'reset_replay_bit_identical': True,
              'stream_sha256': {name: {kind: digest.hexdigest() for kind, digest in values.items()}
                                for name, values in digests.items()}}
    return result, outputs


def run_case(case):
    start = time.monotonic()
    path = Path(case['input'])
    input_hash = sha(path)
    if input_hash != case['expected_input_sha256']:
        raise AssertionError(f"{case['id']}: input fixture hash differs")
    info = sf.info(path)
    if info.samplerate != RATE or info.channels != 1 or info.frames != case['samples']:
        raise AssertionError(f"{case['id']}: fixture shape/rate differs")
    frames = audio_spectra(path)
    result, outputs = compare_frames(frames, case['id'], retain_pcm=True)
    waveforms = {name: synthesize(value)[DELAY:DELAY + info.frames]
                 for name, value in outputs.items()}
    exact(waveforms['baseline'], waveforms['candidate'], f"{case['id']}: aligned PCM")
    if any(value.shape != (info.frames,) for value in waveforms.values()):
        raise AssertionError(f"{case['id']}: output length differs")
    result.update({'case': case['id'], 'scenario': case['scenario'], 'input': str(path),
                   'samples': info.frames, 'duration_seconds': info.frames / RATE,
                   'input_sha256': input_hash, 'pcm_bit_identical': True,
                   'pcm_sha256': {name: pcm_sha(value) for name, value in waveforms.items()},
                   'elapsed_seconds': time.monotonic() - start})
    if case.get('saved_baseline_pcm_sha256'):
        saved = case['saved_baseline_pcm_sha256']
        result['saved_pack_fit5_pcm_sha256'] = saved
        result['saved_pack_fit5_pcm_bit_identical'] = saved == result['pcm_sha256']['baseline']
        if case.get('require_saved_baseline_hash') and not result['saved_pack_fit5_pcm_bit_identical']:
            raise AssertionError(f"{case['id']}: baseline differs from its saved PCM hash")
    return result


def load_cases(data, fixture_report, baseline_audio_report, baseline_hash):
    # This old exact report supplies checksummed inputs for the entire suite;
    # its old output waveform is deliberately not the live baseline here.
    report = json.loads(fixture_report.read_text())
    fixture_hashes = {item['case']['id']: item for item in report['cases']}
    if len(fixture_hashes) != len(report['cases']):
        raise AssertionError('Saved fixture report contains duplicate case IDs')
    manifest = json.loads((data / 'evaluation/manifest.json').read_text())
    jobs = []
    for clip in manifest['clips']:
        jobs.append({'id': 'mixture_' + clip['id'], 'scenario': 'mixture',
                     'input': str(data / clip['noisy'])})
    robust = json.loads((data / 'robustness/manifest.json').read_text())
    for case in robust['cases']:
        identifier = case['case_id']
        jobs.append({'id': identifier, 'scenario': case['scenario'],
                     'input': str(data / 'robustness/audio' / identifier / 'input.wav')})
    for name in ('white', 'pink', 'mechanical'):
        jobs.append({'id': name, 'scenario': 'long_noise',
                     'input': str(data / 'long_noise' / (name + '.wav'))})
    identifiers = {job['id'] for job in jobs}
    if len(identifiers) != len(jobs):
        raise AssertionError('Evaluation manifests contain duplicate case IDs')
    if identifiers != set(fixture_hashes):
        raise AssertionError('Evaluation manifests differ from the saved full fixture suite')
    saved = {}
    saved_library_matches = False
    if baseline_audio_report.exists():
        old = json.loads(baseline_audio_report.read_text())
        saved = {item['case']['id']: item['candidate_pcm_sha256'] for item in old['cases']}
        saved_library_matches = old['candidate_sha256'] == baseline_hash
    for job in jobs:
        prior = fixture_hashes[job['id']]
        job.update({'expected_input_sha256': prior['input_sha256'], 'samples': prior['samples']})
        if job['id'] in saved:
            job.update({'saved_baseline_pcm_sha256': saved[job['id']],
                        'require_saved_baseline_hash': saved_library_matches})
    return jobs, manifest['provenance'], saved_library_matches


def concurrent_contexts(model_path, weights, candidate_build, frames):
    reference = session(model_path)

    def stream(index):
        model = ExtendedModel(reference, *CONFIG, build=candidate_build, weights=weights)
        runner = StreamingRunner(model)
        digest = hashlib.sha256()
        try:
            for frame in frames:
                value = (frame * np.float32((index + 1) / 4)).astype(np.float32)
                output = runner.process(value)
                if not np.isfinite(output).all() or not np.isfinite(runner.state).all():
                    raise AssertionError('Nonfinite concurrent output/state')
                digest.update(byte_view(output))
                digest.update(byte_view(runner.state))
            return digest.hexdigest()
        finally:
            model.close()

    serial = [stream(index) for index in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(stream, range(4)))
    if serial != concurrent:
        raise AssertionError('Independent contexts change under concurrent calls')
    return {'streams': 4, 'hops_per_stream': len(frames),
            'serial_and_concurrent_output_and_state_bit_identical': True,
            'output_and_state_sha256': concurrent}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/dpdfnet8_48khz_hr.onnx')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/rework8/weights.f32')
    parser.add_argument('--baseline-build', type=Path, required=True)
    parser.add_argument('--candidate-build', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path)
    parser.add_argument('--candidate-source', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / 'scratch/fullband')
    parser.add_argument('--fixture-report', type=Path, default=ROOT / 'results/w7_followup_pack32_audio.json')
    parser.add_argument('--baseline-audio-report', type=Path, default=ROOT / 'results/w7_followup_pack_fit5_audio.json')
    parser.add_argument('--model-quality-report', type=Path, help='Saved per-model scored reference for model/weight provenance and PCM hashes')
    parser.add_argument('--limit', type=int, default=0, help='First N selected cases; zero checks the whole suite')
    parser.add_argument('--cases', nargs='+', help='Exact case IDs, e.g. mixture_00033 clean_00033 pink')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--synthetic-frames', type=int, default=1000)
    parser.add_argument('--concurrency-frames', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.limit, args.synthetic_frames, args.concurrency_frames) < 0 or args.workers < 1:
        parser.error('Counts must be nonnegative and workers positive')
    start = time.monotonic()
    libraries = {name: sha(build / 'libdpdf_full.so') for name, build in
                 [('baseline', args.baseline_build), ('candidate', args.candidate_build)]}
    jobs, provenance, saved_library_matches = load_cases(
        args.data, args.fixture_report, args.baseline_audio_report, libraries['baseline'])
    if args.model_quality_report:
        scored = json.loads(args.model_quality_report.read_text())
        artifacts = scored['artifacts']
        provenance = {key: artifacts[str(path.relative_to(ROOT))] for key, path in
                      [('model', args.model), ('weights', args.weights)]}
        saved = {'mixture_' + clip['clip']['id']: clip['systems']['w7a8_fit5']['pcm_sha256']
                 for clip in scored['clips']}
        for job in jobs:
            job.pop('saved_baseline_pcm_sha256', None)
            job.pop('require_saved_baseline_hash', None)
            if job['id'] in saved:
                job.update(saved_baseline_pcm_sha256=saved[job['id']], require_saved_baseline_hash=True)
        saved_library_matches = False  # PCM hashes are verified regardless of compiler identity.
    available_count = len(jobs)
    if args.cases:
        missing = set(args.cases) - {job['id'] for job in jobs}
        if missing:
            parser.error(f'Unknown case IDs: {sorted(missing)}')
        jobs = [job for job in jobs if job['id'] in args.cases]
    if args.limit:
        jobs = jobs[:args.limit]
    for key, path in [('model', args.model), ('weights', args.weights)]:
        if sha(path) != provenance[key]:
            raise AssertionError(f'{key} differs from saved evaluation provenance')
    report = {'mode': 'live baseline versus candidate: exact spectra, recurrent states and aligned PCM',
              'sample_rate': RATE, 'alignment_delay_samples': DELAY,
              'config': list(CONFIG),
              'model_sha256': sha(args.model), 'weights_sha256': sha(args.weights),
              'library_sha256': libraries, 'builds': {name: str(build) for name, build in
                  [('baseline', args.baseline_build), ('candidate', args.candidate_build)]},
              'source_identity': {name: source_identity(build, source) for name, build, source in
                  [('baseline', args.baseline_build, args.baseline_source),
                   ('candidate', args.candidate_build, args.candidate_source)]},
              'harness_sha256': sha(__file__),
              'dependency_code_sha256': {name: sha(ROOT / 'native' / name) for name in
                  ('probe.py', 'streaming_runner.py', 'full_probe.py', 'extended_probe.py')},
              'packages': {name: importlib.metadata.version(name) for name in
                  ('numpy', 'soundfile', 'onnxruntime')},
              'fixture_report_sha256': sha(args.fixture_report),
              'model_quality_report_sha256': sha(args.model_quality_report) if args.model_quality_report else None,
              'saved_pack_fit5_library_matches_baseline': saved_library_matches,
              'available_cases': available_count, 'selected_cases': len(jobs),
              'full_available_suite': len(jobs) == available_count, 'cases': []}
    initargs = tuple(map(str, (args.model, args.weights, args.baseline_build, args.candidate_build)))
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=initargs) as pool:
        for future in as_completed([pool.submit(run_case, job) for job in jobs]):
            item = future.result()
            report['cases'].append(item)
            print(f"{len(report['cases'])}/{len(jobs)} {item['case']}: exact output, state and PCM", flush=True)
    report['cases'].sort(key=lambda item: item['case'])
    report['total_audio_seconds'] = sum(item['duration_seconds'] for item in report['cases'])
    report['total_audio_hops_including_flush'] = sum(item['hops_including_flush'] for item in report['cases'])
    if args.synthetic_frames:
        initialize(*initargs)
        try:
            frames = spectra(args.synthetic_frames)
            # Additional quiet/high-amplitude/silence transitions exercise the
            # same recurrent stream; this is a correctness stress, not speech.
            quarter = len(frames) // 4
            frames[:quarter] *= np.float32(1e-3)
            frames[2 * quarter:3 * quarter] *= np.float32(10)
            frames[-min(64, max(1, len(frames) // 8)):] = 0
            report['synthetic'], _ = compare_frames(frames, 'synthetic')
        finally:
            for model in MODELS.values():
                model.close()
        print(f'{args.synthetic_frames} synthetic recurrent hops: exact output and state', flush=True)
    if args.concurrency_frames:
        report['concurrency'] = concurrent_contexts(args.model, args.weights, args.candidate_build,
                                                    spectra(args.concurrency_frames))
        print('Four independent concurrent contexts: exact replay', flush=True)
    report['all_selected_cases_bit_identical'] = True
    report['elapsed_seconds'] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
