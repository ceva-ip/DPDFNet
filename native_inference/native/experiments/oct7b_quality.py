"""Matched 48 kHz quality evaluation against the frozen Oct7 baseline.

Run in the cached fullband-quality image, separately from latency/memory jobs.
The default evaluates all 50 previously selected EARS-WHAM mixtures. --limit 6
uses one fixed clip per speaker. Optional robustness and long-noise suites use
the existing fixtures unchanged. Per-clip reports are immutable and resumable;
changing any artifact or scoring protocol requires a fresh output directory.
"""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'native'))
sys.path.insert(0, str(EXPERIMENTS))
RATE, DELAY, CONFIG = 48000, 2400, (4, 8, 7)
SIX_IDS = ('00033', '00046', '00084', '00133', '00200', '00364')
PROTOCOL = {
    'sample_rate': RATE, 'alignment_delay_samples': DELAY, 'config': list(CONFIG),
    'streaming': 'One state reset per file; continuous recurrent state through causal STFT and flush hops; no chunk resets',
    'si_snr_48k_db': 'Mean-centered SI-SNR on native 48 kHz aligned waveforms',
    'sigmos': 'Official cached Microsoft SIGMOS v1, seven MOS dimensions, unmodified 48 kHz PCM',
    'pesq_wb_16k': 'PESQ wideband explicitly resampled from 48 kHz to 16 kHz with scipy.signal.resample_poly(1,3); not a 48 kHz PESQ metric',
    'stoi': 'pystoi.stoi(clean, enhanced, 48000), standard STOI; implementation internally resamples to 10 kHz',
    'low_level': 'Cached -50 dBFS clean RMS fixtures; identical gain on speech and mixture; raw scores and a separate common inverse-gain score after enhancement',
    'long_noise': 'Three cached 120-second speech-free noise streams; attenuation/amplification/clipping/state metrics, no referenced speech-quality metrics',
    'aggregation': 'Unweighted per-clip means and paired candidate-minus-baseline deltas; missing metrics stay null and report coverage',
    'parallelism': 'Independent files in worker processes; one thread per native model, ORT session, and math runtime; not a latency benchmark',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(text, encoding='utf-8')
        if immutable:
            # Linking a complete temporary file publishes it atomically and
            # fails if the destination exists, including another writer's clip.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def runtime_imports():
    global np, sf, f, r, noise, ExtendedModel, StreamingRunner, session, audio_spectra, synthesize
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    import numpy as np
    import soundfile as sf
    import fullband_quality as f
    import robustness_quality as r
    import long_noise_quality as noise
    from extended_probe import ExtendedModel
    from streaming_runner import StreamingRunner
    from probe import session, audio_spectra, synthesize


def pcm_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value, dtype='<f4')).cast('B')).hexdigest()


def source_manifest(source):
    current = {p.name: sha(p) for p in sorted(source.iterdir()) if p.is_file()
               and (p.suffix in ('.c', '.h', '.S') or p.name == 'CMakeLists.txt')}
    require(current == read_json(source / 'source_manifest.json'), f'Frozen source changed: {source}')
    return current


def build_identity(build, source):
    values = {}
    for line in (build / 'CMakeCache.txt').read_text().splitlines():
        if ':' in line and '=' in line and not line.startswith('//'):
            key, value = line.split('=', 1)
            values[key.split(':', 1)[0]] = value
    require(Path(values.get('CMAKE_HOME_DIRECTORY', '')).resolve() == source.resolve(),
            f'Build uses a different source directory: {build}')
    require(Path(values.get('DPDF_GENERATED_MODEL', '')).resolve() == (source / 'generated_model.c').resolve(),
            f'Build uses a different generated graph: {build}')
    return {'build': str(build), 'source': str(source), 'source_manifest': source_manifest(source),
            'library_sha256': sha(build / 'libdpdf_full.so'),
            'cmake_cache_sha256': sha(build / 'CMakeCache.txt')}


def fixture_path(data, relative):
    path = (data / relative).resolve()
    require(path.is_relative_to(data.resolve()), 'Fixture path leaves the cached dataset')
    return path


def checked_file(path, expected, artifacts):
    actual = sha(path)
    require(actual == expected, f'Cached fixture hash differs: {path}')
    artifacts[str(path)] = actual
    return actual


def load_catalog(data, saved, artifacts):
    manifest_path = data / 'evaluation/manifest.json'
    manifest = read_json(manifest_path)
    selection = read_json(data / 'subset_selection.json')
    clips = manifest['clips']
    require(len(clips) == 50 and len({c['id'] for c in clips}) == 50, 'Expected the original 50 unique mixtures')
    require({c['id'] for c in clips} == {c['id'] for c in selection['clips']}, 'Cached 50-clip selection changed')
    records = {entry['case']: entry for entry in saved['cases']}
    require(len(records) == len(saved['cases']) == 65, 'Oct7 reference must contain the full 65-case suite')
    jobs = []
    for clip in clips:
        record_path = data / 'evaluation' / (clip['id'] + '.json')
        prior = read_json(record_path)
        require(prior['clip'] == clip, 'Cached clean/noisy clip metadata changed')
        clean = fixture_path(data, clip['clean'])
        noisy = fixture_path(data, clip['noisy'])
        checked_file(clean, prior['clean_sha256'], artifacts)
        require(prior['noisy_sha256'] == records['mixture_' + clip['id']]['input_sha256'], 'Mixture reference hashes disagree')
        artifacts[str(record_path)] = sha(record_path)
        jobs.append({'id': 'mixture_' + clip['id'], 'scenario': 'mixture', 'clip': clip,
                     'input': str(noisy), 'clean': str(clean), 'clean_sha256': prior['clean_sha256']})
    robust_path = data / 'robustness/manifest.json'
    robust = read_json(robust_path)
    require(sum(c['scenario'] == 'low_level' for c in robust['cases']) == 10
            and sum(c['scenario'] == 'clean' for c in robust['cases']) == 2, 'Expected ten quiet and two clean cases')
    for case in robust['cases']:
        identifier, clip = case['case_id'], case['clip']
        require(clip in clips, 'Robustness fixture is outside the original selection')
        prior_path = data / 'robustness' / (identifier + '.json')
        prior = read_json(prior_path)
        require(prior['case'] == case, 'Cached robustness metadata changed')
        source_clean = fixture_path(data, clip['clean'])
        source_input = fixture_path(data, clip['noisy'] if case['scenario'] == 'low_level' else clip['clean'])
        checked_file(source_clean, prior['source_clean_sha256'], artifacts)
        checked_file(source_input, prior['source_input_sha256'], artifacts)
        clean = data / 'robustness/audio' / identifier / 'clean.wav'
        artifacts[str(clean)] = sha(clean)
        artifacts[str(prior_path)] = sha(prior_path)
        gain = float(prior['gain'])
        require(math.isfinite(gain) and gain > 0, 'Invalid cached robustness gain')
        jobs.append({'id': identifier, 'scenario': case['scenario'], 'clip': clip,
                     'input': str(data / 'robustness/audio' / identifier / 'input.wav'),
                     'clean': str(clean), 'clean_sha256': artifacts[str(clean)], 'gain': gain,
                     'source_clean': str(source_clean), 'source_clean_sha256': prior['source_clean_sha256'],
                     'source_input': str(source_input), 'source_input_sha256': prior['source_input_sha256']})
    noise_manifest = data / 'long_noise/manifest.json'
    inputs = read_json(noise_manifest)['inputs']
    require({c['name'] for c in inputs} == {'white', 'pink', 'mechanical'} and len(inputs) == 3,
            'Expected three cached noise fixtures')
    for fixture in inputs:
        require(fixture['samples'] == RATE * 120 and fixture['sample_rate'] == RATE, 'Noise fixture duration/rate changed')
        jobs.append({'id': fixture['name'], 'scenario': 'long_noise',
                     'input': str(data / 'long_noise' / (fixture['name'] + '.wav')), 'fixture': fixture})
    for path in (manifest_path, data / 'subset_selection.json', robust_path, noise_manifest):
        artifacts[str(path)] = sha(path)
    require(len({job['id'] for job in jobs}) == len(jobs) == 65 and {j['id'] for j in jobs} == set(records),
            'Current manifests differ from the Oct7 fixture suite')
    for job in jobs:
        prior = records[job['id']]
        require(prior['scenario'] == job['scenario'], 'Oct7 case scenario changed')
        checked_file(job['input'], prior['input_sha256'], artifacts)
        info = sf.info(job['input'])
        require(info.samplerate == RATE and info.channels == 1 and info.frames == prior['samples'],
                f"Fixture shape/rate differs: {job['id']}")
        job.update({'samples': prior['samples'], 'input_sha256': prior['input_sha256'],
                    'baseline_pcm_sha256': prior['pcm_sha256']['candidate'],
                    'baseline_stream_sha256': prior['stream_sha256']['candidate']})
    return jobs, manifest['provenance']


def select_cases(catalog, limit, robustness, long_noise):
    mixtures = {c['clip']['id']: c for c in catalog if c['scenario'] == 'mixture'}
    require(all(identifier in mixtures for identifier in SIX_IDS), 'Fixed six-clip subset is missing')
    ordered = [mixtures.pop(identifier) for identifier in SIX_IDS]
    require(len({c['clip']['speaker'] for c in ordered}) == 6, 'Six-clip subset must cover all six speakers')
    counts = {c['clip']['speaker']: 1 for c in ordered}
    while mixtures:
        chosen = min(mixtures.values(), key=lambda c: (counts[c['clip']['speaker']],
                     hashlib.sha256(('oct7b-quality:' + c['id']).encode()).hexdigest()))
        ordered.append(chosen)
        counts[chosen['clip']['speaker']] += 1
        del mixtures[chosen['clip']['id']]
    selected = ordered[:limit or len(ordered)]
    selected += [c for c in catalog if (robustness and c['scenario'] in ('low_level', 'clean'))
                 or (long_noise and c['scenario'] == 'long_noise')]
    return selected


def initialize(context):
    global CONTEXT, MODELS, RUNNERS
    runtime_imports()
    CONTEXT = context
    f.ROOT, f.DATA = ROOT, Path(context['data'])
    sys.path.insert(0, context['data'])
    from sigmos import SigMOS
    f.SCORER = SigMOS(context['data'])
    reference = session(context['model'])  # Metadata/state shape only; native inference is measured here.
    MODELS = {name: ExtendedModel(reference, *CONFIG, build=Path(build), weights=Path(context['weights']))
              for name, build in context['builds'].items()}
    RUNNERS = {name: StreamingRunner(model) for name, model in MODELS.items()}
    atexit.register(lambda: [model.close() for model in MODELS.values()])


def continuous_pcm(frames, samples, identifier):
    waveforms, streams = {}, {}
    for name, runner in RUNNERS.items():
        runner.reset()
        output = np.empty_like(frames)
        digest = {key: hashlib.sha256() for key in ('spectrum', 'state')}
        maximum, checkpoints = 0., []
        for index, frame in enumerate(frames):
            enhanced = runner.process(frame)
            require(np.isfinite(enhanced).all() and np.isfinite(runner.state).all(),
                    f'{identifier}: {name} has nonfinite output/state at hop {index}')
            output[index] = enhanced  # Native outputs are borrowed and overwritten at the next hop.
            digest['spectrum'].update(memoryview(enhanced).cast('B'))
            digest['state'].update(memoryview(runner.state).cast('B'))
            state_max = float(np.max(np.abs(runner.state)))
            maximum = max(maximum, state_max)
            if (index + 1) % 1000 == 0:
                checkpoints.append({'input_seconds': (index + 1) / 100, 'state_max_abs': state_max})
        audio = synthesize(output)[DELAY:DELAY + samples]
        require(audio.shape == (samples,) and np.isfinite(audio).all(), f'{identifier}: aligned PCM shape/values differ')
        waveforms[name] = audio
        streams[name] = {'hops_including_flush': len(frames), 'state_reset_count': 1,
                         'all_output_and_state_finite': True, 'max_state_abs': maximum,
                         'state_checkpoints': checkpoints,
                         'stream_sha256': {key: value.hexdigest() for key, value in digest.items()}}
    return waveforms, streams


def noise_metrics(source, audio):
    count = len(source) // 960
    in20 = np.sqrt(np.mean(source[:count * 960].astype(np.float64).reshape(count, 960) ** 2, axis=1))
    out20 = np.sqrt(np.mean(audio[:count * 960].astype(np.float64).reshape(count, 960) ** 2, axis=1))
    gains = 20 * np.log10(np.maximum(out20, 1e-15) / np.maximum(in20, 1e-15))
    per_second = [{'start_seconds': i, 'attenuation_db': noise.db_rms(source[i * RATE:(i + 1) * RATE])
                   - noise.db_rms(audio[i * RATE:(i + 1) * RATE])} for i in range(120)]
    segments = [{'start_seconds': a, 'end_seconds': b,
                 'attenuation_db': noise.db_rms(source[a * RATE:b * RATE]) - noise.db_rms(audio[a * RATE:b * RATE]),
                 'output_rms_dbfs': noise.db_rms(audio[a * RATE:b * RATE])}
                for a, b in ((0, 1), (1, 30), (30, 60), (60, 90), (90, 120))]
    return {'input_rms_dbfs': noise.db_rms(source), 'output_rms_dbfs': noise.db_rms(audio),
            'attenuation_db': noise.db_rms(source) - noise.db_rms(audio),
            'output_peak_dbfs': float(20 * np.log10(max(float(np.max(np.abs(audio))), 1e-30))),
            'clipped_samples': int(np.count_nonzero(np.abs(audio) >= 1)),
            'worst_20ms_gain_db': float(gains.max()), 'worst_20ms_start_seconds': float(np.argmax(gains) * .02),
            'frames_20ms_amplified_over_3db': int(np.count_nonzero(gains > 3)),
            'segments': segments, 'per_second': per_second}


def scalar_metrics(entry):
    return {category + '.' + key: value for category in ('raw', 'preservation', 'level_restored', 'noise')
            for key, value in entry.get(category, {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}


def metric_keys(entry):
    # Keep metrics that failed for every clip visible with zero score coverage.
    return {category + '.' + key for category in ('raw', 'preservation', 'level_restored', 'noise')
            for key, value in entry.get(category, {}).items()
            if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))}


def run_case(case):
    started = time.monotonic()
    path = Path(CONTEXT['output_dir']) / 'clips' / (case['id'] + '.json')
    clip_fingerprint = fingerprint({'experiment': CONTEXT['fingerprint'], 'case': case})
    if path.exists():
        cached = read_json(path)
        require(cached['fingerprint'] == clip_fingerprint and cached['passed'], 'Cached per-clip report is incompatible')
        return cached
    require(sha(case['input']) == case['input_sha256'], 'Input fixture changed during evaluation')
    source, rate = sf.read(case['input'], dtype='float32')
    require(rate == RATE and source.shape == (case['samples'],) and np.isfinite(source).all(), 'Invalid input PCM')
    if case['scenario'] == 'long_noise':
        require(pcm_sha(source) == case['fixture']['pcm_sha256'], 'Noise PCM differs from its cached manifest')
        clean, original_clean = None, None
    else:
        require(sha(case['clean']) == case['clean_sha256'], 'Clean reference changed during evaluation')
        clean, clean_rate = sf.read(case['clean'], dtype='float32')
        require(clean_rate == RATE and clean.shape == source.shape and np.isfinite(clean).all(), 'Invalid clean reference')
        original_clean = clean
        if case['scenario'] in ('low_level', 'clean'):
            require(sha(case['source_clean']) == case['source_clean_sha256']
                    and sha(case['source_input']) == case['source_input_sha256'], 'Original robustness sources changed')
            original_clean, original_rate = sf.read(case['source_clean'], dtype='float32')
            original_input, input_rate = sf.read(case['source_input'], dtype='float32')
            require(original_rate == input_rate == RATE and original_clean.shape == original_input.shape == clean.shape,
                    'Original robustness shape/rate differs')
            require(pcm_sha((original_clean * case['gain']).astype(np.float32)) == pcm_sha(clean)
                    and pcm_sha((original_input * case['gain']).astype(np.float32)) == pcm_sha(source),
                    'Robustness fixtures do not reproduce their cached gain')
            if case['scenario'] == 'low_level':
                require(abs(r.level(clean) + 50) < 1e-5, 'Quiet reference is no longer -50 dBFS RMS')
    frames = audio_spectra(case['input'])
    waveforms, streams = continuous_pcm(frames, case['samples'], case['id'])
    require(pcm_sha(waveforms['baseline']) == case['baseline_pcm_sha256'], 'Live Oct7 baseline PCM differs from its frozen report')
    require(streams['baseline']['stream_sha256'] == case['baseline_stream_sha256'],
            'Live Oct7 baseline spectra/state differ from its frozen report')
    identical = pcm_sha(waveforms['baseline']) == pcm_sha(waveforms['candidate'])
    systems = {}
    for name, audio in waveforms.items():
        entry = {'pcm_sha256': pcm_sha(audio), **streams[name]}
        if case['scenario'] == 'long_noise':
            entry['noise'] = noise_metrics(source, audio)
        elif name == 'candidate' and identical:
            # Identical aligned PCM is sufficient to reuse deterministic quality scores.
            for category in ('raw', 'preservation', 'level_restored'):
                if category in systems['baseline']:
                    entry[category] = dict(systems['baseline'][category])
            entry['quality_scores_reused_from_identical_baseline_pcm'] = True
        else:
            entry['raw'] = r.metrics(clean, audio)
            entry['preservation'] = r.preservation(clean, audio)
            if case['scenario'] == 'low_level':
                entry['level_restored'] = r.metrics(original_clean, (audio / case['gain']).astype(np.float32))
        systems[name] = entry
    left, right = waveforms['baseline'].astype(np.float64), waveforms['candidate'].astype(np.float64)
    error = right - left
    energy, squared = float(np.dot(left, left)), float(np.dot(error, error))
    baseline_metrics, candidate_metrics = (scalar_metrics(systems[name]) for name in ('baseline', 'candidate'))
    deltas = {key: candidate_metrics[key] - value for key, value in baseline_metrics.items() if key in candidate_metrics}
    result = {'case': case, 'fingerprint': clip_fingerprint, 'experiment_fingerprint': CONTEXT['fingerprint'],
              'passed': True, 'sample_rate': RATE, 'alignment_delay_samples': DELAY,
              'duration_seconds': case['samples'] / RATE, 'systems': systems,
              'paired_candidate_minus_baseline': deltas,
              'waveform_difference': {'pcm_bit_identical': identical, 'max_abs': float(np.abs(error).max()),
                 'rmse': math.sqrt(squared / len(error)), 'relative_rms': math.sqrt(squared / max(energy, 1e-30)),
                 'baseline_to_candidate_error_snr_db': 10 * math.log10(max(energy, 1e-30) / squared) if squared else None,
                 'float_bits_different_samples': int(np.count_nonzero(waveforms['baseline'].view(np.uint32)
                                                                        != waveforms['candidate'].view(np.uint32)))},
              'elapsed_seconds': time.monotonic() - started}
    write_json(path, result, immutable=True)
    return result


def summarize(results):
    summary = {}
    for scenario in ('mixture', 'low_level', 'clean', 'long_noise'):
        group = [r for r in results if r['case']['scenario'] == scenario]
        if not group:
            continue
        metrics = {item['case']['id']: {name: scalar_metrics(entry) for name, entry in item['systems'].items()}
                   for item in group}
        keys = sorted({key for clip in group for system in clip['systems'].values() for key in metric_keys(system)})
        entry = {'clips': len(group), 'pcm_bit_identical_clips': sum(c['waveform_difference']['pcm_bit_identical'] for c in group),
                 'metrics': {}}
        for key in keys:
            paired = [(identifier, values['candidate'][key] - values['baseline'][key]) for identifier, values in metrics.items()
                      if key in values['baseline'] and key in values['candidate']]
            item = {'paired_clips': len(paired), 'missing_paired_clips': len(group) - len(paired)}
            for name in ('baseline', 'candidate'):
                values = [v[name][key] for v in metrics.values() if key in v[name]]
                item[name + '_mean'] = sum(values) / len(values) if values else None
                item[name + '_scored_clips'] = len(values)
            if paired:
                minimum, maximum = min(paired, key=lambda v: v[1]), max(paired, key=lambda v: v[1])
                item.update({'paired_mean_delta': sum(v for _, v in paired) / len(paired),
                             'paired_min_delta': minimum[1], 'paired_min_delta_case': minimum[0],
                             'paired_max_delta': maximum[1], 'paired_max_delta_case': maximum[0]})
            entry['metrics'][key] = item
        summary[scenario] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, choices=(2, 8), default=8)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--limit', type=int, default=0, help='Mixture count only; 0 evaluates all 50, 6 uses one fixed clip per speaker')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--robustness', action='store_true', help='Also evaluate the full ten quiet and two clean controls')
    parser.add_argument('--long-noise', action='store_true', help='Also evaluate all three 120-second noise streams')
    parser.add_argument('--data', type=Path, default=ROOT / 'scratch/fullband')
    parser.add_argument('--output-dir', type=Path, help='Fresh/resumable per-clip cache; default scratch/oct7b_quality/SIZE/VARIANT')
    parser.add_argument('--output', type=Path, help='Aggregate report; default results/oct7b_SIZE_VARIANT_quality.json')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_+.-]+', args.variant) or args.variant in ('.', '..'):
        parser.error('Variant must be a safe snapshot name')
    if not 0 <= args.limit <= 50 or args.workers < 1:
        parser.error('Limit must be 0..50 and workers must be positive')
    runtime_imports()
    data = args.data.resolve()
    source = {'baseline': ROOT / f'scratch/oct7/{args.size}/combo_asm_norm',
              'candidate': ROOT / f'scratch/oct7b/{args.size}/{args.variant}'}
    builds = {'baseline': ROOT / f'build/oct7_{args.size}_combo_asm_norm',
              'candidate': ROOT / f'build/oct7b_{args.size}_{args.variant}'}
    model, weights = ROOT / f'models/dpdfnet{args.size}_48khz_hr.onnx', ROOT / f'models/rework{args.size}/weights.f32'
    reference_path = ROOT / f'results/oct7_{args.size}_combo_asm_norm_audio.json'
    saved = read_json(reference_path)
    require(saved['sample_rate'] == RATE and saved['alignment_delay_samples'] == DELAY and saved['config'] == list(CONFIG),
            'Oct7 audio protocol changed')
    require(sha(model) == saved['model_sha256'] and sha(weights) == saved['weights_sha256'], 'Model/weight provenance differs from Oct7')
    identities = {name: build_identity(builds[name], source[name]) for name in builds}
    require(identities['baseline']['library_sha256'] == saved['library_sha256']['candidate'], 'Frozen Oct7 baseline library changed')
    for name, digest in saved['source_identity']['candidate']['source_files_sha256'].items():
        require(identities['baseline']['source_manifest'].get(name) == digest, 'Frozen Oct7 baseline source differs from the audio report')
    oct7_summary_path = ROOT / 'results/oct7_summary.json'
    oct7_summary = read_json(oct7_summary_path)['models'][str(args.size)]
    require(identities['baseline']['source_manifest'] == oct7_summary['source_manifest'], 'Oct7 assembly/source snapshot differs from accepted summary')
    artifacts = {str(model): sha(model), str(weights): sha(weights), str(reference_path): sha(reference_path),
                 str(oct7_summary_path): sha(oct7_summary_path)}
    for name in builds:
        for filename, digest in identities[name]['source_manifest'].items():
            artifacts[str(source[name] / filename)] = digest
        for path in (source[name] / 'source_manifest.json', builds[name] / 'CMakeCache.txt', builds[name] / 'libdpdf_full.so'):
            artifacts[str(path)] = sha(path)
    catalog, original_provenance = load_catalog(data, saved, artifacts)
    for filename, key in (('sigmos.py', 'sigmos_code'), ('model-sigmos_1697718653_41d092e8-epo-200.onnx', 'sigmos_model')):
        checked_file(data / filename, original_provenance[key], artifacts)
    helpers = [Path(__file__), EXPERIMENTS / 'fullband_quality.py', EXPERIMENTS / 'robustness_quality.py',
               EXPERIMENTS / 'long_noise_quality.py', EXPERIMENTS / 'oct7b_optimization.py', ROOT / 'benchmark.py']
    helpers += [ROOT / 'native' / name for name in ('probe.py', 'streaming_runner.py', 'extended_probe.py', 'full_probe.py')]
    for path in helpers:
        artifacts[str(path)] = sha(path)
    from oct7b_optimization import approximate
    provenance = {'size': args.size, 'variant': args.variant, 'candidate_declared_approximate': approximate(args.variant),
                  'protocol': PROTOCOL, 'builds': identities, 'model_sha256': sha(model), 'weights_sha256': sha(weights),
                  'oct7_audio_report_sha256': sha(reference_path), 'artifact_sha256': artifacts,
                  'catalog_sha256': fingerprint(catalog), 'catalog_cases': len(catalog),
                  'python': {'version': platform.python_version(), 'executable': sys.executable, 'sha256': sha(sys.executable)},
                  'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'soundfile', 'onnxruntime', 'pesq', 'pystoi')},
                  'selection': 'Fixed six IDs first (one per speaker), then deterministic speaker-balanced SHA256 order within the original 50; selected before candidate scoring'}
    experiment_fingerprint = fingerprint(provenance)
    output_dir = (args.output_dir or ROOT / f'scratch/oct7b_quality/{args.size}/{args.variant}').resolve()
    output = args.output or ROOT / f'results/oct7b_{args.size}_{args.variant}_quality.json'
    manifest = {'fingerprint': experiment_fingerprint, 'provenance': provenance, 'cases': catalog}
    manifest_path = output_dir / 'manifest.json'
    if manifest_path.exists():
        require(read_json(manifest_path) == manifest, 'Evaluation artifacts/protocol changed; use a fresh output directory')
    else:
        write_json(manifest_path, manifest, immutable=True)
    if output.exists():
        require(read_json(output)['fingerprint'] == experiment_fingerprint, 'Aggregate report belongs to another experiment; choose a new --output')
    selected = select_cases(catalog, args.limit, args.robustness, args.long_noise)
    context = {'fingerprint': experiment_fingerprint, 'output_dir': str(output_dir), 'data': str(data),
               'model': str(model), 'weights': str(weights), 'builds': {k: str(v) for k, v in builds.items()}}
    errors = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(context,)) as pool:
        futures = {pool.submit(run_case, case): case['id'] for case in selected}
        for index, future in enumerate(as_completed(futures), 1):
            identifier = futures[future]
            try:
                result = future.result()
                print(f"{index}/{len(selected)} {identifier}: PCM identical={result['waveform_difference']['pcm_bit_identical']}", flush=True)
            except Exception as error:
                errors.append({'case': identifier, 'error_type': type(error).__name__, 'error': str(error)})
                print(f'{index}/{len(selected)} {identifier}: FAILED {error}', flush=True)
    for path, digest in artifacts.items():
        require(sha(path) == digest, f'Artifact changed during evaluation: {path}')
    completed = []
    for case in catalog:
        path = output_dir / 'clips' / (case['id'] + '.json')
        if path.exists():
            entry = read_json(path)
            require(entry['fingerprint'] == fingerprint({'experiment': experiment_fingerprint, 'case': case}) and entry['passed'],
                    'Completed per-clip report has incompatible provenance')
            completed.append(entry)
    completed.sort(key=lambda entry: entry['case']['id'])
    requested_ids = {case['id'] for case in selected}
    completed_ids = {entry['case']['id'] for entry in completed}
    report = {'fingerprint': experiment_fingerprint, 'provenance': provenance,
              'passed': not errors and requested_ids <= completed_ids, 'errors': errors,
              'request': {'mixture_limit': args.limit, 'workers': args.workers, 'robustness': args.robustness,
                          'long_noise': args.long_noise, 'cases': sorted(requested_ids)},
              'completed_cases': len(completed), 'full_50_mixture_suite_completed': sum(c['case']['scenario'] == 'mixture' for c in completed) == 50,
              'full_65_fixture_suite_completed': len(completed) == 65,
              'all_completed_pcm_bit_identical': bool(completed) and all(c['waveform_difference']['pcm_bit_identical'] for c in completed),
              'summary': summarize(completed), 'cases': completed}
    write_json(output, report)
    print(f'Matched quality report: {output}; {len(completed)} immutable clips cached', flush=True)
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
