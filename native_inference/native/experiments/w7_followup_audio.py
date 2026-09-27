"""Verify an exact candidate against saved fullband PCM, or screen changed gates.

Uses only the existing, checksummed 48 kHz evaluation fixtures. Parallel audio
checks are correctness/quality runs and must not overlap latency measurements.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import fullband_quality as f
from robustness_quality import metrics
from probe import library, ptr


def initialize(build, exact):
    global MODEL, REFERENCE, EXACT
    EXACT = exact
    REFERENCE = f.session(f.MODEL)
    MODEL = f.ExtendedModel(REFERENCE, 4, 8, 7, build=Path(build), weights=f.WEIGHTS)
    if not exact:
        sys.path.insert(0, str(f.DATA))
        from sigmos import SigMOS
        f.SCORER = SigMOS(str(f.DATA))


def run(case):
    inp, sr = sf.read(case['input'], dtype='float32')
    baseline, rate = sf.read(case['baseline'], dtype='float32')
    assert sr == rate == 48000 and inp.ndim == baseline.ndim == 1
    baseline_sha = hashlib.sha256(baseline.astype('<f4').tobytes()).hexdigest()
    assert baseline_sha == case['expected_pcm_sha256']
    frames = f.audio_spectra(case['input'])
    output = f.enhance(MODEL, frames, REFERENCE)[f.DELAY:f.DELAY+len(inp)]
    assert output.shape == baseline.shape and np.isfinite(output).all()
    output_sha = hashlib.sha256(output.astype('<f4').tobytes()).hexdigest()
    item = {'case': case, 'samples': len(inp), 'hops_including_flush': len(frames),
            'input_sha256': f.sha(case['input']), 'baseline_pcm_sha256': baseline_sha,
            'candidate_pcm_sha256': output_sha, 'bit_identical': baseline_sha == output_sha}
    if EXACT:
        assert item['bit_identical'], case['id']
    else:
        if case['scenario'] == 'long_noise':
            def noise_metrics(audio):
                rms = float(np.mean(audio.astype(np.float64)**2))
                power = float(np.mean(inp.astype(np.float64)**2))
                levels = audio.astype(np.float64).reshape(-1, 960)
                source = inp.astype(np.float64).reshape(-1, 960)
                gain = 10*np.log10(np.maximum(np.mean(levels*levels, axis=1), 1e-30)/np.maximum(np.mean(source*source, axis=1), 1e-30))
                return {'attenuation_db': float(10*np.log10(power/max(rms, 1e-30))),
                        'output_rms_dbfs': float(10*np.log10(max(rms, 1e-30))),
                        'output_peak_dbfs': float(20*np.log10(max(float(np.abs(audio).max()), 1e-30))),
                        'worst_20ms_gain_db': float(gain.max()),
                        'amplified_20ms_windows': int(np.count_nonzero(gain>0)),
                        'last_30s_attenuation_db': float(10*np.log10(np.mean(inp[-30*48000:].astype(np.float64)**2)/max(np.mean(audio[-30*48000:].astype(np.float64)**2), 1e-30)))}
            item['baseline'], item['candidate'] = noise_metrics(baseline), noise_metrics(output)
        else:
            clean, rate = sf.read(case['clean'], dtype='float32')
            assert rate == 48000 and clean.shape == output.shape
            item['baseline'] = metrics(clean, baseline)
            item['candidate'] = metrics(clean, output)
        item['delta'] = {k: item['candidate'][k]-v for k, v in item['baseline'].items()
                         if isinstance(v, (float, int)) and isinstance(item['candidate'].get(k), (float, int))}
        diff = output.astype(np.float64)-baseline
        item['output_difference_snr_db'] = float(10*np.log10(np.sum(baseline.astype(np.float64)**2)/max(np.sum(diff*diff), 1e-30)))
        item['max_abs_pcm_difference'] = float(np.abs(diff).max())
    return item


def cases(exact, noise=False):
    result = []
    manifest = json.loads((f.DATA/'evaluation/manifest.json').read_text())
    assert f.sha(f.W7/'libdpdf_full.so') == manifest['provenance']['w7a8_library']
    assert f.sha(f.MODEL) == manifest['provenance']['model']
    assert f.sha(f.WEIGHTS) == manifest['provenance']['weights']
    clips = manifest['clips']
    if not exact:
        clips = [next(c for c in clips if c['speaker'] == speaker)
                 for speaker in sorted({c['speaker'] for c in clips})]
    for c in clips:
        old = json.loads((f.DATA/'evaluation'/(c['id']+'.json')).read_text())
        result.append({'id': 'mixture_'+c['id'], 'scenario': 'mixture',
                       'input': str(f.DATA/c['noisy']), 'clean': str(f.DATA/c['clean']),
                       'baseline': str(f.DATA/'evaluation/audio/w7a8'/c['speaker']/(c['id']+'.wav')),
                       'expected_pcm_sha256': old['systems']['w7a8']['pcm_sha256']})
    for path in sorted((f.DATA/'robustness').glob('*.json')):
        if path.stem == 'manifest' or (not exact and path.stem not in ('clean_00033', 'low_00084', 'low_00525')):
            continue
        old = json.loads(path.read_text())
        directory = f.DATA/'robustness/audio'/path.stem
        result.append({'id': path.stem, 'scenario': old['case']['scenario'],
                       'input': str(directory/'input.wav'), 'clean': str(directory/'clean.wav'),
                       'baseline': str(directory/'w7a8.wav'),
                       'expected_pcm_sha256': old['systems']['w7a8']['pcm_sha256']})
    if exact or noise:
        for name in ('white', 'pink', 'mechanical'):
            old = json.loads((f.DATA/'long_noise'/(name+'_w7a8.json')).read_text())
            result.append({'id': name, 'scenario': 'long_noise',
                           'input': str(f.DATA/'long_noise'/(name+'.wav')),
                           'baseline': str(f.DATA/'long_noise'/(name+'_w7a8.wav')),
                           'expected_pcm_sha256': old['output_pcm_sha256']})
    return result


def activation_accuracy(build):
    lib = library(build/'libdpdf_dprnn.so')
    # Dense grid through the nonlinear region and both tails, including zero.
    x = np.concatenate([np.linspace(-100, 100, 1000000, dtype=np.float32),
                        np.linspace(-1, 1, 1000000, dtype=np.float32), np.zeros(8, np.float32)])
    s, t = np.empty_like(x), np.empty_like(x)
    assert lib.dpdf_test_gates(2, ptr(x), ptr(s), ptr(t), x.size) == 0
    xd = x.astype(np.float64)
    expected_s, expected_t = 1/(1+np.exp(-xd)), np.tanh(xd)
    assert np.isfinite(s).all() and np.isfinite(t).all()
    return {'values': len(x), 'range': [-100, 100],
            'sigmoid_max_absolute_error': float(np.abs(s-expected_s).max()),
            'tanh_max_absolute_error': float(np.abs(t-expected_t).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--noise', action='store_true', help='Also check all three continuous two-minute noise streams')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    report = {'mode': 'exact PCM regression' if args.exact else 'small fullband activation-quality screen',
              'candidate_build': str(args.build), 'candidate_sha256': f.sha(args.build/'libdpdf_full.so'),
              'baseline_sha256': f.sha(f.W7/'libdpdf_full.so'), 'sample_rate': 48000,
              'note': 'All model/PCM processing is 48 kHz. PESQ is explicitly resampled to 16 kHz; STOI internally uses 10 kHz. SIGMOS and SI-SNR use 48 kHz.',
              'cases': []}
    if not args.exact:
        report['activation_accuracy'] = activation_accuracy(args.build)
    jobs = cases(args.exact, args.noise)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize, initargs=(str(args.build), args.exact)) as pool:
        for future in as_completed([pool.submit(run, case) for case in jobs]):
            item = future.result()
            report['cases'].append(item)
            print(len(report['cases']), '/', len(jobs), item['case']['id'],
                  item['bit_identical'] if args.exact else item['delta'], flush=True)
    report['cases'].sort(key=lambda r: r['case']['id'])
    report['total_audio_seconds'] = sum(r['samples'] for r in report['cases'])/48000
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
