"""Compare W8A7/W7A8 research builds with current W8A8 and original ONNX.

Uses the existing seven-fixture engineering suite and local DNSMOS models.
Recomputes every reference; saved audio is available for listening review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags
from extended_probe import ExtendedModel, CONFIGS
from probe import session, audio_spectra
from model_precision_quality import enhance
from quality_probe import fixtures, metrics
from dnsmos import DNSMOS, DNSMOS_KEYS


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'weights', 'baseline-build', 'u7-build', 'w7-build', 'scratch', 'dnsmos-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    voice = Path('scratch/quality_voice.mp3')
    if sha(voice) != '66edbd95aae1d328ad12005a11899642212e224563dd91d8d5c6d26e62cde5cf':
        raise ValueError('Unexpected source voice')
    ref = session(args.model)
    scorer = DNSMOS(args.dnsmos_dir)
    builds = {'baseline': args.baseline_build, 'u7': args.u7_build, 'w7': args.w7_build}
    report = {'scope': 'Three HushMic fixtures and four controlled noise mixtures; small engineering suite, not a non-inferiority study.',
              'dnsmos': scorer.metadata, 'model_sha256': sha(args.model), 'weights_sha256': sha(args.weights),
              'voice_sha256': sha(voice), 'artifacts': {k: sha(v/'libdpdf_full.so') for k, v in builds.items()},
              'fixtures': []}
    for path, clean in fixtures(args.scratch):
        frames = audio_spectra(path)
        original = enhance(ref, frames, ref)
        sample_count = sf.info(path).frames
        pcm, _ = sf.read(path, dtype='float32', always_2d=True)
        item = {'fixture': path.name, 'sha256': sha(path),
                'pcm_sha256': hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest(),
                'frames': len(frames), 'candidates': {}}
        outputs = {'onnx': original}
        for name, build in builds.items():
            model = ExtendedModel(ref, *CONFIGS['fc_and_1x1_8'], build=build, weights=args.weights)
            try:
                outputs[name] = enhance(model, frames, ref)
            finally:
                model.close()
        lag = None
        if clean is not None:
            cross = correlate(original, clean, method='fft')
            lags = correlation_lags(len(original), len(clean))
            valid = (lags >= 0) & (lags <= 4800)
            lag = int(lags[valid][np.argmax(cross[valid])])
            assert 0 < lag < 4800
            item['alignment_samples'] = lag
        for name, output in outputs.items():
            sf.write(args.scratch/f'{path.stem}_{name}.wav', output, 48000, subtype='FLOAT')
            assert output.size >= 2400+sample_count
            entry = {'dnsmos': scorer.score(output[2400:2400+sample_count], 48000),
                     'pcm_sha256': hashlib.sha256(output.astype('<f4').tobytes()).hexdigest()}
            if name != 'onnx':
                entry['fidelity_to_onnx'] = metrics(original, output)
            if name in ('u7', 'w7'):
                entry['fidelity_to_baseline'] = metrics(outputs['baseline'], output)
            if clean is not None:
                entry['clean_reference'] = metrics(clean, output[lag:lag+len(clean)])
            item['candidates'][name] = entry
        for name in ('u7', 'w7'):
            entry, baseline = item['candidates'][name], item['candidates']['baseline']
            entry['dnsmos_delta_from_baseline'] = {k: entry['dnsmos'][k]-baseline['dnsmos'][k] for k in DNSMOS_KEYS}
            if clean is not None:
                entry['clean_delta_from_baseline'] = {k: entry['clean_reference'][k]-baseline['clean_reference'][k]
                                                      for k in baseline['clean_reference']}
        report['fixtures'].append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(path.name, {k: item['candidates'][k]['dnsmos_delta_from_baseline'] for k in ('u7', 'w7')}, flush=True)


if __name__ == '__main__':
    main()
