"""Evaluate final selective FP16/INT8 candidates for one 48 kHz model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import correlate, correlation_lags

from dnsmos import DNSMOS, DNSMOS_KEYS
from extended_probe import ExtendedModel
from model_precision_final import FINAL_CONFIGS
from probe import audio_spectra, initial_state, session, synthesize
from quality_probe import fixtures, metrics


def enhance(model, frames, reference):
    state = initial_state(reference)
    output = []
    for frame in frames:
        enhanced, state = model.run(None, {'spec': frame, 'state_in': state})
        assert np.isfinite(enhanced).all() and np.isfinite(state).all()
        output.append(enhanced)
    return synthesize(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dnsmos-dir', type=Path, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)

    reference = session(args.model)
    dnsmos = DNSMOS(args.dnsmos_dir)
    selected = ['selective_fp16', 'selective_int8']
    result = {
        'model': args.model_name,
        'scope': 'One public-domain speaker; seven speech/noise mixtures',
        'precision_scope': 'DPRNN + dense/grouped FC + 1x1 CNN; other CNN remains FP32',
        'selected': selected,
        'artifacts': {
            str(args.model): hashlib.sha256(args.model.read_bytes()).hexdigest(),
            str(args.build / 'libdpdf_full.so'): hashlib.sha256((args.build / 'libdpdf_full.so').read_bytes()).hexdigest(),
            str(args.weights): hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        },
        'dnsmos': dnsmos.metadata,
        'fixtures': [],
    }
    for path, clean in fixtures(args.scratch):
        frames = audio_spectra(path)
        original = enhance(reference, frames, reference)
        input_samples = sf.info(path).frames
        assert original.size >= 2400 + input_samples
        original_dnsmos = dnsmos.score(original[2400:2400+input_samples], 48000)
        item = {'fixture': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'dnsmos_original_fp32': original_dnsmos, 'candidates': {}}
        lag = None
        if clean is not None:
            cross = correlate(original, clean, method='fft')
            lags = correlation_lags(len(original), len(clean))
            valid = (lags >= 0) & (lags <= 4800)
            lag = int(lags[valid][np.argmax(cross[valid])])
            assert 0 < lag < 4800
            item['alignment_samples'] = lag
            item['clean_reference_original'] = metrics(clean, original[lag:lag + len(clean)])
        for name in selected:
            candidate = ExtendedModel(reference, *FINAL_CONFIGS[name],
                                      build=args.build, weights=args.weights)
            output = enhance(candidate, frames, reference)
            candidate.close()
            entry = {'fidelity': metrics(original, output)}
            candidate_dnsmos = dnsmos.score(output[2400:2400+input_samples], 48000)
            entry['dnsmos'] = candidate_dnsmos
            entry['dnsmos_delta'] = {
                key: candidate_dnsmos[key]-original_dnsmos[key] for key in DNSMOS_KEYS
            }
            if clean is not None:
                entry['clean_reference'] = metrics(clean, output[lag:lag + len(clean)])
                entry['delta'] = {key: value - item['clean_reference_original'][key]
                                  for key, value in entry['clean_reference'].items()}
            item['candidates'][name] = entry
            sf.write(args.scratch / f'{path.stem}_{name}.wav', output, 48000, subtype='FLOAT')
            print(path.name, name, entry, flush=True)
        result['fixtures'].append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
