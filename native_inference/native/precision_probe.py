"""Benchmark opt-in reduced precision; measure drift without hiding failed gates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from full_probe import FullModel, timings
from probe import (session, spectra, audio_spectra, initial_state, synthesize,
                   whole_parity, cpu_name)


def assess(ref, candidate, frames, name):
    left, right = initial_state(ref), initial_state(candidate)
    output = [[], []]
    spectrum_pass = state_pass = True
    max_spectrum = max_state = 0.0
    for x in frames:
        a, left = ref.run(None, {'spec': x, 'state_in': left})
        b, right = candidate.run(None, {'spec': x, 'state_in': right})
        assert all(np.isfinite(v).all() for v in (a, b, left, right))
        spectrum_pass &= bool(np.allclose(a, b, atol=2e-3, rtol=2e-3))
        state_pass &= bool(np.allclose(left, right, atol=2e-3, rtol=2e-3))
        max_spectrum = max(max_spectrum, float(np.max(np.abs(a-b))))
        max_state = max(max_state, float(np.max(np.abs(left-right))))
        output[0].append(a); output[1].append(b)
    a, b = (synthesize(v).astype(np.float64) for v in output)
    snr = float(10*np.log10(max(np.sum(a*a), 1e-30)/max(np.sum((a-b)**2), 1e-30)))
    return {'fixture': name, 'frames': len(frames), 'waveform_snr_db': snr,
            'waveform_max_abs': float(np.max(np.abs(a-b))),
            'spectrum_max_abs': max_spectrum, 'state_max_abs': max_state,
            'spectrum_gate_pass': spectrum_pass, 'state_gate_pass': state_pass,
            'fp32_numerical_equivalence_gate_pass': snr > 70 and spectrum_pass and state_pass}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    ref = session('models/dpdfnet8_48khz_hr.onnx')
    full = FullModel(Path('build/full/libdpdf_full.so'), Path('models/full_c/weights.f32'), ref)
    half = FullModel(Path('build/full/libdpdf_full.so'), Path('models/full_c/weights.f32'), ref, tier=3)
    qnative = FullModel(Path('build/full/libdpdf_full.so'), Path('models/full_c/weights.f32'), ref, tier=4)
    custom = 'build/baseline/libdpdf_ort.so'
    hybrid = session('models/native_blocks/hybrid.onnx', custom)
    quant = session('models/hybrid_int8.onnx', custom)
    frames = spectra(200 if a.smoke else 1100)
    result = {'cpu': cpu_name(), 'fp16_scope': 'DPRNN matrix weights; F16C conversion, FP32 compute/state/bias/norm',
              'int8_scope': 'ten Gemm projections in remaining 256-unit GRUs; dynamic QInt8 reduced range',
              'int8_native_scope': '128 DPRNN matrices; per-output W8, per-row dynamic asymmetric A8, FP32 state/norm',
              'config': {'warmup_frames': 100, 'timed_frames': len(frames)-100, 'repeats': 1 if a.smoke else 3, 'threads': 1},
              'gate_interpretation': 'Numerical equivalence to FP32 only; not a reduced-precision perceptual quality decision',
              'fp32_parity': whole_parity(ref, full, frames, 'synthetic'), 'fp16_parity': []}
    inputs = [(frames, 'synthetic')]
    if not a.smoke:
        for name in ('noisy_public_48k.flac', 'noisy_cafe_48k.flac', 'noisy_keyboard_48k.flac'):
            inputs.append((audio_spectra(Path('references/HushMic/tests/fixtures') / name), name))
    result['int8_native_parity'] = []
    for data, name in inputs:
        item = assess(ref, half, data, name)
        result['fp16_parity'].append(item)
        print('FP16:', item, flush=True)
        item = assess(ref, qnative, data, name)
        result['int8_native_parity'].append(item)
        print('Native INT8:', item, flush=True)
    sessions = {'original_onnx': ref, 'previous_hybrid': hybrid,
                'full_c_fp32': full, 'full_c_fp16_weights': half,
                'full_c_int8': qnative, 'hybrid_int8': quant}
    result['timings'] = timings(sessions, frames, 1 if a.smoke else 3, False)
    if not a.smoke:
        result['paced_timings'] = timings(sessions, frames, 3, True)
    full.close(); half.close(); qnative.close()
    path = Path('results/precision_smoke.json' if a.smoke else 'results/precision_comparison.json')
    path.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
