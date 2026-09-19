"""Assess remaining-GRU INT8 against the quality-preserving numerical gate.

Does not promote the candidate or equate numerical error with perceptual harm.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from probe import session, audio_spectra, initial_state, synthesize, whole_parity


def main():
    reference = session('models/dpdfnet8_48khz_hr.onnx')
    custom = 'build/baseline/libdpdf_ort.so'
    lowered = session('models/hybrid_int8.fp32_lowered.onnx', custom)
    quantized = session('models/hybrid_int8.onnx', custom)
    results = []
    for name in ('noisy_public_48k.flac', 'noisy_cafe_48k.flac', 'noisy_keyboard_48k.flac'):
        path = Path('references/HushMic/tests/fixtures') / name
        frames = audio_spectra(path)
        lower_check = whole_parity(reference, lowered, frames[:100], name)
        left, right = initial_state(reference), initial_state(quantized)
        output = [[], []]
        max_spec = max_state = 0.0
        spec_pass = state_pass = True
        for x in frames:
            a, left = reference.run(None, {'spec': x, 'state_in': left})
            b, right = quantized.run(None, {'spec': x, 'state_in': right})
            assert all(np.isfinite(v).all() for v in (a, b, left, right))
            max_spec = max(max_spec, float(np.max(np.abs(a-b))))
            max_state = max(max_state, float(np.max(np.abs(left-right))))
            spec_pass &= bool(np.allclose(a, b, atol=2e-3, rtol=2e-3))
            state_pass &= bool(np.allclose(left, right, atol=2e-3, rtol=2e-3))
            output[0].append(a); output[1].append(b)
        a, b = (synthesize(v).astype(np.float64) for v in output)
        snr = float(10*np.log10(np.sum(a*a)/max(np.sum((a-b)**2), 1e-30)))
        item = {'fixture': name, 'frames': len(frames),
                'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'lowered_fp32_smoke': lower_check,
                'waveform_snr_db': snr, 'waveform_max_abs': float(np.max(np.abs(a-b))),
                'spectrum_max_abs': max_spec, 'state_max_abs': max_state,
                'spectrum_gate_pass': spec_pass, 'state_gate_pass': state_pass,
                'fp32_numerical_equivalence_gate_pass': snr > 70 and spec_pass and state_pass}
        results.append(item)
        print(name, 'SNR:', snr, 'dB; numerical gate:', item['fp32_numerical_equivalence_gate_pass'], flush=True)
    Path('results/hybrid_int8_quality.json').write_text(json.dumps({
        'quantized_model_sha256': hashlib.sha256(Path('models/hybrid_int8.onnx').read_bytes()).hexdigest(),
        'quantized_nodes': 10, 'policy': 'FP32 remains default; INT8 needs independent quality evaluation',
        'fixtures': results}, indent=2)+'\n')


if __name__ == '__main__':
    main()
