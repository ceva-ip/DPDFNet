"""PESQ/STOI fidelity and clean-reference quality for reduced precision.

Small engineering evaluation, not a diverse-speech non-inferiority study.
All candidates receive identical causal spectra. Clean-reference alignment is
estimated once from original FP32 and applied unchanged to every candidate.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, correlate, correlation_lags, lfilter
from pystoi import stoi
from pesq import pesq
from full_probe import FullModel
from probe import session, initial_state, audio_spectra, synthesize


def metrics(ref, out):
    assert ref.shape == out.shape and np.isfinite(out).all()
    ref16 = resample_poly(ref, 1, 3).astype(np.float32)
    out16 = resample_poly(out, 1, 3).astype(np.float32)
    a, b = ref.astype(np.float64), out.astype(np.float64)
    return {'pesq_wb': float(pesq(16000, ref16, out16, 'wb')),
            'stoi': float(stoi(ref, out, 48000, extended=False)),
            'estoi': float(stoi(ref, out, 48000, extended=True)),
            'snr_db': float(10*np.log10(max(np.sum(a*a), 1e-30)/max(np.sum((a-b)**2), 1e-30)))}


def fixtures(folder):
    paths = [(Path('references/HushMic/tests/fixtures') / name, None)
             for name in ('noisy_public_48k.flac', 'noisy_cafe_48k.flac', 'noisy_keyboard_48k.flac')]
    voice, rate = sf.read('scratch/quality_voice.mp3', dtype='float32', always_2d=True)
    voice = voice.mean(axis=1)
    voice = voice[round(17.55*rate):round(28.3*rate)]
    divisor = np.gcd(rate, 48000)
    voice = resample_poly(voice, 48000//divisor, rate//divisor).astype(np.float32)
    voice *= .08 / np.sqrt(np.mean(voice**2))
    sf.write(folder/'clean_voice.wav', voice, 48000, subtype='FLOAT')
    rng = np.random.default_rng(20260918)
    t = np.arange(voice.size) / 48000
    noise = rng.standard_normal(voice.size)
    fan = lfilter([1], [1, -.93], noise) + .8*np.sin(2*np.pi*120*t)
    keyboard = np.zeros(voice.size)
    for pos in range(1000, voice.size-4000, 6200):
        keyboard[pos:pos+4000] += rng.standard_normal(4000)*np.exp(-np.arange(4000)/350)
    for label, noise in [('fan', fan), ('typing', keyboard)]:
        for db in (-5, 5):
            scaled = noise * (.08 / np.sqrt(np.mean(noise**2))) * 10**(-db/20)
            mix = voice + scaled
            gain = min(1.0, .95/np.max(np.abs(mix)))
            path = folder/f'controlled_{label}_{db:+d}dB.wav'
            sf.write(path, mix*gain, 48000, subtype='FLOAT')
            paths.append((path, (voice*gain).astype(np.float32)))
    return paths


def main():
    folder = Path('scratch/quality_audio'); folder.mkdir(parents=True, exist_ok=True)
    assert hashlib.sha256(Path('scratch/quality_voice.mp3').read_bytes()).hexdigest() == '66edbd95aae1d328ad12005a11899642212e224563dd91d8d5c6d26e62cde5cf'
    ref = session('models/dpdfnet8_48khz_hr.onnx')
    native = {name: FullModel(Path('build/full/libdpdf_full.so'), Path('models/full_c/weights.f32'), ref, tier)
              for name, tier in [('c_fp32', 0), ('c_fp16', 3), ('c_int8', 4)]}
    candidates = {'original': ref, **native,
                  'hybrid_int8': session('models/hybrid_int8.onnx', 'build/baseline/libdpdf_ort.so')}
    result = {'scope': '3 HushMic mixtures plus 4 controlled synthetic-noise mixtures, one public-domain speaker',
              'voice_source': 'https://archive.org/download/spc266_2508_librivox/spc266_afterlove_pac_128kb.mp3',
              'voice_sha256': hashlib.sha256(Path('scratch/quality_voice.mp3').read_bytes()).hexdigest(),
              'versions': {'pesq': '0.0.4', 'pystoi': '0.4.1', 'scipy': '1.17.1'}, 'fixtures': []}
    for path, clean in fixtures(folder):
        frames = audio_spectra(path)
        outputs = {}
        for name, sess in candidates.items():
            state = initial_state(sess); spectra_out = []
            for x in frames:
                y, state = sess.run(None, {'spec': x, 'state_in': state})
                spectra_out.append(y)
            outputs[name] = synthesize(spectra_out)
            sf.write(folder/f'{path.stem}_{name}.wav', outputs[name], 48000, subtype='FLOAT')
        item = {'fixture': path.name, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'frames': len(frames), 'fidelity_to_fp32': {
                    name: metrics(outputs['original'], out) for name, out in outputs.items() if name != 'original'}}
        if clean is not None:
            original = outputs['original']
            xc = correlate(original, clean, mode='full', method='fft')
            lags = correlation_lags(original.size, clean.size, mode='full')
            eligible = (lags >= 0) & (lags <= 4800)
            lag = int(lags[eligible][np.argmax(xc[eligible])])
            assert 0 < lag < 4800, 'Alignment hit search boundary'
            assert all(len(out) >= lag+len(clean) for out in outputs.values())
            item['alignment_samples'] = lag
            item['clean_reference'] = {name: metrics(clean, out[lag:lag+len(clean)]) for name, out in outputs.items()}
            original_scores = item['clean_reference']['original']
            item['delta_from_original'] = {name: {key: value-original_scores[key] for key, value in scores.items()}
                                            for name, scores in item['clean_reference'].items() if name != 'original'}
        result['fixtures'].append(item)
        print(path.name, 'fidelity:', item['fidelity_to_fp32'], flush=True)
        if clean is not None:
            print('Clean-reference deltas:', item['delta_from_original'], flush=True)
        Path('results/perceptual_quality.json').write_text(json.dumps(result, indent=2)+'\n')
    for model in native.values():
        model.close()


if __name__ == '__main__':
    main()
