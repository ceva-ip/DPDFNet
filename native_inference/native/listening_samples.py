"""Render side-by-side listening comparisons for both 48 kHz HR models."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from extended_probe import ExtendedModel
from model_precision_final import FINAL_CONFIGS
from probe import audio_spectra, initial_state, session, synthesize


MODEL_SPECS = [
    {
        'name': 'dpdfnet8_48khz_hr',
        'model': Path('models/dpdfnet8_48khz_hr.onnx'),
        'build': Path('build/extended'),
        'weights': Path('models/extended/weights.f32'),
        'filename_prefix': '',
    },
    {
        'name': 'dpdfnet2_48khz_hr',
        'model': Path('models/dpdfnet2_48khz_hr.onnx'),
        'build': Path('build/dpdfnet2_extended'),
        'weights': Path('models/dpdfnet2_extended/weights.f32'),
        'filename_prefix': 'dpdfnet2_48khz_hr_',
    },
]


def render(model, reference, frames):
    state = initial_state(reference)
    enhanced = []
    for frame in frames:
        output, state = model.run(None, {'spec': frame, 'state_in': state})
        assert np.isfinite(output).all() and np.isfinite(state).all()
        enhanced.append(output)
    return synthesize(enhanced)


def write_verified(destination, samples, rate):
    peak = float(np.max(np.abs(samples)))
    assert np.isfinite(samples).all() and peak < 1, 'Refuse to clip or normalize output'
    sf.write(destination, samples, rate, subtype='PCM_24')
    check, check_rate = sf.read(destination, dtype='float32')
    assert check_rate == rate and check.shape == samples.shape
    assert np.max(np.abs(check - samples)) <= 2**-23
    return peak


def main():
    folder = Path('listening_comparison')
    folder.mkdir(exist_ok=True)
    models = []
    for spec in MODEL_SPECS:
        reference = session(spec['model'])
        models.append((spec, reference, {
            'original_fp32': reference,
            'fp16': ExtendedModel(reference, *FINAL_CONFIGS['selective_fp16'],
                                  build=spec['build'], weights=spec['weights']),
            'int8': ExtendedModel(reference, *FINAL_CONFIGS['selective_int8'],
                                  build=spec['build'], weights=spec['weights']),
        }))
    report = {
        'sample_rate': 48000,
        'format': 'PCM_24',
        'gain': 1,
        'removed_model_delay_samples': 2400,
        'precision_scope': 'DPRNN + dense/grouped FC + 1x1 CNN; other CNN remains FP32',
        'models': [spec['name'] for spec in MODEL_SPECS],
        'files': [],
        'artifact_sha256': {},
    }
    for spec in MODEL_SPECS:
        for path in (spec['model'], spec['weights'], spec['build'] / 'libdpdf_full.so'):
            report['artifact_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    sections = []
    for label, stem in [('Fan', 'noisy_public_48k'), ('Cafe', 'noisy_cafe_48k'),
                        ('Keyboard', 'noisy_keyboard_48k')]:
        source = Path('references/HushMic/tests/fixtures') / (stem + '.flac')
        noisy, rate = sf.read(source, dtype='float32')
        assert rate == 48000 and noisy.ndim == 1
        frames = audio_spectra(source)
        noisy_name = f'{label.lower()}_noisy.wav'
        noisy_peak = write_verified(folder / noisy_name, noisy, rate)
        report['files'].append({
            'file': noisy_name, 'model': None, 'variant': 'noisy', 'frames': len(noisy),
            'peak': noisy_peak, 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'sha256': hashlib.sha256((folder / noisy_name).read_bytes()).hexdigest(),
        })
        groups = [f'<div class="card noisy"><h3>Noisy input</h3><audio controls preload="metadata" '
                  f'src="{noisy_name}"></audio><p><a href="{noisy_name}" download>Download WAV</a></p></div>']
        for spec, reference, variants in models:
            cards = []
            for variant, model in variants.items():
                raw = render(model, reference, frames)
                assert len(raw) >= 2400 + len(noisy)
                samples = raw[2400:2400 + len(noisy)]
                filename = f'{label.lower()}_{spec["filename_prefix"]}{variant}.wav'
                peak = write_verified(folder / filename, samples, rate)
                report['files'].append({
                    'file': filename, 'model': spec['name'], 'variant': variant,
                    'frames': len(samples), 'peak': peak,
                    'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'sha256': hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
                })
                title = {'original_fp32': 'Original FP32', 'fp16': 'Selective FP16',
                         'int8': 'Selective INT8'}[variant]
                cards.append(f'<div class="card"><h3>{title}</h3><audio controls preload="metadata" '
                             f'src="{filename}"></audio><p><a href="{filename}" download>Download WAV</a></p></div>')
            groups.append(f'<div class="model"><h3>{spec["name"]}</h3><div class="grid">'
                          + ''.join(cards) + '</div></div>')
        sections.append(f'<section><h2>{label}</h2>{"".join(groups)}</section>')
        print(f'Rendered and verified {label} for both models', flush=True)
    for _, _, variants in models:
        variants['fp16'].close()
        variants['int8'].close()
    (folder / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    shutil.copyfile('references/HushMic/docs/demo/ASSETS.md', folder / 'ASSETS.md')
    (folder / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DPDFNet listening comparison</title>
<style>body{font:16px system-ui;background:#131b27;color:#edf3ff;max-width:1200px;margin:40px auto;padding:0 24px}
h1{font-size:30px}p{line-height:1.6;color:#c2d0df}section{margin:40px 0}.model{margin:20px 0}.model>h3{color:#9ed2ff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.card{background:#202d40;padding:16px;border-radius:12px}
.noisy{max-width:360px}audio{width:100%}a{color:#83caff}.card h3{font-size:16px}</style>
<h1>DPDFNet 48 kHz HR listening comparison</h1><p>The same noisy recordings through the original FP32,
selective FP16, and selective INT8 versions of both high-resolution models. DPRNN, dense/grouped FC,
and 1×1 CNN weights use reduced precision; other convolutions remain FP32.</p><p>48 kHz mono, lossless 24-bit WAV.
No loudness normalization or gain changes. The same 50 ms model delay is removed from every enhanced version.
Playing a clip pauses the others.</p>''' + ''.join(sections) + '''
<p>Sources and licenses: <a href="ASSETS.md">HushMic asset credits</a>.
Fan: Gravity Sound; keyboard: C40115 (both CC BY 4.0). Speech: LibriVox, public domain.
Café ambience: stephan / pdsounds.org, public domain.</p>
<script>document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>{
document.querySelectorAll('audio').forEach(b=>{if(b!==a)b.pause()})}));</script></html>''', encoding='utf-8')


if __name__ == '__main__':
    main()
