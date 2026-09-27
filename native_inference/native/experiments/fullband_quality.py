"""Matched EARS-WHAM_v2 test evaluation; all model processing remains at 48 kHz.

PESQ-WB alone uses explicit 16 kHz resampling. STOI's algorithm internally
resamples to 10 kHz. SI-SNR and official SIGMOS use the original 48 kHz PCM.
Run from /bench in the fullband Docker image. Outputs are resumable per clip.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extended_probe import ExtendedModel
from probe import audio_spectra, session
from model_precision_quality import enhance

ROOT = Path('/bench')
DATA = ROOT/'scratch/fullband'
MODEL = ROOT/'models/dpdfnet8_48khz_hr.onnx'
WEIGHTS = ROOT/'models/rework8/weights.f32'
BASE = ROOT/'build/next_baseline8'
W7 = ROOT/'build/range_w78'
DELAY = 2400
SYSTEMS = ('noisy', 'onnx', 'fp16', 'int8', 'w7a8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def si_snr(clean, estimate):
    x = np.asarray(clean, dtype=np.float64)
    y = np.asarray(estimate, dtype=np.float64)
    x = x-x.mean()
    y = y-y.mean()
    energy = np.dot(x, x)
    if energy <= 1e-20:
        raise ValueError('Silent clean reference')
    target = x*(np.dot(y, x)/energy)
    residual = y-target
    return float(10*np.log10(max(np.dot(target, target), 1e-30)/max(np.dot(residual, residual), 1e-30)))


def init_worker():
    global REFERENCE, MODELS, SCORER
    sys.path.insert(0, str(DATA))
    from sigmos import SigMOS
    REFERENCE = session(MODEL)
    MODELS = {'onnx': REFERENCE,
              'fp16': ExtendedModel(REFERENCE, 3, 16, 7, build=BASE, weights=WEIGHTS),
              'int8': ExtendedModel(REFERENCE, 4, 8, 7, build=BASE, weights=WEIGHTS),
              'w7a8': ExtendedModel(REFERENCE, 4, 8, 7, build=W7, weights=WEIGHTS)}
    SCORER = SigMOS(str(DATA))


def score(clean, estimate, clean16):
    assert clean.shape == estimate.shape and np.isfinite(estimate).all()
    est16 = resample_poly(estimate, 1, 3)
    return {'pesq_wb_16k': float(pesq(16000, clean16, est16, 'wb')),
            'stoi': float(stoi(clean, estimate, 48000, extended=False)),
            'si_snr_48k_db': si_snr(clean, estimate),
            **SCORER.run(estimate, sr=48000)}


def evaluate(job):
    entry, output, fingerprint = job
    destination = output/(entry['id']+'.json')
    if destination.exists():
        cached = json.loads(destination.read_text())
        assert cached['fingerprint'] == fingerprint and cached['clip'] == entry
        return entry['id'], True
    start = time.monotonic()
    noisy_path = DATA/entry['noisy']
    clean, rate = sf.read(DATA/entry['clean'], dtype='float32')
    noisy, nrate = sf.read(noisy_path, dtype='float32')
    assert rate == nrate == 48000 and clean.ndim == noisy.ndim == 1 and clean.shape == noisy.shape
    frames = audio_spectra(noisy_path)
    clean16 = resample_poly(clean, 1, 3)
    result = {'fingerprint': fingerprint, 'clip': entry, 'samples': len(clean),
              'clean_sha256': sha(DATA/entry['clean']), 'noisy_sha256': sha(noisy_path), 'systems': {}}
    for name in SYSTEMS:
        waveform = noisy if name == 'noisy' else enhance(MODELS[name], frames, REFERENCE)[DELAY:DELAY+len(clean)]
        metrics = score(clean, waveform, clean16)
        metrics['pcm_sha256'] = hashlib.sha256(waveform.astype('<f4').tobytes()).hexdigest()
        result['systems'][name] = metrics
        if name != 'noisy':
            wavepath = output/'audio'/name/entry['speaker']/(entry['id']+'.wav')
            wavepath.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wavepath, waveform, 48000, subtype='FLOAT')
    result['elapsed_seconds'] = time.monotonic()-start
    tmp = destination.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    tmp.replace(destination)
    return entry['id'], False


def manifest():
    entries = []
    # Upstream test.csv header has 9 fields but data rows have 12. Read the
    # explicit upstream save_files layout, including three loudness fields.
    rows = list(csv.reader((DATA/'data/EARS-WHAM_v2/test.csv').open()))[1:]
    for row in rows:
        assert len(row) == 12
        identifier, speaker = row[:2]
        prefix = Path('data/EARS-WHAM_v2/test')
        noisy = list((DATA/prefix/'noisy'/speaker).glob(identifier+'_*.wav'))
        assert len(noisy) == 1
        entries.append({'id': identifier, 'speaker': speaker, 'speech_file': row[2],
                        'speech_start': int(row[3]), 'speech_end': int(row[4]),
                        'noise_file': row[5], 'snr_db': float(row[-1]),
                        'clean': str(prefix/'clean'/speaker/(identifier+'.wav')),
                        'noisy': str(noisy[0].relative_to(DATA))})
    assert len(entries) == len({e['id'] for e in entries})
    assert {e['speaker'] for e in entries} == {f'p{i}' for i in range(102, 108)}
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Smoke check only; zero is the entire split')
    parser.add_argument('--output', type=Path, default=DATA/'evaluation')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'model': sha(MODEL), 'weights': sha(WEIGHTS),
                  'baseline_library': sha(BASE/'libdpdf_full.so'), 'w7a8_library': sha(W7/'libdpdf_full.so'),
                  'sigmos_model': sha(DATA/'model-sigmos_1697718653_41d092e8-epo-200.onnx'),
                  'sigmos_code': sha(DATA/'sigmos.py'), 'evaluator': sha(__file__),
                  'generator': sha(DATA/'generate_ears_wham.py'), 'cuts': sha(DATA/'test_files.json'),
                  'selection': sha(DATA/'subset_selection.json'),
                  'subset_preparer': sha(Path(__file__).with_name('prepare_fullband_subset.py')),
                  'test_csv': sha(DATA/'data/EARS-WHAM_v2/test.csv'),
                  'delay_samples': DELAY, 'sample_rate': 48000,
                  'packages': {p: importlib.metadata.version(p) for p in
                               ('numpy', 'scipy', 'soundfile', 'onnxruntime', 'pesq', 'pystoi', 'librosa', 'torch', 'torchaudio', 'pyloudnorm')}}
    assert provenance['sigmos_model'] == 'f939dcc1945055a435565b4369e27dafd0f87df3cea4e2ff6eb81225e52cc53b'
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    entries = manifest()
    selection = json.loads((DATA/'subset_selection.json').read_text())
    assert {e['id'] for e in entries} == {e['id'] for e in selection['clips']}
    if (args.output/'manifest.json').exists():
        previous = json.loads((args.output/'manifest.json').read_text())
        assert previous['fingerprint'] == fingerprint and previous['clips'] == entries, 'Use a new output directory for a changed experiment'
    (args.output/'manifest.json').write_text(json.dumps({'provenance': provenance, 'fingerprint': fingerprint, 'clips': entries}, indent=2)+'\n')
    selected = entries[:args.limit] if args.limit else entries
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=init_worker) as pool:
        futures = [pool.submit(evaluate, (e, args.output, fingerprint)) for e in selected]
        for done, future in enumerate(as_completed(futures), 1):
            print(done, '/', len(selected), future.result(), flush=True)


if __name__ == '__main__':
    main()
