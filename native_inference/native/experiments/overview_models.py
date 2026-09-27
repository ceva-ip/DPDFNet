"""Reproduce the README's DPDFNet-2 extension and missing DPDFNet-8 RSS.

Run from /bench in dpdfnet-native-fullband, with BLAS thread counts set to one.
Stages are separate so quality evaluation cannot interfere with timing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'native'))
BUILD = ROOT / 'build/w7_followup_pack_fit5_2'
IDS = ('00033', '00046', '00084', '00133', '00200', '00364')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, result):
    (ROOT / 'results' / name).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def main(stage):
    if stage == 'build':
        subprocess.run(['cmake', '-S', str(ROOT/'scratch/w7_followup/pack_fit5'),
                        '-B', str(BUILD), '-DDPDF_EXTENDED_MODEL=ON',
                        f'-DDPDF_GENERATED_MODEL={ROOT}/models/rework2/generated_model.c',
                        f'-DDPDF_TEST_WEIGHTS={ROOT}/models/rework2/weights.f32'], check=True)
        subprocess.run(['cmake', '--build', str(BUILD), '-j4'], check=True)
        subprocess.run(['ctest', '--test-dir', str(BUILD), '--output-on-failure'], check=True)
        return
    result = {'generated_at': datetime.now(timezone.utc).isoformat(),
              'driver_sha256': sha(__file__), 'stage': stage}
    if stage == 'memory':
        result['method'] = 'Median of four fresh processes; 120 warmup hops; common Python/NumPy/ORT import baseline subtracted.'
        result['models'] = {}
        for size, build in ((8, ROOT/'build/w7_followup_pack_fit5'), (2, BUILD)):
            runs = [json.loads(subprocess.check_output([
                sys.executable, 'native/onnx_memory_probe.py',
                '--model', f'models/dpdfnet{size}_48khz_hr.onnx',
                '--weights', f'models/rework{size}/weights.f32',
                '--build', str(build), '--variant', 'selective_int8'], text=True)) for _ in range(4)]
            result['models'][str(size)] = {'library_sha256': sha(build/'libdpdf_full.so'),
                'weights_sha256': sha(ROOT/f'models/rework{size}/weights.f32'),
                'runs': runs, 'rss_delta_bytes': statistics.median(r['rss_delta_bytes'] for r in runs),
                'owned_bytes': runs[0]['owned_bytes']}
        save('overview_fitted_rss.json', result)
        print(json.dumps(result, indent=2), flush=True)
        return
    from extended_probe import ExtendedModel
    from probe import session, spectra, whole_parity, cpu_name
    reference = session(ROOT/'models/dpdfnet2_48khz_hr.onnx')
    weights = ROOT/'models/rework2/weights.f32'
    base = ROOT/'build/followup_final2'
    result['cpu'] = cpu_name()
    result['artifacts'] = {str(p.relative_to(ROOT)): sha(p) for p in
        (ROOT/'models/dpdfnet2_48khz_hr.onnx', weights, base/'libdpdf_full.so', BUILD/'libdpdf_full.so')}
    models = {'onnx': reference,
              'fp16': ExtendedModel(reference, 3, 16, 7, build=base, weights=weights),
              'int8': ExtendedModel(reference, 4, 8, 7, build=base, weights=weights),
              'w7a8_fit5': ExtendedModel(reference, 4, 8, 7, build=BUILD, weights=weights)}
    if stage == 'timing':
        from full_probe import timings
        fp32 = ExtendedModel(reference, 0, 0, 0, build=BUILD, weights=weights)
        result['fp32_parity'] = whole_parity(reference, fp32, spectra(300), 'synthetic')
        fp32.close()
        result['method'] = 'Median of four run means; standalone 10 ms cadence; 100 warmup + 1000 timed hops; rotating order; allocating Python wrappers; no outliers removed.'
        result['paced'] = timings(models, spectra(1100), 4, True, 100)
        save('dpdfnet2_overview_timing.json', result)
    elif stage == 'quality':
        import fullband_quality as fq
        sys.path.insert(0, str(fq.DATA))
        from sigmos import SigMOS
        fq.SCORER = SigMOS(str(fq.DATA))
        fq.REFERENCE = reference
        fq.MODELS = models
        fq.SYSTEMS = tuple(models)
        manifest = json.loads((ROOT/'results/fullband_ears_wham_v2_clips.json').read_text())
        entries = [c['clip'] for c in manifest['clips'] if c['clip']['id'] in IDS]
        assert len(entries) == 6
        result['sample_rate'] = 48000
        result['delay_samples'] = fq.DELAY
        result['sigmos_model_sha256'] = sha(fq.DATA/'model-sigmos_1697718653_41d092e8-epo-200.onnx')
        result['evaluator_sha256'] = sha(fq.__file__)
        fingerprint = sha(__file__) + ''.join(result['artifacts'].values())
        output = ROOT/'scratch/fullband/overview_dpdfnet2'
        output.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            print(fq.evaluate((entry, output, fingerprint)), flush=True)
        result['clips'] = [json.loads((output/(e['id']+'.json')).read_text()) for e in entries]
        metrics = ('pesq_wb_16k', 'stoi', 'si_snr_48k_db', 'MOS_SIG', 'MOS_NOISE', 'MOS_OVRL')
        result['means'] = {name: {metric: statistics.mean(c['systems'][name][metric] for c in result['clips'])
                                  for metric in metrics} for name in models}
        save('dpdfnet2_overview_quality.json', result)
        print(json.dumps(result['means'], indent=2), flush=True)
    for name, model in models.items():
        if name != 'onnx':
            model.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['build', 'memory', 'timing', 'quality'])
    main(parser.parse_args().stage)
