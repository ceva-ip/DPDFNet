"""Aggregate the complete matched test set with paired bootstrap intervals."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path('/bench')
EVAL = ROOT/'scratch/fullband/evaluation'
OUT = ROOT/'results/fullband_ears_wham_v2'
SYSTEMS = ['noisy', 'onnx', 'fp16', 'int8', 'w7a8']
KEYS = ['pesq_wb_16k', 'stoi', 'si_snr_48k_db', 'MOS_COL', 'MOS_DISC', 'MOS_LOUD',
        'MOS_NOISE', 'MOS_REVERB', 'MOS_SIG', 'MOS_OVRL']


def main():
    manifest = json.loads((EVAL/'manifest.json').read_text())
    clips = [json.loads((EVAL/(e['id']+'.json')).read_text()) for e in manifest['clips']]
    assert all(c['fingerprint'] == manifest['fingerprint'] for c in clips)
    assert all(c['clip'] == e for c, e in zip(clips, manifest['clips']))
    values = np.array([[[c['systems'][s][k] for k in KEYS] for s in SYSTEMS] for c in clips])
    assert np.isfinite(values).all()
    n = len(clips)
    speakers = sorted({c['clip']['speaker'] for c in clips})
    speaker_indices = [np.array([i for i,c in enumerate(clips) if c['clip']['speaker'] == s]) for s in speakers]
    rng = np.random.default_rng(20260926)
    clip_draws = rng.integers(n, size=(10000, n))
    cluster_draws = rng.integers(len(speakers), size=(10000, len(speakers)))
    cluster_counts = np.array([len(ix) for ix in speaker_indices])

    def paired(candidate, reference):
        delta = values[:, SYSTEMS.index(candidate)]-values[:, SYSTEMS.index(reference)]
        answer = {}
        for j, key in enumerate(KEYS):
            d = delta[:, j]
            cluster_totals = np.array([d[ix].sum() for ix in speaker_indices])
            cluster_means = cluster_totals[cluster_draws].sum(axis=1)/cluster_counts[cluster_draws].sum(axis=1)
            answer[key] = {'mean_delta': float(d.mean()), 'median_delta': float(np.median(d)),
                           'clip_bootstrap_95ci': np.quantile(d[clip_draws].mean(axis=1), [.025,.975]).tolist(),
                           'speaker_cluster_bootstrap_95ci': np.quantile(cluster_means, [.025,.975]).tolist(),
                           'p05_delta': float(np.quantile(d,.05)), 'min_delta': float(d.min()),
                           'max_delta': float(d.max()), 'negative_clips': int((d<0).sum())}
        return answer

    def group(indices):
        if len(indices) == 0:
            return {'clips': 0, 'means': None}
        return {'clips': len(indices), 'means': {s: dict(zip(KEYS, values[indices,si].mean(axis=0).tolist()))
                                                 for si,s in enumerate(SYSTEMS)}}

    summary = {'dataset': 'Deterministic 50-clip subset of the official paired EARS-WHAM_v2 test split',
               'selection': json.loads((EVAL.parent/'subset_selection.json').read_text()),
               'clips': n, 'hours': sum(c['samples'] for c in clips)/48000/3600,
               'speakers': speakers, 'provenance': manifest['provenance'],
               'bootstrap': {'replicates':10000, 'seed':20260926, 'interval':'percentile paired 95%; unadjusted descriptive intervals',
                             'cluster_unit':'speaker; resample speakers and retain all their clips; clip-weighted mean'},
               **group(np.arange(n)),
               'paired': {a+'_minus_'+b: paired(a,b) for a,b in
                          [('w7a8','onnx'),('w7a8','fp16'),('w7a8','int8'),('fp16','onnx'),('int8','onnx')]
                          +[(s,'noisy') for s in SYSTEMS[1:]]},
               'by_speaker': {s:group(ix) for s,ix in zip(speakers,speaker_indices)},
               'by_snr': {f'{lo}_to_{hi}_db': group(np.array([i for i,c in enumerate(clips) if lo <= c['clip']['snr_db'] < hi]))
                          for lo,hi in [(-100,0),(0,5),(5,10),(10,15),(15,100)]},
               'worst_w7a8_vs_int8': {k: [{'clip': clips[i]['clip'], 'delta':float(values[i,4,j]-values[i,3,j])}
                                         for i in np.argsort(values[:,4,j]-values[:,3,j])[:10]] for j,k in enumerate(KEYS)}}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    OUT.with_name(OUT.name+'_clips.json').write_text(json.dumps({'manifest':manifest,'clips':clips},indent=2,allow_nan=False)+'\n')
    OUT.with_name(OUT.name+'_sources.json').write_text(json.dumps({
        'downloads':json.loads((EVAL.parent/'subset_download_manifest.json').read_text()),
        'subset_replay_check':json.loads((EVAL.parent/'subset_replay_validation.json').read_text())},indent=2)+'\n')
    with OUT.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow(['id','speaker','speech_file','snr_db','seconds','system',*KEYS,'pcm_sha256'])
        for c in clips:
            for s in SYSTEMS:
                writer.writerow([c['clip'][k] for k in ('id','speaker','speech_file','snr_db')]
                                +[c['samples']/48000,s]+[c['systems'][s][k] for k in KEYS]+[c['systems'][s]['pcm_sha256']])
    print(json.dumps({'clips':n,'hours':summary['hours'],'means':summary['means'],
                      'w7a8_minus_int8':summary['paired']['w7a8_minus_int8']},indent=2))


if __name__ == '__main__':
    main()
