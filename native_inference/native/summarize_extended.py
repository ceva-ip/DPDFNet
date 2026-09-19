"""Summarize recorded extended-precision measurements without rerunning inference."""
import json
from pathlib import Path
from statistics import median


def main():
    root = Path(__file__).resolve().parents[1] / 'results'
    final = json.loads((root / 'extended_final.json').read_text())
    quality = json.loads((root / 'extended_quality.json').read_text())
    ablations = json.loads((root / 'extended_ablations.json').read_text())
    out = {'cpu': final['cpu'], 'timing': {}, 'memory': {}, 'quality': {}, 'screening': {}}
    for pacing in ('continuous', 'paced'):
        out['timing'][pacing] = {}
        for name, rows in final[pacing].items():
            out['timing'][pacing][name] = {
                'median_mean_ms': median(r['mean_ms'] for r in rows),
                'p99_range_ms': [min(r['p99_ms'] for r in rows), max(r['p99_ms'] for r in rows)],
                'over_10ms_total': sum(r['over_10ms'] for r in rows),
            }
    for name, rows in final['memory'].items():
        out['memory'][name] = {k: median(r[k] for r in rows) for k in rows[0]}
    for name in quality['selected']:
        candidates = [f['candidates'][name] for f in quality['fixtures']]
        out['quality'][name] = {}
        for section in ('fidelity', 'delta'):
            rows = [c[section] for c in candidates if section in c]
            out['quality'][name][section] = {
                k: [min(r[k] for r in rows), max(r[k] for r in rows)] for k in rows[0]
            }
    for name, row in ablations['results'].items():
        out['screening'][name] = {'median_mean_ms': median(r['mean_ms'] for r in row['timings']),
                                  'owned_bytes': row['owned_bytes']}
    (root / 'extended_summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
