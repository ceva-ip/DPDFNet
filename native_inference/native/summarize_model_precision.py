"""Summarize a generic model_precision_final/quality result pair."""
import argparse
import json
from pathlib import Path
from statistics import median


def ranges(rows):
    return {key: [min(row[key] for row in rows), max(row[key] for row in rows)]
            for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('final', type=Path)
    parser.add_argument('quality', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    final = json.loads(args.final.read_text())
    quality = json.loads(args.quality.read_text())
    result = {
        'model': final['model'],
        'environment': final['environment'],
        'precision_scope': final['precision_scope'],
        'timing': {},
        'memory': {},
        'quality': {},
        'fp32_parity': final['fp32_parity'],
    }
    for pacing in ('continuous', 'paced'):
        result['timing'][pacing] = {}
        for name, rows in final[pacing].items():
            result['timing'][pacing][name] = {
                'median_mean_ms': median(row['mean_ms'] for row in rows),
                'p99_range_ms': [min(row['p99_ms'] for row in rows),
                                 max(row['p99_ms'] for row in rows)],
                'over_10ms_total': sum(row['over_10ms'] for row in rows),
            }
    for name, rows in final['memory'].items():
        result['memory'][name] = {
            key: median(row[key] for row in rows) for key in rows[0]
        }
    for name in quality['selected']:
        candidates = [fixture['candidates'][name] for fixture in quality['fixtures']]
        result['quality'][name] = {
            'fidelity': ranges([candidate['fidelity'] for candidate in candidates]),
            'clean_reference_delta': ranges([
                candidate['delta'] for candidate in candidates if 'delta' in candidate
            ]),
            'dnsmos': ranges([candidate['dnsmos'] for candidate in candidates]),
            'dnsmos_delta_from_original_fp32': ranges([
                candidate['dnsmos_delta'] for candidate in candidates
            ]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
