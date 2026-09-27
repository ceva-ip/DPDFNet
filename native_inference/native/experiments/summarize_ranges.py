"""Summarize the recorded September 26 INT8 research runs without filtering."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'results'


def load(name):
    return json.loads((RESULTS/name).read_text())


def compare(runs, wrapper=False):
    result = {}
    for name in ('baseline', 'candidate'):
        values = [r['implementations'][name] for r in runs if name in r['implementations']]
        walls = values if wrapper else [v['wall'] for v in values]
        result[name] = {'runs': len(walls),
                        'median_run_mean_ms': statistics.median(v['mean_ms'] for v in walls),
                        'median_run_p99_ms': statistics.median(v['p99_ms'] for v in walls),
                        'largest_call_ms': max(v['max_ms'] for v in walls),
                        'calls_over_10ms': sum(v['over_10ms'] for v in walls)}
    result['reduction_percent'] = 100*(1-result['candidate']['median_run_mean_ms']/result['baseline']['median_run_mean_ms'])
    return result


def main():
    result = {'screening': {}, 'final': {}, 'quality': {}, 'validation': {}}
    for variant in ('inline', 'fixed64', 'unroll', 'asm4', 'u7', 'w7'):
        data = load(f'next_{variant}8_screen.json')
        result['screening'][variant] = compare(data['continuous'])
    for variant in ('asm4', 'w7'):
        data = load(f'range_{variant}8_final.json')
        wrapper = load(f'range_{variant}8_wrapper.json')
        result['final'][variant] = {mode: compare(data[mode]) for mode in ('continuous', 'paced', 'standalone_paced')}
        result['final'][variant]['preallocated_continuous'] = compare(
            [r for r in wrapper['runs'] if r['mode'] == 'preallocated'], wrapper=True)
        result['final'][variant]['owned_bytes'] = data['owned_bytes']
        result['final'][variant]['artifacts'] = data['artifacts']
        result['final'][variant]['comparison'] = data['parity']
    quality = load('range_precision8_quality.json')
    if len(quality['fixtures']) != 7:
        raise ValueError('Quality pass is incomplete')
    if quality['artifacts']['w7'] != result['final']['w7']['artifacts']['candidate']:
        raise ValueError('Quality and final timing used different W7A8 libraries')
    for variant in ('u7', 'w7'):
        out = {}
        for field in ('dnsmos_delta_from_baseline', 'clean_delta_from_baseline'):
            entries = [f['candidates'][variant][field] for f in quality['fixtures'] if field in f['candidates'][variant]]
            out[field] = {key: {'mean': statistics.mean(e[key] for e in entries),
                                'min': min(e[key] for e in entries), 'max': max(e[key] for e in entries)}
                          for key in entries[0]}
        result['quality'][variant] = out
    for variant in ('u7', 'w7', 'asm4'):
        result['validation'][variant] = load(f'range_{variant}8_validation.json')
    result['contracts'] = load('range_contracts.json')
    (RESULTS/'range_experiments_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('screening', 'final', 'quality')}, indent=2))


if __name__ == '__main__':
    main()
