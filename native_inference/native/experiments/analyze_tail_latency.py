"""Read complete traces; compare repeated frames and attribute observed events."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    report = json.loads((ROOT/'results/w7_tail_latency.json').read_text())
    raw = np.load(ROOT/'results/w7_tail_latency.npz')
    result = {'trace_sha256': hashlib.sha256((ROOT/'results/w7_tail_latency.npz').read_bytes()).hexdigest(),
              'conditions': {}, 'repeatability': {}}
    for condition, implementations in report['pooled'].items():
        result['conditions'][condition] = {}
        for name, stats in implementations.items():
            runs = [raw[f'{condition}_{name}_{i}'] for i in range(report['repeats'])]
            trace = np.concatenate(runs)
            # This includes CPU changes during the sleep interval, which may
            # affect caches but are not an involuntary switch during inference.
            changed = np.concatenate([np.r_[False, r[1:, 4]!=r[:-1, 5]] for r in runs])
            flags = ((trace[:, 6]+trace[:, 7]+trace[:, 8]+trace[:, 9]+trace[:, 11])>0) | (trace[:, 4]!=trace[:, 5]) | changed
            item = {'mean_ms': stats['wall']['mean_ms'], 'p99_ms': stats['wall']['p99_ms'],
                    'p99.9_ms': stats['wall']['p99.9_ms'], 'max_ms': stats['wall']['max_ms'],
                    'over_3ms': stats['wall']['over_3ms'], 'finish_max_ms': stats['finish_from_release']['max_ms'],
                    'finish_over_10ms': stats['finish_from_release']['over_10ms'],
                    'observations': stats['observations'], 'gc_total_ms': stats['gc_total_ms'],
                    'cpu_changes_between_hops': int(changed.sum()),
                    'over_3ms_without_any_observed_flag_including_between_hops': int(((trace[:, 0]>3000000) & ~flags).sum()),
                    'run_maxima_ms': [float(r[:, 0].max()/1e6) for r in runs],
                    'slowest': stats['slowest_calls'][0]}
            result['conditions'][condition][name] = item
            if len(runs)==2:
                # Identical input sequence and reset state in both repetitions.
                assert np.array_equal(runs[0][:, 12], runs[1][:, 12])
                top = [set(np.argsort(r[:, 0])[-15:].tolist()) for r in runs]
                result['repeatability'][f'{condition}_{name}'] = {
                    'same_input_frame_latency_correlation': float(np.corrcoef(runs[0][:, 0], runs[1][:, 0])[0, 1]),
                    'top_15_slowest_frame_overlap': len(top[0]&top[1]),
                    'slowest_frame_each_repeat': [int(r[np.argmax(r[:, 0]), 12]) for r in runs]}
    target = ROOT/'results/w7_tail_analysis.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
