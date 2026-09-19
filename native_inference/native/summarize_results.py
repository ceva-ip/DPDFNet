"""Summarize measured precision results without treating numerical drift as quality loss."""
import hashlib
import json
from pathlib import Path
from statistics import median


def main():
    path = Path('results/precision_comparison.json')
    data = json.loads(path.read_text())
    # The initial run's label predated the perceptual evaluation. Correct only
    # its schema/interpretation; keep every measured number unchanged.
    for key in ('fp16_parity', 'int8_native_parity'):
        for row in data[key]:
            if 'quality_preserving_gate_pass' in row:
                row['fp32_numerical_equivalence_gate_pass'] = row.pop('quality_preserving_gate_pass')
    data['gate_interpretation'] = 'Numerical equivalence to FP32 only; not a reduced-precision perceptual quality decision'
    data['int8_native_scope'] = '128 DPRNN matrices; per-output W8, per-row dynamic asymmetric A8, FP32 state/norm'
    data['config'] = {'warmup_frames': 100, 'timed_frames': 1000, 'repeats': 3, 'threads': 1}
    path.write_text(json.dumps(data, indent=2)+'\n')
    summary = {'cpu': data['cpu'], 'timings': {}}
    for key in ('timings', 'paced_timings'):
        table = {}
        original = median(x['mean_ms'] for x in data[key]['original_onnx'])
        hybrid = median(x['mean_ms'] for x in data[key]['previous_hybrid'])
        for name, rows in data[key].items():
            ms = median(x['mean_ms'] for x in rows)
            table[name] = {'median_mean_ms': ms, 'reduction_vs_original_percent': 100*(1-ms/original),
                           'reduction_vs_previous_hybrid_percent': 100*(1-ms/hybrid),
                           'p99_range_ms': [min(x['p99_ms'] for x in rows), max(x['p99_ms'] for x in rows)],
                           'over_10ms_total': sum(x['over_10ms'] for x in rows)}
        summary['timings'][key] = table
    quality = json.loads(Path('results/perceptual_quality.json').read_text())
    assert len(quality['fixtures']) == 7, 'Quality evaluation is incomplete'
    summary['quality'] = {}
    for name in ('c_fp32', 'c_fp16', 'c_int8', 'hybrid_int8'):
        speech = [x['fidelity_to_fp32'][name] for x in quality['fixtures'][:3]]
        paired = [x['delta_from_original'][name] for x in quality['fixtures'][3:]]
        summary['quality'][name] = {'hushmic_fidelity_range': {
            key: [min(x[key] for x in speech), max(x[key] for x in speech)] for key in speech[0]},
            'clean_reference_delta_range': {key: [min(x[key] for x in paired), max(x[key] for x in paired)] for key in paired[0]}}
    summary['artifact_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        Path('build/full/libdpdf_full.so'), Path('build/baseline/libdpdf_ort.so'),
        Path('models/full_c/generated_model.c'), Path('models/full_c/weights.f32')]}
    Path('results/full_precision_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
