"""Summarize measured W7 follow-up results without pooling run percentiles."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timing(data, phase):
    runs = data[phase]
    result = {}
    for name in ('baseline', 'candidate'):
        values = [r['implementations'][name] for r in runs if name in r['implementations']]
        result[name] = {key: mean(v['wall'][key] for v in values)
                        for key in ('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms')}
        result[name].update({'max_ms': max(v['wall']['max_ms'] for v in values),
                             'over_10ms': sum(v['wall']['over_10ms'] for v in values),
                             'thread_cpu_mean_ms': mean(v['thread_cpu']['mean_ms'] for v in values)})
    result['mean_reduction_percent'] = 100*(1-result['candidate']['mean_ms']/result['baseline']['mean_ms'])
    if phase != 'standalone_paced':
        result['per_repeat_mean_reduction_percent'] = [100*(1-r['implementations']['candidate']['wall']['mean_ms']/
                                                          r['implementations']['baseline']['wall']['mean_ms']) for r in runs]
    return result


def main():
    report = {'method': 'Mean of per-run means and percentiles; percentiles are not pooled. Every timing sample is retained.',
              'screening': {}, 'final': {}, 'quality': {}}
    order = ['pack32', 'epilogue', 'reciprocal', 'unroll2', 'unroll4', 'gate4', 'lto',
             'batch8', 'column_first', 'poly5', 'combined', 'combined_lto', 'pack_gate',
             'pack_gate_lto', 'exact_all', 'pgo', 'pack_poly5', 'pack_fit5']
    for name in order:
        path = RESULTS/f'w7_followup_{name}_screen.json'
        data = json.loads(path.read_text())
        report['screening'][name] = timing(data, 'continuous')
        report['screening'][name]['artifacts'] = data['artifacts']
        report['screening'][name]['parity'] = data['parity']
    for name in ('pack32', 'pack_poly5', 'pack_fit5'):
        data = json.loads((RESULTS/f'w7_followup_{name}_final.json').read_text())
        report['final'][name] = {phase: timing(data, phase) for phase in ('continuous', 'paced', 'standalone_paced')}
        report['final'][name].update({'artifacts': data['artifacts'], 'owned_bytes': data['owned_bytes']})
        wrapper_path = RESULTS/f'w7_followup_{name}_wrapper.json'
        if wrapper_path.exists():
            wrapper = json.loads(wrapper_path.read_text())
            assert wrapper['artifacts'] == data['artifacts']
            report['final'][name]['wrapper'] = {}
            for mode in ('allocating', 'preallocated'):
                values = {who: mean(r['implementations'][who]['mean_ms'] for r in wrapper['runs'] if r['mode'] == mode)
                          for who in ('baseline', 'candidate')}
                report['final'][name]['wrapper'][mode] = {
                    'mean_ms': values, 'mean_reduction_percent': 100*(1-values['candidate']/values['baseline'])}
        quality = json.loads((RESULTS/f'w7_followup_{name}_audio.json').read_text())
        if name == 'pack32':
            report['quality'][name] = {'files': len(quality['cases']), 'seconds': quality['total_audio_seconds'],
                                       'all_pcm_bit_identical': all(c['bit_identical'] for c in quality['cases']),
                                       'hops': sum(c['hops_including_flush'] for c in quality['cases'])}
        else:
            result = {'activation_accuracy': quality['activation_accuracy']}
            for scenario in ('mixture', 'low_level', 'clean', 'long_noise'):
                rows = [r for r in quality['cases'] if r['case']['scenario'] == scenario]
                if not rows:
                    continue
                result[scenario] = {'files': len(rows), 'delta': {},
                                    'minimum_output_difference_snr_db': min(r['output_difference_snr_db'] for r in rows)}
                for key in rows[0]['delta']:
                    values = [r['delta'][key] for r in rows if key in r['delta']]
                    result[scenario]['delta'][key] = {'mean': mean(values), 'min': min(values), 'max': max(values)}
            report['quality'][name] = result
    report['activation_contracts'] = json.loads((RESULTS/'w7_followup_activation_contracts.json').read_text())
    report['fitted_coefficients'] = json.loads((ROOT/'scratch/w7_followup/pack_fit5_coefficients.json').read_text())
    report['sources'] = {p.name: sha(p) for p in Path(__file__).parent.glob('*w7_followup*.py')}
    report['libraries'] = {p.parent.name: sha(p) for p in (ROOT/'build').glob('w7_followup_*/libdpdf_full.so')}
    report['sources_generated'] = {p.name: {f: sha(p/f) for f in ('int8.c', 'avx2.c', 'CMakeLists.txt')}
                                   for p in (ROOT/'scratch/w7_followup').iterdir() if p.is_dir() and (p/'int8.c').exists()}
    (RESULTS/'w7_followup_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k in ('final', 'quality')}, indent=2))


if __name__ == '__main__':
    main()
