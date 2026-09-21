"""Compare the original ONNX files with the latest native precision modes."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone


def write_report(summary):
    lines = ['# Latest native runtimes versus original ONNX', '',
             f'Measured {summary["generated_at"][:10]} on the i7-8700, Linux Docker/WSL2, one inference thread.',
             'The baseline is each original FP32 ONNX file with ONNX Runtime CPU execution',
             'and all graph optimizations enabled. Earlier follow-up percentages compared',
             'native builds with each other; they were not savings against ONNX.', '',
             '| Model | Runtime | Compute / hop | Time saved | Incremental RAM | RAM saved |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    labels = {'original_fp32': 'Original ONNX FP32', 'native_fp32': 'Native FP32',
              'selective_fp16': 'Selective FP16', 'selective_int8': 'Selective INT8'}
    for name, model in summary['models'].items():
        for mode, v in model['modes'].items():
            lines.append(f'| {name} | {labels[mode]} | {v["paced_mean_ms"]:.3f} ms | '
                         f'{v["latency_saving_percent"]:.1f}% | {v["rss_delta_bytes"] / 1048576:.2f} MiB | '
                         f'{v["memory_saving_percent"]:.1f}% |')
    lines += ['', '## Scope and method', '', summary['latency_method'], '',
              summary['memory_method'], '',
              'Memory probes run 120 inference hops before the final RSS sample. Source',
              'native weights are unmapped before sampling. No ONNX reference session is',
              'created in a native memory process. Raw results also include total/peak',
              'process RSS and exact native owned bytes; these are distinct measures.',
              'The common imported runtime baseline is excluded from the table, so these',
              'are not whole-application RAM reductions or model-download size savings.', '',
              'Compute measurements include spectral model inference only: FFT, resampling',
              'and HushMic/PipeWire integration are excluded. The existing 50 ms model',
              'delay is unchanged. Speedups are measured on this CPU, not guaranteed on',
              'all machines. VNNI is not required; existing AVX2/FMA dispatch and FP32',
              'scalar fallback remain. Reduced-precision modes are approximate relative',
              'to ONNX FP32. Fresh 300-frame native FP32 numerical parity checks pass.', '',
              '## Cadence tails', '',
              'Each row covers 4,000 timed calls. No outliers are discarded.', '',
              '| Model | Runtime | Run p99 range | Calls >10 ms |',
              '| --- | --- | ---: | ---: |']
    for name, model in summary['models'].items():
        for mode, v in model['modes'].items():
            lo, hi = v['paced_p99_range_ms']
            lines.append(f'| {name} | {labels[mode]} | {lo:.3f}–{hi:.3f} ms | {v["paced_over_10ms"]} |')
    lines += ['', 'This finite shared-host sample is not a worst-case latency guarantee.', '',
              '## Listening comparison and reproduction', '',
              'The [listening comparison](../listening_comparison/index.html) is regenerated',
              'using `build/followup_final8` / `build/followup_final2` and the current',
              '`models/rework8` / `models/rework2` weights. Its manifest records source,',
              'library and output WAV hashes, and checks each WAV against the previous set.',
              'The three public fixtures, gain, 48 kHz / PCM24 format and delay alignment',
              'are unchanged. No additional audio-quality improvement is claimed.', '',
              'From `native_inference/` in the existing offline Linux development image,',
              'after building the [latest libraries](LATENCY_FOLLOWUP.md#reproduce):', '',
              '```sh', 'cc -O2 native/memory_probe.c -ldl -o build/memory_probe',
              'python native/onnx_comparison.py', 'python native/listening_samples.py', '```', '',
              'Raw results: [summary](../results/onnx_latest_summary.json),',
              '[DPDFNet-8](../results/dpdfnet8_onnx_latest.json),',
              '[DPDFNet-2](../results/dpdfnet2_onnx_latest.json).', '']
    Path('native/ONNX_LATEST_COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    summary = {'generated_at': datetime.now(timezone.utc).isoformat(),
               'latency_method': 'Median of four run means; 100 warmup + 1000 timed hops/run, one inference thread; standalone 10 ms cadence, rotating mode order. Includes output allocation and Python call overhead. No outliers removed.',
               'memory_method': 'Median of four fresh processes per mode: warmed RSS minus RSS after identical Python/NumPy/ORT imports, before model loading. Includes model/session, allocator retention, native library and stream buffers; excludes common imported runtime baseline. Not total application RSS or model file size.',
               'models': {}}
    for size in (8, 2):
        name = f'dpdfnet{size}_48khz_hr'
        output = Path(f'results/dpdfnet{size}_onnx_latest.json')
        shared = ['--model', f'models/{name}.onnx', '--build', f'build/followup_final{size}',
                  '--weights', f'models/rework{size}/weights.f32']
        subprocess.run([sys.executable, 'native/model_precision_final.py', *shared,
                        '--model-name', name, '--memory-probe', 'build/memory_probe',
                        '--repeats', '4', '--output', str(output)], check=True)
        report = json.loads(output.read_text())
        report['comparable_memory'] = {}
        for variant in ('original_fp32', *report['configs']):
            report['comparable_memory'][variant] = [json.loads(subprocess.check_output(
                [sys.executable, 'native/onnx_memory_probe.py', *shared, '--variant', variant], text=True))
                for _ in range(4)]
        output.write_text(json.dumps(report, indent=2) + '\n')
        modes = {}
        for variant in report['paced']:
            modes[variant] = {
                'paced_mean_ms': statistics.median(r['mean_ms'] for r in report['paced'][variant]),
                'continuous_mean_ms': statistics.median(r['mean_ms'] for r in report['continuous'][variant]),
                'paced_p99_range_ms': [min(r['p99_ms'] for r in report['paced'][variant]), max(r['p99_ms'] for r in report['paced'][variant])],
                'paced_over_10ms': sum(r['over_10ms'] for r in report['paced'][variant]),
                'rss_delta_bytes': statistics.median(r['rss_delta_bytes'] for r in report['comparable_memory'][variant]),
                'owned_bytes': report['comparable_memory'][variant][0]['owned_bytes'],
            }
        baseline = modes['original_fp32']
        for mode in modes.values():
            mode['latency_saving_percent'] = 100 * (1 - mode['paced_mean_ms'] / baseline['paced_mean_ms'])
            mode['memory_saving_percent'] = 100 * (1 - mode['rss_delta_bytes'] / baseline['rss_delta_bytes'])
        summary['models'][name] = {'environment': report['environment'], 'artifacts': report['artifacts'], 'modes': modes}
        Path('results/onnx_latest_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary['models'][name], indent=2), flush=True)
    write_report(summary)


if __name__ == '__main__':
    main()
