# Latest native runtimes versus original ONNX

Measured 2026-09-21 on the i7-8700, Linux Docker/WSL2, one inference thread.
The baseline is each original FP32 ONNX file with ONNX Runtime CPU execution
and all graph optimizations enabled. Earlier follow-up percentages compared
native builds with each other; they were not savings against ONNX.

| Model | Runtime | Compute / hop | Time saved | Incremental RAM | RAM saved |
| --- | --- | ---: | ---: | ---: | ---: |
| dpdfnet8_48khz_hr | Original ONNX FP32 | 5.571 ms | 0.0% | 38.01 MiB | 0.0% |
| dpdfnet8_48khz_hr | Native FP32 | 3.207 ms | 42.4% | 17.23 MiB | 54.7% |
| dpdfnet8_48khz_hr | Selective FP16 | 2.831 ms | 49.2% | 10.93 MiB | 71.2% |
| dpdfnet8_48khz_hr | Selective INT8 | 2.381 ms | 57.3% | 7.53 MiB | 80.2% |
| dpdfnet2_48khz_hr | Original ONNX FP32 | 2.124 ms | 0.0% | 28.37 MiB | 0.0% |
| dpdfnet2_48khz_hr | Native FP32 | 1.500 ms | 29.4% | 12.73 MiB | 55.1% |
| dpdfnet2_48khz_hr | Selective FP16 | 1.156 ms | 45.6% | 8.41 MiB | 70.4% |
| dpdfnet2_48khz_hr | Selective INT8 | 1.018 ms | 52.1% | 6.03 MiB | 78.8% |

## Scope and method

Median of four run means; 100 warmup + 1000 timed hops/run, one inference thread; standalone 10 ms cadence, rotating mode order. Includes output allocation and Python call overhead. No outliers removed.

Median of four fresh processes per mode: warmed RSS minus RSS after identical Python/NumPy/ORT imports, before model loading. Includes model/session, allocator retention, native library and stream buffers; excludes common imported runtime baseline. Not total application RSS or model file size.

Memory probes run 120 inference hops before the final RSS sample. Source
native weights are unmapped before sampling. No ONNX reference session is
created in a native memory process. Raw results also include total/peak
process RSS and exact native owned bytes; these are distinct measures.
The common imported runtime baseline is excluded from the table, so these
are not whole-application RAM reductions or model-download size savings.

Compute measurements include spectral model inference only: FFT, resampling
and HushMic/PipeWire integration are excluded. The existing 50 ms model
delay is unchanged. Speedups are measured on this CPU, not guaranteed on
all machines. VNNI is not required; existing AVX2/FMA dispatch and FP32
scalar fallback remain. Reduced-precision modes are approximate relative
to ONNX FP32. Fresh 300-frame native FP32 numerical parity checks pass.

## Cadence tails

Each row covers 4,000 timed calls. No outliers are discarded.

| Model | Runtime | Run p99 range | Calls >10 ms |
| --- | --- | ---: | ---: |
| dpdfnet8_48khz_hr | Original ONNX FP32 | 6.269–7.055 ms | 1 |
| dpdfnet8_48khz_hr | Native FP32 | 3.668–3.845 ms | 0 |
| dpdfnet8_48khz_hr | Selective FP16 | 3.048–3.484 ms | 0 |
| dpdfnet8_48khz_hr | Selective INT8 | 2.653–2.932 ms | 0 |
| dpdfnet2_48khz_hr | Original ONNX FP32 | 2.805–3.551 ms | 0 |
| dpdfnet2_48khz_hr | Native FP32 | 1.927–2.207 ms | 0 |
| dpdfnet2_48khz_hr | Selective FP16 | 1.536–1.620 ms | 0 |
| dpdfnet2_48khz_hr | Selective INT8 | 1.305–1.839 ms | 0 |

This finite shared-host sample is not a worst-case latency guarantee.

## Listening comparison and reproduction

The [listening comparison](../listening_comparison/index.html) is regenerated
using `build/followup_final8` / `build/followup_final2` and the current
`models/rework8` / `models/rework2` weights. Its manifest records source,
library and output WAV hashes, and checks each WAV against the previous set.
The three public fixtures, gain, 48 kHz / PCM24 format and delay alignment
are unchanged. No additional audio-quality improvement is claimed.

From `native_inference/` in the existing offline Linux development image,
after building the [latest libraries](LATENCY_FOLLOWUP.md#reproduce):

```sh
cc -O2 native/memory_probe.c -ldl -o build/memory_probe
python native/onnx_comparison.py
python native/listening_samples.py
```

Raw results: [summary](../results/onnx_latest_summary.json),
[DPDFNet-8](../results/dpdfnet8_onnx_latest.json),
[DPDFNet-2](../results/dpdfnet2_onnx_latest.json).
