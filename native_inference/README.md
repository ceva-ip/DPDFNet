# DPDFNet native inference investigation

**Progress overview — `dpdfnet8_48khz_hr` (48 kHz).** The latest optimized
research candidate is **W7A8 + pack32 + fitted degree-5 GRU gates**
(`build/w7_followup_pack_fit5`). FP16 and INT8 below are the selective native
precision presets; W7A8 uses 7-bit weights and 8-bit activations in its quantized
kernels. The production INT8 preset remains unchanged.

**Latency and memory footprint** — Intel i7-8700, one inference thread,
Linux Docker/WSL2, 10 ms audio hops:

| Version | Inference / hop ↓ | Warmed incremental RSS ↓ | Native owned allocations ↓ |
| --- | ---: | ---: | ---: |
| Original ONNX FP32 | 5.571 ms | 38.01 MiB | Not available |
| Native selective FP16 | 2.831 ms | 10.93 MiB | 9.72 MiB |
| Native selective INT8 | 2.381 ms | 7.53 MiB | 6.43 MiB |
| Latest W7A8 + pack32 + fitted gates | 1.921 ms | 7.47 MiB | 6.43 MiB |

The first three rows use the [2026-09-21 matched comparison](native/ONNX_LATEST_COMPARISON.md)
(median of four run means). The latest row uses the
[2026-09-27 follow-up](native/W7_LATENCY_FOLLOWUP.md)
(mean of two standalone cadence run means). Both include Python call overhead
and output allocation, with 100 warmup and 1,000 timed hops per run. These are
saved measurements from separate sessions, so the latest-versus-ONNX difference
is indicative, not a fresh matched speedup. FFT, audio I/O and resampling are
excluded; the model's **50 ms algorithmic delay is unchanged**. Peak latency
does not improve uniformly; see the [tail investigation](native/TAIL_LATENCY_INVESTIGATION.md).

RSS is warmed process memory above the common imported-runtime baseline,
including allocator retention and stream buffers; it is not total application
RAM or model file size. Native owned allocations count memory owned by the C
model and are **not interchangeable with RSS**. The latest candidate's allocation
count is identical to INT8. Its [RSS measurement](results/overview_fitted_rss.json)
on 2026-09-27 uses the same protocol: median of four fresh processes, each with
120 warmup hops and source weights unmapped before sampling.

**Output quality** — unweighted means on the **same six EARS-WHAM_v2 mixtures**
for every row; higher is better in all columns:

| Version | PESQ-WB (16 kHz) ↑ | STOI ↑ | SI-SNR (48 kHz, dB) ↑ | SIGMOS SIG (48 kHz) ↑ | SIGMOS NOISE (48 kHz) ↑ | SIGMOS OVRL (48 kHz) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original ONNX FP32 | 2.53436 | 0.940853 | 16.6703 | 3.84674 | 4.64805 | 3.34049 |
| Native selective FP16 | 2.53442 | 0.940850 | 16.6702 | 3.84677 | 4.64758 | 3.34038 |
| Native selective INT8 | 2.53397 | 0.940635 | 16.6565 | 3.81811 | 4.66332 | 3.33587 |
| Latest W7A8 + pack32 + fitted gates | 2.52099 | 0.940449 | 16.6446 | 3.84506 | 4.63650 | 3.35663 |

The common subset is `00033`, `00046`, `00084`, `00133`, `00200`, `00364`
(six speakers). The first three rows are recomputed from the saved
[per-clip results](results/fullband_ears_wham_v2_clips.json); the latest row comes
from the mixture cases in the [fitted-gate quality screen](results/w7_followup_pack_fit5_audio.json).
Inference and audio remain at 48 kHz: SI-SNR and SIGMOS are fullband; PESQ-WB
explicitly resamples to 16 kHz, and STOI internally uses 10 kHz.
The latest candidate has only this small matched quality screen, not the full
50-clip evaluation. The [50-clip report](native/FULLBAND_EVALUATION.md) covers
ONNX, FP16, INT8 and the earlier W7A8 build. These six-clip means show small
quality differences, not proof of equivalence or an overall quality improvement.

**Progress overview — `dpdfnet2_48khz_hr` (48 kHz).** The same fitted W7A8
kernel candidate is now built for this model in `build/w7_followup_pack_fit5_2`.
It remains an experimental candidate, with the production INT8 preset unchanged.

**Latency and memory footprint** — same CPU, single-thread execution and
10 ms cadence as above:

| Version | Inference / hop ↓ | Warmed incremental RSS ↓ | Native owned allocations ↓ |
| --- | ---: | ---: | ---: |
| Original ONNX FP32 | 2.312 ms | 28.37 MiB | Not available |
| Native selective FP16 | 1.337 ms | 8.41 MiB | 7.55 MiB |
| Native selective INT8 | 1.069 ms | 6.03 MiB | 5.16 MiB |
| Latest W7A8 + pack32 + fitted gates | 0.940 ms | 6.03 MiB | 5.16 MiB |

All four latency rows were [remeasured together on 2026-09-27](results/dpdfnet2_overview_timing.json):
median of four run means, 100 warmup + 1,000 timed hops per run, rotating order,
including Python overhead and output allocation. The fitted candidate reduces
this typical latency by **12.1% versus INT8** and **59.3% versus ONNX**.
Its observed maximum was **7.605 ms**, versus **1.976 ms** for INT8; none of the
16,000 timed calls exceeded 10 ms. The mean improvement does not imply a lower
worst-case latency. These timings replace the older DPDFNet-2 values for this
overview; cross-model timing ratios remain approximate because DPDFNet-8 was
measured in separate sessions.

ONNX/FP16/INT8 RSS comes from the [original matched memory study](results/onnx_latest_summary.json).
The [new fitted-candidate RSS measurement](results/overview_fitted_rss.json) uses
the same four-process protocol on 2026-09-27. Both INT8 variants round to
6.03 MiB; the small difference between sessions is not evidence of memory savings.
Memory definitions and the unchanged 50 ms algorithmic delay are as above.

**Output quality** — newly measured on the **same six EARS-WHAM_v2 mixtures**
listed in the DPDFNet-8 overview, with the same metrics and sample-rate handling:

| Version | PESQ-WB (16 kHz) ↑ | STOI ↑ | SI-SNR (48 kHz, dB) ↑ | SIGMOS SIG (48 kHz) ↑ | SIGMOS NOISE (48 kHz) ↑ | SIGMOS OVRL (48 kHz) ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original ONNX FP32 | 2.43120 | 0.935887 | 16.1189 | 3.85311 | 4.66154 | 3.29926 |
| Native selective FP16 | 2.43123 | 0.935887 | 16.1191 | 3.85261 | 4.66126 | 3.29865 |
| Native selective INT8 | 2.42767 | 0.935712 | 16.1053 | 3.86265 | 4.66954 | 3.30783 |
| Latest W7A8 + pack32 + fitted gates | 2.44130 | 0.935874 | 16.0841 | 3.86312 | 4.61457 | 3.28124 |

These are unweighted six-clip means, not a full 50-clip evaluation. Compared
with ONNX, the fitted candidate changes mean SI-SNR by −0.0348 dB and SIGMOS
OVRL by −0.0180, while PESQ increases by 0.0101. The mixed small changes do
not establish equivalence or an overall quality improvement. All outputs and
states remained finite. See the [per-clip measurements](results/dpdfnet2_overview_quality.json)
and [validation, memory methodology and reproduction](native/MODEL_OVERVIEW_EVALUATION.md).

**HushMic integration:** [C ABI v1 and Python-free build](native/integration/README.md)
now provide symbol-prefixed models in one library, the named
`DPDF_PRESET_INT8_SELECTIVE` preset with CPU capability checks, and
[checksummed generated C, headers and weights for both models](artifacts/v1/).
The [opt-in W7A8 build](native/integration/README.md#opt-in-to-the-latest-w7a8-candidate-both-models)
exposes the latest fitted-gate candidate for **both model sizes** through
`DPDF_PRESET_W7A8_FITTED`, using the same C ABI and weights.

**Latest measurements against original ONNX:** the [matched comparison](native/ONNX_LATEST_COMPARISON.md)
measures selective INT8 at **5.57 → 2.38 ms/hop (57.3% less)** for DPDFNet-8
and **2.12 → 1.02 ms/hop (52.1% less)** for DPDFNet-2. Warmed incremental model
RAM falls **80.2% / 78.8%**, respectively. These are single-thread i7-8700
Linux/WSL2 results; the model's 50 ms audio delay is unchanged. The report also
includes native FP32 and selective FP16, memory methodology, and cadence tails.
The [listening comparison](listening_comparison/index.html) uses these latest builds.

**Latest kernels:** the [convolution, quantization and gate follow-up](native/LATENCY_FOLLOWUP.md)
continues from the completed latency/memory pass, with additional exact kernel
optimizations and direct comparisons against both the preceding and original
builds. It keeps the existing CPU requirements and scalar fallback.

**Further research:** the [INT8 range and handwritten assembly experiments](native/INT8_RANGE_EXPERIMENTS.md)
compare exact AVX2 scheduling changes with faster W7A8/W8A7 arithmetic,
including cadence timing, recurrent validation and speech-quality measurements.
These are isolated research builds; the production W8A8 preset is unchanged.

The [W7A8 latency follow-up](native/W7_LATENCY_FOLLOWUP.md) ranks further
directions and screens 18 candidate builds. Packing plus fitted GRU activations
reduces mean inference time another **3.6–4.9% beyond W7A8**, passes the existing
activation tolerance, and stays close in a small fullband quality screen.
Exact packing alone saves **1.2–2.7%**, preserving all 65 prior audio outputs
byte for byte. Tail latency does not improve uniformly; both remain research builds.

The [peak-latency investigation](native/TAIL_LATENCY_INVESTIGATION.md) traces
30,000 calls across allocating, reusable-buffer and C-only execution, with and
without CPU affinity. A new streaming runner removes the observed in-call GC
and page faults; residual peak variation remains, including in C-only runs.

**Fullband quality:** the [50-clip EARS-WHAM_v2 comparison](native/FULLBAND_EVALUATION.md)
evaluates original ONNX, selective FP16, INT8, and W7A8 with native 48 kHz
SI-SNR and SIGMOS, plus explicitly labelled PESQ/STOI. W7A8 stays close to
INT8 on average, with a small PESQ decline; paired intervals and worst cases
are included.

The [low-level and clean-speech follow-up](native/LOW_LEVEL_CLEAN_EVALUATION.md)
adds ten mixtures at −50 dBFS speech RMS and two clean controls. W7A8 remains
close to the existing presets; one quiet-speech outlier exposes a level-sensitive
suppression issue shared with original ONNX.

The [continuous noise-only tests](native/LONG_NOISE_EVALUATION.md) add three
two-minute synthetic noise streams. All four variants stayed numerically stable
with strong suppression throughout; W7A8 attenuated the inputs by 89.74–92.71 dB.

The preceding [latency and memory optimization](native/LATENCY_REWORK.md)
reduces tensor-layout work, reuses temporary storage, and accelerates exact
normalization and batched INT8 operations. It includes balanced latency and
tail measurements for both models, with unchanged CPU requirements and
byte-identical output/state in each tested precision mode. This supersedes the
small [AVX2 register-lifetime optimization](native/AVX2_OPTIMIZATION.md) result.

The preceding [`dpdfnet8_48khz_hr` architecture-first optimization](native/DPDFNET8_ARCHITECTURE.md)
reduces final selective INT8 latency by 7.8% continuously and 5.3% at real
cadence, with bit-identical output and recurrent state. The preceding
[FC/CNN precision and memory experiments](native/EXTENDED_PRECISION.md) extend
FP16/INT8 to the remaining dense and convolution layers.

The same final selective FP16/INT8 configuration now supports and has measured
results for `dpdfnet2_48khz_hr`. See the repository [README](../README.md#experimental-native-fp16--int8-results),
the [machine-readable summary](results/dpdfnet2_48khz_hr_summary.json), and the
[two-model listening comparison](listening_comparison/index.html).

**Complete C update:** the entire spectral model now runs without ONNX Runtime,
with FP32 as the default and explicit FP16-weight / INT8 experiments. See the
[full model, precision comparison and build instructions](native/FULL_MODEL.md).
The earlier DPRNN-only hybrid achieved **5.53 to 3.61 ms/hop** on this i7-8700
under Linux Docker/WSL2; its report remains in [native/README.md](native/README.md).
The original investigation below is retained as the starting baseline.

**Conclusion: a specialized CPU runtime is technically feasible and worth prototyping,
but faster-enhancer.c is not a drop-in backend for DPDFNet.** Start with the
DPDFNet recurrent blocks on Linux x86-64, preserve FP32 behavior first, then
evaluate selective int8 kernels. No evidence yet supports promising a 3.3x
DPDFNet speedup or unchanged quality after quantization.

This folder contains the feasibility investigation, reproducible graph probes,
an experimental partial-int8 converter, a native DPRNN prototype, and measured
results, and a complete standalone C spectral model. It does not install a
HushMic backend. The existing DPDFNet
package, exporters, model definitions, and downloaded source weights are unchanged.

- [Detailed findings and implementation plan](FEASIBILITY.md)
- [Complete C model, FP16 and native INT8](native/FULL_MODEL.md)
- [Convolution, quantization and gate follow-up](native/LATENCY_FOLLOWUP.md)
- [INT8 range and handwritten assembly experiments](native/INT8_RANGE_EXPERIMENTS.md)
- [Exact latency and memory optimization, both models](native/LATENCY_REWORK.md)
- [`dpdfnet8_48khz_hr` architecture and exact optimization results](native/DPDFNET8_ARCHITECTURE.md)
- [Optional Linux assembly kernels: feasibility and compiler audit](native/ASSEMBLY_FEASIBILITY.md)
- [Implemented AVX2 optimization and assembly comparison](native/AVX2_OPTIMIZATION.md)
- [Implemented C kernels and native hybrid results](native/README.md)
- [Raw benchmark results](results/)
- [Graph inventory, timing, profiling and numerical comparison](benchmark.py)
- [Experimental partial-int8 conversion](quantize_probe.py)

Created on branch `codex/dpdfnet-native-inference`, 2026-09-18. The canonical
model name in this repository and HushMic is `dpdfnet8_48khz_hr` (the model
referred to as `dpdfnet_8_48khz_hr` in the request).

## Reproduce locally

Run from the DPDFNet repository root. Python 3.11 is used for the recorded runs.

```powershell
python -m venv native_inference/.venv
native_inference/.venv/Scripts/python.exe -m pip install -r native_inference/requirements.txt
native_inference/.venv/Scripts/python.exe native_inference/download_model.py
native_inference/.venv/Scripts/python.exe native_inference/benchmark.py native_inference/models/dpdfnet8_48khz_hr.onnx --output native_inference/results/local_fp32.json --profile
```

On Linux/macOS use `native_inference/.venv/bin/python` instead. The downloader
checks HushMic's model SHA-256; it refuses changed upstream content. Packages
and models stay in this folder. References, environments, binary models and
large raw profiler traces are ignored by Git; summary JSON is retained.

## Reproduce in Linux Docker

Docker is useful on a Windows development machine for the Linux runtime probe.
This Debian Bookworm image is **not** a HushMic release-build environment: it has
glibc 2.36, whereas HushMic's examined release workflow uses Ubuntu 22.04 and
enforces a maximum required glibc symbol version of 2.35.

```powershell
docker build -t dpdfnet-native-probe native_inference
$probeDir = (Resolve-Path native_inference).Path
docker run --rm --network none --mount "type=bind,source=$probeDir,target=/bench" dpdfnet-native-probe models/dpdfnet8_48khz_hr.onnx --output results/linux_fp32.json --profile
docker run --rm --network none --mount "type=bind,source=$probeDir,target=/bench" dpdfnet-native-probe models/dpdfnet8_48khz_hr.onnx --api binding --reference models/dpdfnet8_48khz_hr.onnx --output results/linux_binding.json
docker run --rm --network none --mount "type=bind,source=$probeDir,target=/bench" dpdfnet-native-probe models/dpdfnet8_48khz_hr.onnx --paced --output results/linux_fp32_paced.json
```

Equivalent Linux mount syntax: `--mount "type=bind,source=$(pwd)/native_inference,target=/bench"`.
Run timing comparisons sequentially on an otherwise idle machine. Docker Desktop
runs a Linux VM here; these are not bare-metal Linux or PipeWire measurements.

### Reproduce the `dpdfnet2_48khz_hr` precision results

The exporter accepts only the two recorded model hashes. From the repository
root, prepare the smaller model and generated graph:

```powershell
native_inference/.venv/Scripts/python.exe native_inference/download_model.py dpdfnet2_48khz_hr
native_inference/.venv/Scripts/python.exe native_inference/native/generate_extended.py native_inference/models/dpdfnet2_48khz_hr.onnx native_inference/models/dpdfnet2_extended
$nativeDir = (Resolve-Path native_inference).Path
$nativeMount = "type=bind,source=$nativeDir,target=/bench"
docker run --rm --network none --mount $nativeMount --entrypoint cmake dpdfnet-native-dev -S native -B build/dpdfnet2_extended '-DDPDF_GENERATED_MODEL=/bench/models/dpdfnet2_extended/generated_model.c' '-DDPDF_TEST_WEIGHTS=/bench/models/dpdfnet2_extended/weights.f32' -DDPDF_EXTENDED_MODEL=ON
docker run --rm --network none --mount $nativeMount --entrypoint cmake dpdfnet-native-dev --build build/dpdfnet2_extended -j 4
docker run --rm --network none --mount $nativeMount --entrypoint ctest dpdfnet-native-dev --test-dir build/dpdfnet2_extended --output-on-failure
docker run --rm --network none --mount $nativeMount --entrypoint cc dpdfnet-native-dev -O2 native/memory_probe.c -ldl -o build/dpdfnet2_memory_probe
```

Run the final timing/memory comparison and summarize it after the quality pass:

```powershell
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-dev native/model_precision_final.py --model-name dpdfnet2_48khz_hr --model models/dpdfnet2_48khz_hr.onnx --build build/dpdfnet2_extended --weights models/dpdfnet2_extended/weights.f32 --memory-probe build/dpdfnet2_memory_probe --output results/dpdfnet2_48khz_hr_final.json
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-quality native/model_precision_quality.py --model-name dpdfnet2_48khz_hr --model models/dpdfnet2_48khz_hr.onnx --build build/dpdfnet2_extended --weights models/dpdfnet2_extended/weights.f32 --scratch scratch/dpdfnet2_extended_audio --output results/dpdfnet2_48khz_hr_quality.json
native_inference/.venv/Scripts/python.exe native_inference/native/summarize_model_precision.py native_inference/results/dpdfnet2_48khz_hr_final.json native_inference/results/dpdfnet2_48khz_hr_quality.json native_inference/results/dpdfnet2_48khz_hr_summary.json
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-dev native/listening_samples.py
```

## Partial-int8 experiment

```powershell
native_inference/.venv/Scripts/python.exe native_inference/quantize_probe.py native_inference/models/dpdfnet8_48khz_hr.onnx native_inference/models/dpdfnet8_partial_int8.onnx
docker run --rm --network none --mount "type=bind,source=$probeDir,target=/bench" dpdfnet-native-probe models/dpdfnet8_partial_int8.onnx --reference models/dpdfnet8_48khz_hr.onnx --output results/linux_partial_int8.json --profile
docker run --rm --network none --mount "type=bind,source=$probeDir,target=/bench" dpdfnet-native-probe models/dpdfnet8_partial_int8.fp32_lowered.onnx --reference models/dpdfnet8_48khz_hr.onnx --output results/linux_lowered_fp32.json --repeats 1
```

This lowers 58 Gemm nodes to MatMul + Add, then selects 74 constant-weight 2D
MatMuls for dynamic quantization with per-channel QInt8 weights and
`reduce_range=True`. GRU, convolution and grouped/batched MatMul remain FP32.
The intermediate FP32 model makes the lowering independently reviewable.
This is a deliberately limited experiment, not an optimized model to distribute.

An initial attempt to include grouped 3D weights failed in ORT 1.27's fused
`DynamicQuantizeMatMul` with `b zero point is not valid`. The converter excludes
those nodes. ORT's dynamic quantization registry also has no GRU handler.

## What is measured

- CPU EP, ORT 1.27.0 (HushMic's pinned runtime version), one intra/inter-op
  thread, sequential execution, all graph optimizations.
- 200 warmup frames, then 1,000 timed frames, three repeats unless the JSON
  says otherwise. Each repeat resets metadata-seeded state. Every frame carries
  its output state into the next; comparison uses independent recurrent states.
- Deterministic synthetic PCM, causal 960-point Vorbis-window STFT, 480-sample
  hops. Silence, noise, impulses and tones exercise recurrence. This is not a
  speech test set. STFT input preparation is outside the timing interval.
- Graph invocation plus Python API overhead only; excludes FFT/iFFT,
  resampling, Rust wrapper output copies, PipeWire, worker scheduling and UI.
  Mean/p50/p95/p99/max and calls exceeding 10 ms are recorded. A late graph call
  does not by itself imply a PipeWire dropout because HushMic buffers output.
- `--paced` releases a frame every 10 ms using ordinary sleeps and records start
  lateness separately. It is not a real-time scheduler or a loaded-call test.
- Profiling is a separate, instrumented 100-frame run, including startup.
  Its node-duration proportions identify candidates; they cannot be treated
  as uninstrumented CPU percentages or used to promise an Amdahl speedup.
- `--reference` reports numerical differences, not perceptual quality, and does
  not impose a pass/fail tolerance. Identical binding/lowering results were
  verified in the deposited probes; general equivalence needs broader inputs.

The exact model checksum is
`7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631`.
The base image resolved during this investigation to
`python:3.11-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84`.
The Dockerfile uses the named tag; pin that digest to reproduce the same base.
