# DPDFNet native inference investigation

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
