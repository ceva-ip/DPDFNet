# Native DPRNN prototype

**Follow-up:** the complete model and reduced-precision experiments are now
implemented. See [FULL_MODEL.md](FULL_MODEL.md) for current results and usage.
The report below records the earlier hybrid milestone; remaining-layer comments
describe that milestone, not the current standalone implementation.

Implemented and measured on 2026-09-18, on branch `codex/dpdfnet-native-inference`.

**The working hybrid reduces whole-model graph time from 5.53 to 3.61 ms per
10 ms audio hop on this i7-8700: 34.8% less time, or 1.53x throughput.** The
weights remain FP32. All 16 DPRNN blocks execute the new C implementation;
ONNX Runtime still executes the remaining model layers. This is a prototype,
not a complete standalone DPDFNet runtime or an installed HushMic backend.

## Measurements

Linux x86-64 under Docker Desktop/WSL2, GCC 12.2, ORT 1.27.0, one inference
thread. Median of three repeat means, each with 100 warmup and 1,000 timed
frames. Reference/candidate order alternates across repeats. Synthetic audio
is prepared before timing; FFT/iFFT, PipeWire and the HushMic wrapper are not
timed. These are whole-graph invocation times, not isolated C kernel timings.

| Measurement | Original ONNX | Native hybrid | Reduction in mean time |
| --- | ---: | ---: | ---: |
| Frames processed continuously | 5.530 ms | 3.609 ms | 34.8% |
| Frames released every 10 ms | 5.750 ms | 3.893 ms | 32.3% |
| Continuous p99, range across repeats | 6.40–8.30 ms | 4.07–5.07 ms | — |
| Paced p99, range across repeats | 6.69–9.63 ms | 4.67–5.13 ms | — |

The hybrid had no calls above 10 ms in either 3,000-frame timing series. This
does not establish dropout-free behavior under competing load or on another
CPU. Docker/WSL scheduling, CPU clocks and background activity remain factors.
The original 2–3 ms development target has **not** been reached.

[Full timing results](../results/native_prototype.json),
[extended validation](../results/native_validation.json),
[scalar-only validation](../results/native_scalar_validation.json).
The earlier `native_quick.json` is a short smoke run, not the headline result.

## What was implemented

- The complete fixed-shape DPRNN block: forward/reverse frequency GRUs,
  intra projection, LayerNorm and residual, per-frequency temporal GRU,
  inter projection, LayerNorm and residual. Both F=40 and F=48, hidden size 64.
- A standalone C ABI with immutable copied weights and explicit caller-owned
  temporal state; portable scalar and runtime-dispatched AVX2/FMA kernels.
- Prepacked input-major matrices, batched affine kernels and fused GRU gates.
  The C process function uses about 133 KiB of stack scratch and performs no
  heap allocation. A single context can serve concurrent independent streams.
- FP32 weights, features and hidden state. AVX2 uses explicit FMA and a
  range-reduced degree-7 exponential for sigmoid/tanh. LayerNorm accumulates
  mean/variance in double precision. Small floating-point differences are
  expected; this is not bit-identical ONNX arithmetic or quantization.
- A C++ adapter registering the C kernel as an ORT CPU custom operator. It
  checks shapes and translates exceptions to ORT errors. ORT and this adapter
  still have their own allocation/lifetime behavior; the whole hybrid is not
  claimed to be allocation-free.
- An exporter restricted to the exact audited model checksum. It extracts
  original ONNX block oracles, packs weights with explicit ONNX-to-PyTorch GRU
  gate reordering, and replaces block internals in the hybrid. The graph shrinks
  from 937 to 377 nodes. Original state slices and output names are preserved.

The implementation is original code in this repository; no faster-enhancer.c
source was copied. Its specialization approach informed the investigation.

## Validation performed

1. All 16 blocks, scalar and AVX2, against their original ONNX subgraphs over
   256 recurrent frames per block/tier, including changing amplitudes and
   silence. Worst block output absolute error was about `3.9e-6`; worst state
   absolute error about `3.2e-6`. Gate: `atol=rtol=3e-4` per element.
2. Whole-model recurrent comparison over 1,100 synthetic frames and three
   HushMic public speech/noise fixtures, with independent states. Spectrum
   and state gate: `atol=rtol=2e-3`; reconstructed waveform agreement must
   exceed 70 dB signal-to-difference ratio when reference energy is sufficient.
3. Fan: **134.7 dB** waveform agreement; café: **134.1 dB**; keyboard:
   **136.0 dB**. Maximum PCM difference across these was `2.69e-7` on the
   normalized waveform scale. This measures agreement with the original
   model, not enhancement quality or improvement over noisy input. It is
   encouraging numerical parity, not a broad listening/quality evaluation.
4. 220,002 activation samples against double-precision NumPy reference;
   maximum sigmoid/tanh absolute difference `5.96e-8` for the AVX2 build.
5. Reset determinism, independent contexts, and eight streams on four threads
   sharing one immutable C context; concurrent outputs were bit-identical to
   serial execution for each stream.
6. Separate build with AVX2 completely disabled: all 16 block checks and
   300-frame synthetic whole-model comparison passed, including concurrency.
7. Native C contract tests for invalid arguments/weights, finite outputs and
   in-place spectrum/state operation passed in scalar and ASan+UBSan builds.
   The bounded fail-fast sanitizer CTest run passed in 7.7 seconds.

Fixture provenance is in HushMic's pinned
[fixture README](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/tests/fixtures/README.md)
and [asset credits](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/docs/demo/ASSETS.md).
Audio is read locally from the ignored reference checkout, not redistributed
here. Fixture hashes are recorded in the validation JSON.

## Build and reproduce on this Windows host

From the DPDFNet repository root, after the Python setup in the
[investigation README](../README.md):

```powershell
$probePython = 'native_inference/.venv/Scripts/python.exe'
& $probePython native_inference/download_model.py
& $probePython native_inference/native/fetch_ort_headers.py
& $probePython native_inference/native/export_blocks.py native_inference/models/dpdfnet8_48khz_hr.onnx native_inference/models/native_blocks
docker build -t dpdfnet-native-probe native_inference
docker build -t dpdfnet-native-dev -f native_inference/native/Dockerfile native_inference
$probeDir = (Resolve-Path native_inference).Path
$probeMount = "type=bind,source=$probeDir,target=/bench"
docker run --rm --network none --mount $probeMount --entrypoint cmake dpdfnet-native-dev -S native -B build -DDPDF_ORT_INCLUDE=/bench/vendor/onnxruntime
docker run --rm --network none --mount $probeMount --entrypoint cmake dpdfnet-native-dev --build build -j 4
docker run --rm --network none --mount $probeMount dpdfnet-native-dev --frames 1000 --block-frames 256 --repeats 3 --paced --output results/native_prototype.json
```

To include the speech checks, append one or more `--audio` options:

```powershell
docker run --rm --network none --mount $probeMount dpdfnet-native-dev --check-only --frames 1000 --block-frames 256 --audio references/HushMic/tests/fixtures/noisy_public_48k.flac --audio references/HushMic/tests/fixtures/noisy_cafe_48k.flac --audio references/HushMic/tests/fixtures/noisy_keyboard_48k.flac --output results/native_validation.json
```

The reference checkout already exists in this workspace. For a fresh workspace:

```powershell
git clone https://github.com/Fovty/HushMic.git native_inference/references/HushMic
git -C native_inference/references/HushMic checkout ede5673734f4bd8f3676c1195464bb0b146ddb8a
```

Scalar-only and memory-safety checks:

```powershell
docker run --rm --network none --mount $probeMount --entrypoint cmake dpdfnet-native-dev -S native -B build/scalar -DDPDF_ENABLE_AVX2=OFF -DDPDF_ORT_INCLUDE=/bench/vendor/onnxruntime
docker run --rm --network none --mount $probeMount --entrypoint cmake dpdfnet-native-dev --build build/scalar -j 4
docker run --rm --network none --mount $probeMount dpdfnet-native-dev --build build/scalar --check-only --frames 200 --block-frames 64 --output results/native_scalar_validation.json
docker run --rm --network none --mount $probeMount --entrypoint cmake dpdfnet-native-dev -S native -B build/sanitize -DDPDF_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
docker run --rm --network none --mount $probeMount --entrypoint cmake dpdfnet-native-dev --build build/sanitize -j 4
docker run --rm --network none --mount $probeMount --entrypoint ctest dpdfnet-native-dev --test-dir build/sanitize --output-on-failure
```

On native Linux the same sources build with ordinary CMake from
`native_inference/`; Docker is optional. Install the investigation Python
requirements plus `soundfile==0.13.1` for FLAC fixtures. Use an absolute local
path for `DPDF_ORT_INCLUDE`, or omit it to build just the standalone C library.
Only the Linux x86-64 build was executed here. Non-x86 builds select the scalar
path; ARM SIMD and other OS/toolchain builds remain untested.

## Using the hybrid

The generated artifacts stay in ignored folders:

- `build/libdpdf_dprnn.so`: standalone C block library, about 29 KB in this build.
- `build/libdpdf_ort.so`: ORT adapter plus C kernels, about 51 KB.
- `models/native_blocks/hybrid.onnx`: complete graph using the custom blocks.
- `models/native_blocks/*.onnx` and `*.f32`: extracted oracles and packed weights.

The hybrid ONNX requires registering the custom library before loading it:

```python
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
options.register_custom_ops_library('/absolute/path/libdpdf_ort.so')
session = ort.InferenceSession('/absolute/path/hybrid.onnx', options,
                               providers=['CPUExecutionProvider'])
```

The public C interface is in [dpdf_dprnn.h](dpdf_dprnn.h). The standalone
library has no ONNX dependency. Caller-owned state is `F*64` floats, initially
zero; input/output layout is `[64,F]`. It does not implement the model's full
90,228-element state, FFT or audio-hop interface by itself. The hybrid preserves
that existing model interface. Never share mutable state arrays across streams.

## Remaining work

This result justifies further native work but does not establish a 2x or 3x
whole-model gain. Remaining convolution, grouped linear, 256-unit GRU, feature
normalization and complex filtering paths still execute in ORT. Profile the
hybrid before selecting the next stage; test selective precision separately.

HushMic integration needs an explicit backend/library registration change and
packaging; simply replacing its model file will fail. Keep the current Rust
FFT, worker, alignment, attenuation/mode behavior and adaptive multi-model
transitions. Build distribution artifacts against HushMic's Ubuntu 22.04/glibc
floor; this Bookworm development image is not the release target. Validate
bare-metal Linux PipeWire under screen sharing/load, longer sessions, and
lower-power CPUs before publishing a backend or changing defaults.
