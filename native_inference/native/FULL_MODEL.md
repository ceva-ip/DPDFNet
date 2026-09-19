# Complete C model and reduced-precision experiments

**Follow-up:** [FC/CNN precision and memory experiments](EXTENDED_PRECISION.md)
now extend these modes and remove retained FP32 weight copies. The measurements
and memory description below document the earlier complete-model milestone.

The entire `dpdfnet8_48khz_hr` **spectral model now runs in C11**, without an
ONNX Runtime or C++ dependency at inference time. Python/ONNX are used only
to generate the fixed graph and to compare results. HushMic's existing Rust
STFT/iSTFT and audio worker remain the intended surrounding integration.

The portable default retains FP32 weights and arithmetic, with runtime AVX2/FMA
dispatch on supported x86 CPUs. Two explicit experiments are also implemented:

| C mode | What changes | What stays FP32 |
| --- | --- | --- |
| `DPDF_AUTO` / `DPDF_AVX2` | Specialized fixed graph and SIMD kernels | All model weights, features and state |
| `DPDF_EXPERIMENTAL_FP16` | All eight matrix projections in each of 16 DPRNN blocks load FP16 weights through F16C | Arithmetic, activations, biases, normalization, recurrent state, remaining model |
| `DPDF_EXPERIMENTAL_INT8` | Those same DPRNN projections use per-output-channel W8 and dynamically quantized A8 | Bias/dequantization, gates, normalization, recurrent state, remaining model |

A separate ORT experiment quantizes the ten dense projections in the remaining
256-unit GRUs. This is a different scope from the native INT8 implementation;
the experiments are not combined. `AUTO` never selects reduced precision.

## Measured results

Intel i7-8700, Linux x86-64 under Docker Desktop/WSL2, GCC 12.2, ORT 1.27,
one inference thread. Median of three means, each with 100 warmup and 1,000
timed frames, rotating candidate order. FFT/iFFT and HushMic are outside these
invocation timings. These are development-machine measurements, not release
or bare-metal latency guarantees.

| Backend | Continuous mean | Paced mean (10 ms hops) | Continuous time saved vs original |
| --- | ---: | ---: | ---: |
| Original ONNX | 5.576 ms | 5.722 ms | — |
| Preserved earlier C/ORT hybrid | 3.664 ms | 3.929 ms | 34.3% |
| Complete C, FP32 | 3.437 ms | 3.547 ms | 38.4% |
| Complete C, FP16 DPRNN weights | 3.286 ms | 3.355 ms | 41.1% |
| Complete C, INT8 DPRNN matrices | **3.155 ms** | **3.340 ms** | **43.4%** |
| Earlier hybrid + ten INT8 GRU projections | 3.397 ms | 3.678 ms | 39.1% |

Native INT8 is 13.9% faster than the prior hybrid in continuous runs and 15.0%
in paced runs. Its paced p99 ranged from 3.91 to 4.14 ms; no native C mode
exceeded 10 ms in either 3,000-call series in this comparison. Earlier runs
did encounter scheduling outliers, recorded in `full_c.json`; these results
do not establish deadline guarantees under load. The 2–3 ms target remains
unreached on this host.

| Quality measurement | Native FP16 | Native INT8 |
| --- | ---: | ---: |
| HushMic fixture fidelity PESQ-WB, relative to original output | 4.64389 | 4.64015–4.64260 |
| HushMic fixture fidelity STOI, relative to original output | >0.9999999 | 0.999887–0.999936 |
| HushMic fixture output agreement SNR | 80.7–85.8 dB | 47.4–51.5 dB |
| Clean-reference PESQ change vs original, four controlled mixtures | +0.000002 to +0.000130 | -0.010104 to +0.000040 |
| Clean-reference STOI change vs original, four controlled mixtures | +0.000003 to +0.000014 | -0.000709 to -0.000215 |

FP16 has negligible measured quality differences on this set. INT8 is very
close, but the clean-reference metrics do show small decreases; it is not
honest to claim zero loss. FP16 is the more conservative reduced-precision
candidate: paced performance here was almost identical to INT8, with much
smaller output differences. Neither experiment establishes general perceptual
equivalence across speakers, microphones or long sessions. FP32 remains default.

Evidence: [summary](../results/full_precision_summary.json),
[all timing repeats](../results/precision_comparison.json),
[perceptual and clean-reference scores](../results/perceptual_quality.json),
[full FP32 parity](../results/full_c.json),
[scalar parity](../results/full_scalar_validation.json),
[independent-stream concurrency](../results/full_state_validation.json).

## What changed

- Generated straight-line C for all remaining operators: convolutions, grouped
  projections, 256-unit GRUs, reshaping, feature normalization, mask generation
  and complex deep filtering. Unsupported graphs fail export; the original
  source checksum is mandatory.
- Batched 1x1 convolutions, cache-contiguous dense weight tiles, aligned DPRNN
  weights, vectorized ReLU and matrix tails. No global `-march=native` or
  `-ffast-math`.
- No allocation inside `dpdf_model_process`. A model instance owns its scratch
  arena; use separate instances for concurrent calls. Stream state is explicit.
- Native W8A8 uses all 255 symmetric signed weight levels. Activation range is
  selected per row at each call, using 254 asymmetric steps, with a zero-point
  correction. Its AVX2 sign-transfer dot avoids `-128` and saturating pair sums:
  `2*127*127 = 32258 < 32767`. No calibration freeze or static activation range.

These kernels are original implementation work; no faster-enhancer.c code was
copied. Its documented quantization approach motivated the additional tests.

## Quality interpretation

The FP32 port uses strict numerical/state comparison plus waveform agreement
to detect implementation bugs. That criterion is **not** a perceptual quality
gate for reduced precision. A lower waveform SNR does not, by itself, mean
speech quality deteriorated. Likewise, high output-to-output STOI alone does
not establish unchanged enhancement quality against clean speech.

`quality_probe.py` reports both:

1. PESQ-WB, STOI, eSTOI and SNR relative to the original model on three HushMic
   speech/noise fixtures.
2. The same metrics against known clean speech on four controlled mixtures:
   fan-like and typing-like synthetic noise at -5 and +5 dB input SNR. Clean
   speech is HushMic's public-domain LibriVox source. Alignment is measured
   once from original FP32 and then held fixed for every candidate.

This is a small evaluation involving one speaker, not an 824-utterance
VoiceBank-DEMAND evaluation or a listening study. Reduced precision remains
opt-in until broader data and real HushMic sessions validate it. Audio files
for listening are generated under `scratch/quality_audio/` and are not committed.

## Build

Use the development image and pinned source download described in
[README.md](README.md). From the repository root in PowerShell:

```powershell
& native_inference/.venv/Scripts/python.exe native_inference/native/generate_model.py native_inference/models/dpdfnet8_48khz_hr.onnx native_inference/models/full_c
$nativeDir = (Resolve-Path native_inference).Path
$nativeMount = "type=bind,source=$nativeDir,target=/bench"
docker run --rm --network none --mount $nativeMount --entrypoint cmake dpdfnet-native-dev -S native -B build/full '-DDPDF_GENERATED_MODEL=/bench/models/full_c/generated_model.c'
docker run --rm --network none --mount $nativeMount --entrypoint cmake dpdfnet-native-dev --build build/full -j 4
docker run --rm --network none --mount $nativeMount --entrypoint ctest dpdfnet-native-dev --test-dir build/full --output-on-failure
```

On Linux, run the same CMake commands directly, setting `DPDF_GENERATED_MODEL`
to an absolute local path. CMake does not require ORT headers or C++ for this
target. `-DDPDF_ENABLE_AVX2=OFF` produces an FP32 scalar build; explicit FP16/INT8
requests fail there. `-DDPDF_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo`
enables ASan/UBSan in a separate build folder.

The full-model contract covers invalid arguments, in-place spectrum/state,
reset, finite outputs and unsupported-ISA rejection. An independent INT8 test
exercises saturation limits, zero/positive/negative activations and batch tails.
ASan/UBSan instrumentation remains enabled; CTest disables ASan's SIGSEGV
handler (`handle_segv=0`) because an initial Docker/WSL run recursively printed
`DEADLYSIGNAL` and timed out. Native faults still fail the test. The direct
instrumented block test passed. The initial failure is retained in
[the condensed log](../results/full_sanitizer_initial_failure.log); final
results are in [the sanitizer log](../results/full_sanitizer.log).
The final release build passed all three contracts, the scalar-only build
passed both applicable contracts, and the bounded sanitizer run passed all
three. Four concurrent independent streams matched serial output bit for bit
in each of FP32, FP16 and INT8 mode.

Outputs: `build/full/libdpdf_full.so`, `models/full_c/weights.f32`, and an export
manifest recording both the source-model and packed-weight checksums. The
weight blob is little-endian float32. Validate its size and SHA-256 before
calling the C constructor. Generated source and binaries remain Git-ignored;
the checked-in generator reproduces them.

## C interface

See [full_model.h](full_model.h). After loading and validating the weights:

```c
dpdf_model *model = dpdf_model_create(weights, count, DPDF_AUTO);
/* Check model != NULL. Allocate state[90228], spec[962], enhanced[962]. */
dpdf_model_init_state(state);  /* Includes learned normalization seeds. */
dpdf_model_process(model, spec, state, enhanced, state);
dpdf_model_destroy(model);
```

Each hop contains 481 interleaved complex bins, matching the existing ONNX
`[1,1,481,2]` interface at 48 kHz / 480 samples per hop. Both spectrum and state
can be updated in place; other overlap is invalid. Reset state at stream reset.
Do not share a mutable model instance between simultaneous calls. Separate
streams can alternate on an instance only when each supplies its own state.

The model uses about 14.53 MB of copied graph weights, 4.90 MB of arena, and
5.60 MB of copied DPRNN weights, plus small bookkeeping. Experimental FP16/INT8
adds a compressed DPRNN cache; it currently retains the FP32 copies. These
experiments reduce matrix bandwidth, not total resident memory. DPRNN processing
uses roughly 134 KiB of stack plus quantization scratch in INT8 mode.

## Reproduce comparisons

`full_probe.py` validates the full FP32 path; `precision_probe.py` compares
original ONNX, the earlier hybrid, C FP32, C FP16, C INT8 and selective ORT
INT8. Their `build/baseline/libdpdf_ort.so` is the preserved earlier hybrid
binary in this workspace. On a fresh checkout, first build the hybrid using
README.md and supply a copy there; that compares the rebuilt hybrid kernels,
not necessarily the exact historical binary. Avoid concurrent benchmark jobs.

```powershell
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-dev quantize_probe.py models/native_blocks/hybrid.onnx models/hybrid_int8.onnx
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-dev native/precision_probe.py
docker build -t dpdfnet-native-quality -f native_inference/native/Dockerfile.quality native_inference
New-Item -ItemType Directory -Force native_inference/scratch | Out-Null
Invoke-WebRequest 'https://archive.org/download/spc266_2508_librivox/spc266_afterlove_pac_128kb.mp3' -OutFile native_inference/scratch/quality_voice.mp3
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-quality native/quality_probe.py
```

The voice source is Sara Teasdale's *After Love*, LibriVox SPC 266, reader
`pac`, public domain. See HushMic's pinned `docs/demo/ASSETS.md` for fixture
noise credits (including CC BY 4.0 Gravity Sound and C40115). Existing fixture
audio stays in the reference checkout; this folder redistributes no audio.

## HushMic deployment boundary

Only Linux x86-64 Docker/WSL2 builds have been executed. The examined HushMic
release target is Linux x86-64, built against Ubuntu 22.04 / glibc 2.35. This
Bookworm development image has glibc 2.36 and must not supply release binaries.
Build against HushMic's existing release floor, then verify on bare-metal Linux,
lower-power CPUs and under PipeWire/screen-sharing load. Windows, macOS, ARM
and their compilers are unvalidated; portable scalar code is not proof of
support on those platforms. ARM NEON/VNNI-specific kernels are not implemented.

This folder does not install a HushMic backend. Integration still needs Rust
FFI, library/weight packaging, independent state for mode transitions and the
existing latency/alignment/attenuation behavior. Model computation time is
separate from the unchanged audio pipeline delay.
