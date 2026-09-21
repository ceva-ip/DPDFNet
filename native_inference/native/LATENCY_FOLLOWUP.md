# Convolution, quantization and recurrent-gate follow-up

Continued on 2026-09-21 from the completed [latency and memory pass](LATENCY_REWORK.md).
The incremental reference is its preserved `rework_final8` / `rework_final2`
libraries, not the original slower implementation. Separate cumulative runs
compare with the original `avx2_baseline_8` / `avx2_baseline_2` libraries.

## Results

The follow-up reduces mean latency further: **4.1% / 4.5% continuously** and
**5.2% / 2.8% at paired 10 ms cadence** for DPDFNet-8 / DPDFNet-2. The paired
measurements compare adjacent calls and better control slow host drift.

| Model | Mode | Previous optimized build | Follow-up | Reduction |
| --- | --- | ---: | ---: | ---: |
| DPDFNet-8 | Continuous paired | 2.265 ms | 2.171 ms | 4.1% |
| DPDFNet-8 | Paired 10 ms cadence | 2.414 ms | 2.287 ms | 5.2% |
| DPDFNet-8 | Standalone 10 ms cadence | 2.284 ms | 2.188 ms | 4.2% |
| DPDFNet-2 | Continuous paired | 1.002 ms | 0.958 ms | 4.5% |
| DPDFNet-2 | Paired 10 ms cadence | 1.099 ms | 1.068 ms | 2.8% |
| DPDFNet-2 | Standalone 10 ms cadence | 1.082 ms | 0.999 ms | 7.7% |

These are medians of four run means. Every one of the twelve run pairs per
model improved mean latency. DPDFNet-2's standalone gains ranged from 2.8% to
13.9%, whereas its paired cadence gains ranged from 2.8% to 3.1%. The 7.7%
standalone aggregate should therefore not be treated as a stable cross-run
speedup. DPDFNet-8's standalone gains ranged from 3.6% to 4.3%.

Standalone cadence tail measurements, 4,000 timed calls per version/model:

| Model | Version | Median run p99 | Largest call | Calls >10 ms |
| --- | --- | ---: | ---: | ---: |
| DPDFNet-8 | Previous optimized | 2.855 ms | 3.652 ms | 0 |
| DPDFNet-8 | Follow-up | 2.731 ms | 3.333 ms | 0 |
| DPDFNet-2 | Previous optimized | 1.525 ms | 2.313 ms | 0 |
| DPDFNet-2 | Follow-up | 1.390 ms | 2.225 ms | 0 |

No calls exceeded 10 ms in the 12,000 measured incremental calls per version
and model. However, continuous-call maxima **did not improve**: DPDFNet-8
rose from 4.503 to 7.128 ms and DPDFNet-2 from 2.248 to 6.423 ms. All samples
remain included. The means and median run p99 improve in all three modes;
this is not a claim that every tail statistic improves.

Direct cumulative comparisons against the original pre-register-change build:

| Model | Mode | Original | Follow-up | Total reduction |
| --- | --- | ---: | ---: | ---: |
| DPDFNet-8 | Continuous paired | 2.705 ms | 2.217 ms | 18.0% |
| DPDFNet-8 | Paired 10 ms cadence | 2.803 ms | 2.320 ms | 17.2% |
| DPDFNet-2 | Continuous paired | 1.300 ms | 0.989 ms | 24.0% |
| DPDFNet-2 | Paired 10 ms cadence | 1.378 ms | 1.075 ms | 22.0% |

Cumulative medians use two runs per mode. No calls exceeded 10 ms in these
runs either, but isolated continuous maxima again favor the old baseline.
The separate wrapper diagnostic below investigates this limitation.

Raw results: [summary and hashes](../results/latency_followup_summary.json),
incremental [DPDFNet-8](../results/dpdfnet8_latency_followup.json) /
[DPDFNet-2](../results/dpdfnet2_latency_followup.json), cumulative
[DPDFNet-8](../results/dpdfnet8_followup_cumulative.json) /
[DPDFNet-2](../results/dpdfnet2_followup_cumulative.json).

## Retained changes

**Convolution accumulation stays in registers.** A private AVX2 kernel handles
single-output-row, stride-one convolutions whose input height equals kernel
height and which need no vertical padding. It accumulates up to 64 spatial
outputs through the complete ordered channel/kernel reduction before storing
them. This avoids repeatedly loading/storing output and calling a small axpy
function for every kernel tap. Dispatch remains behind the existing AVX2/FMA
feature check. Other shapes use the existing generic convolution.

Exact rounding at the edges matters: the original axpy fuses full groups of
eight, then uses separate multiply/add for its tail. Applying FMA everywhere
would silently change the model. The new interior is the intersection where
every tap originally used FMA. Edge vectors blend fused and separate results
per lane and skip padding contributions. Loads stay within each input row;
masked stores handle incomplete output tiles.

The first prototype used scalar edge processing. It passed parity but slightly
regressed whole-model latency, so it was replaced. A 21-shape microbenchmark
then showed that short padded depthwise reductions still lost performance.
Those retain the generic path when there is one input channel per group, a
one-row kernel wider than one, positive horizontal padding, and output width
below 64. These are arithmetic-shape conditions, with no model-ID dispatch.

**INT8 range reduction stays in registers.** Per-row quantization now reduces
the vector min/max directly with SIMD instead of storing eight lanes and
comparing them individually. Zero remains the lower/upper initialization;
unordered lane comparisons are treated like the previous scalar comparisons.
Scale, zero point, rounding, clamping, packing, and the dot-product epilogue
are unchanged. The independent scalar quantization oracle still passes.

**Two gate vectors are scheduled together.** A two-way unroll of the inner
AVX2 gate loop exposes independent work to the compiler. The existing exp,
sigmoid, tanh and state-update formulas are unchanged. No new approximation,
precision mode, worker thread or instruction set is introduced.

The generated graph, model weights and context memory sizes are unchanged
from the preceding pass. This follow-up needs a rebuild; it does not require
another graph-layout change. A fresh checkout should still regenerate models
with the current generator to include the preceding pass's arena/layout work.

## Evidence and validation

The refreshed diagnostic profile of the preceding optimized code found about
0.327 ms of convolution-family work and 0.369 ms of DPRNN gates in DPDFNet-8.
Explicit transposes were down to 0.026 ms and DPRNN normalization to 0.061 ms.
Instrumentation and startup are included, so these identify candidates rather
than establish release latencies. Profiles:
[DPDFNet-8](../results/dpdfnet8_followup_profile.json),
[DPDFNet-2](../results/dpdfnet2_followup_profile.json).

Screening evidence is retained, including the rejected first prototype:
[scalar-edge convolution](../results/followup_conv8_screen.json),
[vector edges plus quantization reduction](../results/followup_quant8_screen.json),
[gate unroll plus short-depthwise fallback](../results/followup_gate8_screen.json),
and [per-shape convolution measurements](../results/followup_conv_shapes.json).
The last file's shape indices follow the convolution contexts in the generated
DPDFNet-8 manifest. It compares the preceding pass with the second screening
candidate, before adding the short-depthwise fallback. These are screening
measurements, not additive estimates of isolated contributions.

The expanded layout contract checks 351 convolution shapes in both scalar and
AVX2 modes when available. Its output-by-output oracle reproduces the original
per-tap rounding rule independently of the new tiled implementation. Shapes
cover narrow and odd widths, 1/3/5-wide kernels, strides 1/2/3, grouped and
ungrouped operation, null biases, unaligned buffers, output canaries, and
multi-output-row fallback. Existing transpose and ordered-normalization tests
remain enabled.

Both models passed all **28 release, scalar-only and ASan/UBSan CTest tests**,
500-frame byte-exact output/state comparisons in FP32, selective FP16 and
selective INT8, and four-context serial/concurrent comparisons. Scalar-only
FP32 matched over 100 frames. Sanitizer executables use the previously
documented `-no-pie` workaround; libraries remain PIC and instrumented.

## Python-wrapper tail diagnostic

The largest continuous peaks warranted a separate investigation. The new
[`wrapper_latency_probe.py`](wrapper_latency_probe.py) records garbage
collection callbacks while measuring the existing allocating wrapper, then
calls the same native ABI with preallocated input pointers, output and
in-place recurrent-state buffers. GC remains enabled in both modes. Each
library's preallocated output/state first matches its ordinary wrapper over
128 recurrent frames. Timing then uses three continuous paired runs of 1,000
measured calls per mode/version/model, with 100 warmup calls per run.

The allocating diagnostic reproduced these peaks:

| Model/version affected | Whole call | Garbage collection within call |
| --- | ---: | ---: |
| DPDFNet-8 follow-up | 7.762 ms | 5.465 ms |
| DPDFNet-2 preceding baseline | 6.676 ms | 5.539 ms |

The collection lands on different implementations in the two runs. This is
direct evidence of wrapper-induced tail distortion, not evidence that the
optimized C kernel itself takes those extra milliseconds. Some other slow
calls had no collection, and the original uninstrumented benchmark cannot
retroactively attribute its individual peaks. None of its samples are removed.

The preallocated diagnostic's maxima were 4.197 -> 3.902 ms for DPDFNet-8 and
1.952 -> 1.622 ms for DPDFNet-2. Mean improvements were 3.3% and 5.7%,
respectively, using median run means. No collections occurred inside the
preallocated calls. These remain short shared-host measurements, not an audio
deadline guarantee or a replacement for the primary benchmarks.

Results: [DPDFNet-8](../results/dpdfnet8_wrapper_diagnostic.json),
[DPDFNet-2](../results/dpdfnet2_wrapper_diagnostic.json). Reproduce with:

```sh
python native/wrapper_latency_probe.py --model models/dpdfnet8_48khz_hr.onnx --weights models/rework8/weights.f32 --baseline-build build/rework_final8 --candidate-build build/followup_final8 --output results/dpdfnet8_wrapper_diagnostic.json
```

## Measurement scope

The final selective INT8 setting remains `fc_and_1x1_8`. Incremental comparisons
use four continuous paired runs, four paired 10 ms cadence runs, and four
standalone 10 ms cadence runs per implementation/model. Each run has 100
warmup and 1,000 timed recurrent frames. Paired calls alternate order each hop;
standalone run order is balanced. Cumulative comparisons use two continuous
and two paired cadence runs against the original reference. Their speedup is
measured directly, not obtained by multiplying percentages from different days.

Wall-time samples remain unfiltered. Reports include p50/p95/p99/max, calls
over 10 ms, calling-thread CPU time and scheduling observations. Paired cadence
runs two models per hop; standalone cadence runs one. Results are from the
i7-8700 under Linux Docker/WSL2 and GCC 12.2, with unrestricted affinity. They
exclude FFT, resampling and HushMic/PipeWire, and establish no worst-case or
cross-machine latency guarantee. AVX2/FMA requirements and scalar FP32 support
are unchanged; VNNI is not used.

## Reproduce

From `native_inference/` in the existing Linux development environment, with
the pinned ONNX models and dependencies installed:

```sh
for model in 8 2; do
  python native/generate_extended.py models/dpdfnet${model}_48khz_hr.onnx models/rework${model}
  cmake -S native -B build/followup_final${model} -DDPDF_EXTENDED_MODEL=ON -DDPDF_GENERATED_MODEL="$(pwd)/models/rework${model}/generated_model.c" -DDPDF_TEST_WEIGHTS="$(pwd)/models/rework${model}/weights.f32"
  cmake --build build/followup_final${model} -j 4
  ctest --test-dir build/followup_final${model} --output-on-failure
  python native/latency_validation.py --model models/dpdfnet${model}_48khz_hr.onnx --weights models/rework${model}/weights.f32 --baseline-build build/rework_final${model} --candidate-build build/followup_final${model} --output results/dpdfnet${model}_followup_validation.json
  python native/latency_probe.py --model models/dpdfnet${model}_48khz_hr.onnx --weights models/rework${model}/weights.f32 --baseline-build build/rework_final${model} --candidate-build build/followup_final${model} --repeats 4 --paced-repeats 4 --standalone-paced-repeats 4 --output results/dpdfnet${model}_latency_followup.json
done
```

The incremental commands assume the preceding pass's baseline binaries have
been preserved; do not rebuild them with the new source. The local pre-change
source snapshot is `scratch/latency_followup/preserved/` and can be compiled in
a fresh directory with `-DBUILD_TESTING=OFF` against the same generated graph.
Snapshots, model files and binaries are ignored by Git. To reproduce a
cumulative comparison from a clean checkout, build the original baseline from
commit `2dd237fe93e19ef162c986cdeea1086e219e9584` using the
[previous report's instructions](LATENCY_REWORK.md#reproduce), then pass that
build as `--baseline-build` with `--repeats 2 --paced-repeats 2`.

Use separate fresh directories with `-DDPDF_SANITIZE=ON` and
`-DCMAKE_EXE_LINKER_FLAGS=-no-pie`, or `-DDPDF_ENABLE_AVX2=OFF`, for the other
CTest configurations. Benchmarks must run sequentially after validation.
