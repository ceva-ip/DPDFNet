# Exact latency and memory optimization

**Continued:** [convolution, quantization and recurrent-gate optimization](LATENCY_FOLLOWUP.md)
uses this completed pass as its incremental baseline. Results below describe
the preserved September 20 build.

Implemented 2026-09-20. The previous small register-lifetime change did not
establish better real-time behavior: five DPDFNet-8 calls exceeded 10 ms in
its paced sample, versus none in the baseline. That observation must be kept,
not dismissed because the mean improved. It also does not identify the cause
of those particular delays; the old benchmark did not record CPU time or
scheduling diagnostics.

This pass targets more of the actual selective INT8 workload. It keeps one
inference thread, the same model weights, quantization, arithmetic order,
recurrent state, and public API. It requires neither handwritten assembly nor
VNNI. The performance reference is the preserved implementation at commit
`2dd237fe93e19ef162c986cdeea1086e219e9584`, before the small register change.

## Measurements

The new candidate improves both average and p99 latency on this host. At
standalone 10 ms cadence, median run mean falls **15.2% for DPDFNet-8** and
**21.5% for DPDFNet-2**. Against the same baseline, the continuous paired
comparison improves 14.4% and 17.3%, respectively.

| Model | Measurement mode | Baseline mean | Candidate mean | Reduction |
| --- | --- | ---: | ---: | ---: |
| DPDFNet-8 | Continuous paired | 2.680 ms | 2.294 ms | 14.4% |
| DPDFNet-8 | Paired 10 ms cadence | 2.784 ms | 2.412 ms | 13.4% |
| DPDFNet-8 | Standalone 10 ms cadence | 2.703 ms | 2.291 ms | 15.2% |
| DPDFNet-2 | Continuous paired | 1.259 ms | 1.041 ms | 17.3% |
| DPDFNet-2 | Paired 10 ms cadence | 1.385 ms | 1.143 ms | 17.5% |
| DPDFNet-2 | Standalone 10 ms cadence | 1.328 ms | 1.042 ms | 21.5% |

Standalone cadence tails, over 4,000 measured calls per implementation/model:

| Model | Implementation | p95 | p99 | Largest call | Calls >10 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| DPDFNet-8 | Baseline | 2.951 ms | 3.423 ms | 4.593 ms | 0 |
| DPDFNet-8 | Candidate | 2.577 ms | 2.936 ms | 4.482 ms | 0 |
| DPDFNet-2 | Baseline | 1.542 ms | 1.797 ms | 2.355 ms | 0 |
| DPDFNet-2 | Candidate | 1.228 ms | 1.430 ms | 2.066 ms | 0 |

Means and percentiles above are medians of the four run statistics, not pooled
percentiles; the largest call is the maximum across all four runs. All twelve
run pairs per model improve mean latency. Standalone DPDFNet-8 gains range
from 9.4% to 16.3%, and DPDFNet-2 from 18.5% to 23.9%; shared-host variation
still exists. There are zero calls over 10 ms in **all 12,000 timed calls per
implementation/model**, including the continuous and paired cadence modes.
This sample supports better mean/p99 behavior, not a worst-case guarantee or
proof that the original five delays have a known, fixed cause.

Results: [summary, hashes and validation](../results/latency_rework_summary.json),
[DPDFNet-8 runs](../results/dpdfnet8_latency_rework.json),
[DPDFNet-2 runs](../results/dpdfnet2_latency_rework.json).

The benchmark uses the existing `fc_and_1x1_8` configuration, 100 warmup and
1,000 measured recurrent hops per run. Four runs per implementation are made
in each of three modes:

- Continuous paired calls, reversing baseline/candidate order every hop.
- Paired 10 ms cadence with the same alternating order. This controls slow
  host drift but runs two implementations per hop, increasing duty cycle.
- Standalone 10 ms cadence: one implementation per stream, balancing the
  order of the separate baseline/candidate runs.

All measurements are retained, including every call over 10 ms. Wall time,
calling-thread CPU time, context-switch observations and release lateness are
recorded. CPU time includes a few microseconds of extra timer bookkeeping;
under WSL2 it cannot completely explain host-side interference. There is no
outlier filtering, real-time scheduling, affinity restriction, or extra
inference worker. A concurrent-context correctness test runs before timing.

The machine is an i7-8700 under Linux Docker/WSL2, using GCC 12.2 and the same
offline development image as the preceding experiment. Timings include the
Python native-call wrapper and output/state allocation. They exclude FFT,
resampling, PipeWire, and the HushMic application. This is not a bare-metal
Linux latency guarantee or an Intel/AMD fleet benchmark.

## What changed

1. **Generate simpler layout conversions.** Static nested loops and contiguous
   copies replace per-element divide/modulo indexing in graph transposes.
   Pointwise convolutions use a blocked scalar transpose or an exact AVX2
   8-by-8 transpose, selected during model creation. These operations move
   existing float bits without changing values.
2. **Reuse temporary graph storage.** The generator computes conservative
   allocation lifetimes, including all view references. Two values live in
   the same node never share storage. Slots are coalesced/reused after the
   last reference, and a separate overlap check verifies the plan. The
   manifest records original offsets, reused offsets and live ranges;
   unused allocations have a null reused offset. This changes the existing
   static arena allocation, adding no allocation to processing.
3. **Vectorize normalization across independent rows.** Four rows occupy four
   double-precision lanes. Each lane sums all 64 channels in the original
   order and retains the original double variance, square root/division and
   FP32 residual operations. No approximate reciprocal, reassociation or new
   FMA is introduced. This avoids the quality tradeoff that a less careful
   FP32 reduction or activation approximation would require.
4. **Retile batched INT8 work.** A four-row/two-tile integer helper reuses
   activation preparation and keeps reduction temporaries separate from the
   floating-point epilogue. It serves ordinary batched projections and the
   two-direction input projection. Unsupported tile shapes retain the
   existing path. Packing, quantization, correction and dequantization are
   unchanged. The preceding one-row helper is retained.

Graph arena storage drops from **4,901,768 to 977,344 bytes** for DPDFNet-8
(80.1%), and **4,361,096 to 879,040 bytes** for DPDFNet-2 (79.8%). These are
temporary arena sizes, not process RSS or total model size. Total allocations
owned by the INT8 model context fall from 10,663,239 to 6,738,951 bytes and
8,888,691 to 5,406,675 bytes, respectively; caller-owned state is separate.

## Why model size alone did not predict the gain

A larger model offers more absolute work to remove, but an optimization only
helps the operations it actually reaches. The earlier patch changed one-row
INT8 reductions; it did not accelerate all of DPDFNet-8's extra work. Its
6.1%/0.9% split should not be treated as an intrinsic model property from
those noisy, separate runs.

The old patch was rechecked using the preserved old binaries, with two
continuous and two paced paired runs per model (1,000 timed hops/run):

| Old patch only | Continuous mean reduction | Paired paced mean reduction |
| --- | ---: | ---: |
| DPDFNet-8 | 1.9% | 2.5% |
| DPDFNet-2 | 0.9% | -1.0% (slower) |

Thus the earlier 6.1%/0.9% split **does not reproduce** under this comparison.
The two measurements use different pacing duty cycles, and neither isolates
the cause of the old outliers. The defensible conclusion is that the old
patch's small end-to-end effect was sensitive to measurement conditions. It
was not evidence of a model-size scaling limit. Raw rechecks:
[DPDFNet-8](../results/dpdfnet8_registers_recheck.json),
[DPDFNet-2](../results/dpdfnet2_registers_recheck.json).

A fresh diagnostic profile of the selective INT8 path, including the previous
small patch, found the following mean times. Instrumentation and startup are
included, so these are a cost map, not release latencies or precise Amdahl
predictions:

| Work | DPDFNet-8 | DPDFNet-2 |
| --- | ---: | ---: |
| All DPRNN blocks | 1.949 ms | 0.485 ms |
| Normalization within those blocks | 0.207 ms | 0.051 ms |
| Convolution nodes, including their layouts | 0.356 ms | 0.351 ms |
| Explicit graph transposes | 0.144 ms | 0.143 ms |

The larger model has four times the DPRNN blocks but nearly the same shared
convolution/layout work. Removing that shared work saves a larger percentage
of the smaller model's total. Accelerating normalization and batched work
also reaches the larger model's repeated blocks. That combination is more
productive than pursuing only the previous register-spill candidate.
In the new standalone comparison, DPDFNet-8 actually saves more absolute time:
0.412 ms per hop versus 0.286 ms for DPDFNet-2. The smaller denominator still
gives DPDFNet-2 a larger percentage improvement.

Diagnostic data: [DPDFNet-8](../results/dpdfnet8_int8_profile.json),
[DPDFNet-2](../results/dpdfnet2_int8_profile.json). Screening stages are also
retained: [layouts and arena](../results/rework_graph8_screen.json),
[plus normalization](../results/rework_norm8_screen.json),
[plus batched kernels](../results/rework_batch8_screen.json). These sequential
screening runs are not isolated causal estimates of each component's gain.

## Correctness and portability

- Both models pass five release, five ASan/UBSan and four scalar-only CTest
  contracts: **28/28 tests**. Sanitizer executables use `-no-pie` for the
  previously [diagnosed startup instability](AVX2_OPTIMIZATION.md#correctness-and-sanitizer-validation);
  the libraries remain PIC and sanitizer instrumentation stays enabled.
- Each model matches the preserved library's output and complete recurrent
  state byte-for-byte over **500 frames in each of FP32, selective FP16 and
  selective INT8**. Scalar-only FP32 matches over 100 frames as well.
- Four independent contexts, each with different inputs, match serial results
  during concurrent execution over 64 frames in all those modes. A single
  context still belongs to one stream; this adds no intra-model threading.
- The generator contract compiles and checks 61 transpose permutations and a
  case with overlapping view lifetimes. The layout contract checks 144
  dimension pairs, edge tiles, unaligned buffers, canaries and NaN payloads.
  Ordered-double normalization matches its independent scalar reference.
  Expanded INT8 contracts cover batched/paired paths and tile fallbacks.

The portable graph and arena improvements also apply to the scalar build.
AVX2/FMA kernels retain the existing runtime dispatch and are compiled in
separate translation units; there is no global `-march=native`. Scalar FP32
remains available without AVX2. As before, explicit INT8 requires AVX2/FMA,
and FP16 weights require F16C as well. No new CPU requirement has been added.
Performance on other CPUs still needs measurement.

## Reproduce

Regenerate the model source: rebuilding an old generated graph does **not**
include the new layouts or arena plan. The pinned ONNX models and the existing
development dependencies must already be present, as described in the parent
README. Run the following inside Linux from `native_inference/`, sequentially
on an otherwise idle machine. Use fresh build/scratch directories if the
listed names already exist.

Prepare the historical source in a fresh directory from the repository root:

```sh
mkdir -p native_inference/scratch/latency_reproduce
git archive 2dd237fe93e19ef162c986cdeea1086e219e9584 native_inference/native | tar -x -C native_inference/scratch/latency_reproduce
cd native_inference
```

Then build the old and new generated graphs for each model:

```sh
baseline=scratch/latency_reproduce/native_inference/native
python native/export_blocks.py models/dpdfnet8_48khz_hr.onnx models/native_blocks
python native/generator_contract.py
for model in 8 2; do
  python "$baseline/generate_extended.py" models/dpdfnet${model}_48khz_hr.onnx models/latency_baseline${model}
  python native/generate_extended.py models/dpdfnet${model}_48khz_hr.onnx models/rework${model}
  cmake -S "$baseline" -B build/latency_baseline${model} -DBUILD_TESTING=OFF -DDPDF_GENERATED_MODEL="$(pwd)/models/latency_baseline${model}/generated_model.c"
  cmake --build build/latency_baseline${model} -j 4
  cmake -S native -B build/rework_final${model} -DDPDF_EXTENDED_MODEL=ON -DDPDF_GENERATED_MODEL="$(pwd)/models/rework${model}/generated_model.c" -DDPDF_TEST_WEIGHTS="$(pwd)/models/rework${model}/weights.f32"
  cmake --build build/rework_final${model} -j 4
  ctest --test-dir build/rework_final${model} --output-on-failure
  python native/latency_validation.py --model models/dpdfnet${model}_48khz_hr.onnx --weights models/rework${model}/weights.f32 --baseline-build build/latency_baseline${model} --candidate-build build/rework_final${model} --output results/dpdfnet${model}_latency_validation.json
  python native/latency_probe.py --model models/dpdfnet${model}_48khz_hr.onnx --weights models/rework${model}/weights.f32 --baseline-build build/latency_baseline${model} --candidate-build build/rework_final${model} --repeats 4 --paced-repeats 4 --standalone-paced-repeats 4 --output results/dpdfnet${model}_latency_rework.json
done
```

In separate fresh build directories, repeat CMake/build/CTest with
`-DDPDF_SANITIZE=ON -DCMAKE_EXE_LINKER_FLAGS=-no-pie` for ASan/UBSan, and with
`-DDPDF_ENABLE_AVX2=OFF` for scalar-only validation. Comparing two scalar-only
libraries uses `latency_validation.py --configs compact_fp32 --frames 100`.
The development image has glibc 2.36; these binaries are not HushMic release
artifacts and do not replace its documented glibc build floor.
