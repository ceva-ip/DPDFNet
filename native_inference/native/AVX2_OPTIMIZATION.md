# AVX2 INT8 register-lifetime optimization

**Superseded by the [latency and memory optimization](LATENCY_REWORK.md).**
The measurements below are retained as historical results. In particular, the
6.1% versus 0.9% paced gains and five DPDFNet-8 calls above 10 ms did not
establish a robust real-time improvement. The follow-up rechecks this small
patch with per-hop balanced comparisons and measures a broader optimization.
The code and reproduction commands below describe the earlier source state.

Implemented 2026-09-20. The selected change is C with AVX2 intrinsics, with
the same CPU requirements as the existing INT8 implementation. No VNNI,
AVX-512, new weight format, quantization change, or extra thread is involved.

## Implementation and assembly comparison

`int8.c` now isolates the 64-output integer reduction in a deliberately
non-inlined `qdot64()` helper. Weight temporaries are consumed immediately.
This separates the accumulator lifetimes from the quantization and floating
point epilogue. GCC 12.2 keeps all eight accumulators in registers throughout
the reduction, eliminating the two accumulator spills found in the previous
compiler audit. Stores of completed sums occur once after the reduction.

The smaller output tail and batch kernels are unchanged. The helper supports
the existing packed layout and supported K range, rather than naming a model.
The epilogue retains the original zero-point correction, scale multiplication
and FMA. No public API, dispatch policy, model weights or recurrent state
representation changes.

A handwritten SysV AVX2 assembly helper was also built and tested, using eight
fixed accumulator registers and two alternating weight temporaries. It passed
the scalar integer oracle and 500-frame byte-exact recurrent comparison. In
the initial three-way whole-model comparison, assembly did not establish an
advantage over the revised intrinsics, so it was not added to the runtime.
The experimental source remains in the local ignored
`scratch/avx2_experiment/dot64.S`.

The initial DPDFNet-8 comparison used three continuous runs and one paced run:

| Implementation | Continuous median run mean | Paced run mean |
| --- | ---: | ---: |
| Previous kernel | 2.812 ms | 2.974 ms |
| Revised intrinsics | 2.713 ms | 2.791 ms |
| Assembly prototype | 2.716 ms | 2.837 ms |

These are screening measurements, not the final repeated comparison below.
Raw results: [candidate screening](../results/avx2_kernel_candidates.json).

The isolated affine benchmark includes quantization and the epilogue. Its
median nanoseconds per call across seven runs were:

| K x N, M=1 | Previous | Intrinsics | Assembly prototype |
| --- | ---: | ---: | ---: |
| 64 x 192 | 256.5 | 246.3 | 234.7 |
| 128 x 64 | 189.5 | 171.4 | 170.7 |
| 256 x 768 | 3552.7 | 3338.9 | 3263.6 |
| 512 x 768 | 7061.9 | 7609.7 | 6951.7 |

The last, larger stress shape regressed in this intrinsic microbenchmark;
the ten dense GRU contexts in the audited models use K=256, N=768. This is
not a claim of faster performance at every supported shape. The screening
assembly's microbenchmark advantage also did not translate into better
whole-model screening times. Raw results:
[affine comparison](../results/avx2_affine_candidates.json), backend indices
0=previous, 1=intrinsics, 2=assembly. The benchmark driver is
[`affine_benchmark.c`](affine_benchmark.c).

## Final comparison

Five continuous runs and three 10 ms paced runs per implementation, using the
final selective INT8 configuration (`fc_and_1x1_8`). Each value below is the
median of the run means, in milliseconds per hop:

| Model | Mode | Previous | Optimized | Reduction |
| --- | --- | ---: | ---: | ---: |
| DPDFNet-8 | Continuous | 2.765 | 2.742 | 0.8% |
| DPDFNet-8 | Paced | 3.017 | 2.990 | 0.9% |
| DPDFNet-2 | Continuous | 1.311 | 1.262 | 3.8% |
| DPDFNet-2 | Paced | 1.632 | 1.532 | 6.1% |

Median frame p50 improved by 1.6% / 2.5% for DPDFNet-8 (continuous / paced)
and 2.8% / 3.8% for DPDFNet-2. Both models matched the previous INT8 output
and complete recurrent state byte-for-byte over 500 frames before timing.

**Tail latency did not consistently improve.** The DPDFNet-8 paced candidate
recorded five calls over 10 ms across 3,000 timed frames, versus zero in its
baseline. Its largest call was 12.786 ms versus the baseline's 9.858 ms.
There were no such exceedances in either continuous comparison or in the
DPDFNet-2 paced comparison. These shared-host measurements support a modest
average improvement, not a claim of better worst-case latency or fewer audio
dropouts. The smaller DPDFNet-8 gain in the final run also shows why the
initial screening result must not be treated as the final speedup.

Results: [summary and validation](../results/avx2_registers_summary.json),
[DPDFNet-8 raw runs](../results/dpdfnet8_avx2_registers.json), and
[DPDFNet-2 raw runs](../results/dpdfnet2_avx2_registers.json).

## Compatibility and limits

AVX2/FMA-capable Intel and AMD CPUs can use the existing INT8 path with this
change; no newer ISA is selected. The FP32 scalar fallback remains available
on other supported platforms. Explicit INT8 requests still require AVX2/FMA.
The optimization benefits the one-row INT8 path, not FP32/FP16 mode.

Measurements are from one i7-8700 under Linux Docker/WSL2, GCC 12.2. They are
not bare-metal HushMic/PipeWire timings, and do not establish an AMD or
cross-machine speedup. SIMD source portability is preserved, but compiler
register allocation and CPU behavior still require validation on other hosts.

## Correctness and sanitizer validation

Both models pass four release contracts, four ASan/UBSan contracts and three
scalar-only contracts (22 final-build tests total). The expanded independent
scalar integer oracle directly tests M=1 at six K values and seven output
widths, including short widths, tile tails, zero/extreme/mixed inputs, zero
weight columns and deliberately unaligned float buffers. Output canaries and
sanitizer instrumentation check bounds. Existing batch, in-place, reset and
finite-state tests remain enabled.

The initial sanitizer run had early SIGSEGV failures in two PIE test
executables, without a memory-error report. Both passed when invoked directly.
An empty `int main(void) { return 0; }` program built with ASan/UBSan also
timed out in 2/10 PIE starts, versus 0/10 non-PIE starts. The final sanitizer
executables were linked with `-DCMAKE_EXE_LINKER_FLAGS=-no-pie`; instrumentation
remained enabled and all tests passed. This addresses the observed environment
instability without changing the release libraries. Diagnostic observations:
[empty-program starts](../results/avx2_asan_startup.json).

To reproduce sanitizer validation, use the same candidate CMake command in a
fresh build directory with `-DDPDF_SANITIZE=ON` and
`-DCMAKE_EXE_LINKER_FLAGS=-no-pie`, then build and run CTest. For scalar-only
validation, use another fresh directory with `-DDPDF_ENABLE_AVX2=OFF`.

## Reproduction

The pre-change INT8 source is at repository commit
`2dd237fe93e19ef162c986cdeea1086e219e9584`. To prepare a baseline source copy
on Linux from `native_inference/` (do this once, in a fresh scratch directory):

```sh
mkdir -p scratch/avx2_reproduce
cp -R native scratch/avx2_reproduce/baseline
git show 2dd237fe93e19ef162c986cdeea1086e219e9584:native_inference/native/int8.c > scratch/avx2_reproduce/baseline/int8.c
```

For DPDFNet-8, use `weights=extended` and `model=8`; for DPDFNet-2, use
`weights=dpdfnet2_extended` and `model=2`. The models must already be generated
as described in the parent README. Run these Linux commands in an environment
with the documented compiler/Python dependencies:

```sh
weights=extended
model=8
cmake -S scratch/avx2_reproduce/baseline -B build/avx2_baseline_$model -DBUILD_TESTING=OFF -DDPDF_GENERATED_MODEL="$(pwd)/models/$weights/generated_model.c"
cmake --build build/avx2_baseline_$model -j 4
cmake -S native -B build/avx2_final_$model -DDPDF_EXTENDED_MODEL=ON -DDPDF_GENERATED_MODEL="$(pwd)/models/$weights/generated_model.c" -DDPDF_TEST_WEIGHTS="$(pwd)/models/$weights/weights.f32"
cmake --build build/avx2_final_$model -j 4
ctest --test-dir build/avx2_final_$model --output-on-failure
python native/optimization_probe.py --model models/dpdfnet${model}_48khz_hr.onnx --weights models/$weights/weights.f32 --baseline-build build/avx2_baseline_$model --candidate-build build/avx2_final_$model --output results/dpdfnet${model}_avx2_registers.json
```

Use fresh build directories for reproduction if the named directories already
contain a build configured from another source path. The benchmark rotates
baseline/candidate order, uses five continuous and three 10 ms paced runs,
100 warmup plus 1,000 timed frames per run, and checks byte-exact output and
complete recurrent state over 500 frames before timing. The JSON records
model, weights and shared-library hashes. Run benchmarks sequentially on an
otherwise idle machine.

For isolated kernel timing:

```sh
cc -O2 -Wall -Wextra -Werror native/affine_benchmark.c -ldl -o build/affine_benchmark
build/affine_benchmark "$(pwd)/build/avx2_baseline_8/libdpdf_dprnn.so" "$(pwd)/build/avx2_final_8/libdpdf_dprnn.so"
```
