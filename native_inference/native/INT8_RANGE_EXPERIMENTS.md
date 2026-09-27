# Further DPDFNet-8 optimization experiments

Investigated 2026-09-26 against repository commit
`6a5dbd3ea5dbea88c30a2be8e7c689b5eb26b533`, after the completed convolution,
quantization and gate follow-up. All comparisons use the same current generated
graph and weights, with a freshly compiled preserved baseline.

This pass implements isolated research builds, including a handwritten AVX2
kernel. The production kernels, integration ABI, generated artifacts and named
W8A8 preset are unchanged. Research libraries use the development ABI and must
not be installed as replacements for `DPDF_PRESET_INT8_SELECTIVE`.

## Measured result

**W7A8 is the strongest candidate: 11.4% less time at paired 10 ms cadence,
with small mixed quality changes on the seven-fixture suite.** The assembly
kernel is exact but saves only 0.8% in that cadence measurement.

Values are median run means, in milliseconds per hop. Each candidate has its
own matched baseline runs; differences between baseline columns are host drift.

| Measurement | Baseline → W7A8 | Reduction | Baseline → assembly | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Continuous paired | 2.305 → 2.035 | 11.7% | 2.327 → 2.295 | 1.4% |
| Paired 10 ms cadence | 2.422 → 2.145 | 11.4% | 2.429 → 2.408 | 0.8% |
| Standalone 10 ms cadence | 2.407 → 2.140 | 11.1% | 2.457 → 2.418 | 1.6% |
| Preallocated continuous diagnostic | 2.225 → 1.949 | 12.4% | 2.276 → 2.257 | 0.8% |

Every W7A8 run pair improved its mean. Assembly had one slight mean regression
in four paired cadence runs. W8A7 saved 13.7% in continuous screening, but its
quality deltas were less favorable; it did not advance to the final cadence
comparison. Reported owned context storage is **6,738,951 bytes** in all builds.
This is not a new process-RSS measurement.

**Tail results are mixed.** W7A8 paired-cadence median run p99 improved from
2.620 to 2.364 ms, but standalone median run p99 rose from 2.665 to 2.836 ms,
and standalone maximum rose from 3.630 to 3.925 ms. Assembly's continuous
maximum rose from 4.427 to 8.022 ms. None of the final primary benchmark calls
exceeded 10 ms (10,000 timed calls per implementation in each comparison).

The separate wrapper diagnostic recorded an assembly call of 12.237 ms,
including 8.705 ms of Python GC, and a baseline W7A8-comparison call of
8.390 ms, including 6.017 ms of GC. The corresponding preallocated runs had
no calls above 10 ms. The assembly screening also contained one 11.738 ms
call; that uninstrumented peak cannot be retrospectively attributed to GC.
These samples remain in their reports; no worst-case latency improvement is
claimed.

Raw evidence: [summary](../results/range_experiments_summary.json),
[W7A8 final timing](../results/range_w78_final.json),
[assembly final timing](../results/range_asm48_final.json),
[W7A8 wrapper diagnostic](../results/range_w78_wrapper.json),
[assembly wrapper diagnostic](../results/range_asm48_wrapper.json), and
[speech quality](../results/range_precision8_quality.json).

## What was tested

| Variant | Change | Numerical contract |
| --- | --- | --- |
| `inline` | Inline the four-row integer dot into scaling/bias code | Exact W8A8 |
| `fixed64` | Specialize the one-row dot for K=64 | Exact W8A8 |
| `unroll` | Unroll the four-row dot by two K steps | Exact W8A8 |
| `asm4` | Explicit register scheduling for four rows and two output vectors | Exact W8A8 |
| `u7` | Unsigned activations 0..127, weights -127..127 | Changed W8A7 grid |
| `w7` | Unsigned activations 0..254, weights -63..63 | Changed W7A8 grid |

The existing `qdot4pair` disassembly has no accumulator spills. Inlining and
K=64 specialization were slightly slower in whole-model screening; unrolling
alone improved about 0.5%. The explicit assembly schedule showed a larger,
but still modest, improvement. These candidates were measured independently;
their percentages must not be added together.

## Quality tradeoff

Changes relative to the current selective W8A8 implementation:

| Metric | W8A7 mean delta | W7A8 mean delta | W7A8 worst delta |
| --- | ---: | ---: | ---: |
| Clean-reference PESQ, four mixtures | -0.00264 | +0.01573 | +0.00122 |
| Clean-reference STOI, four mixtures | -0.00016 | +0.00165 | +0.00034 |
| Clean-reference ESTOI, four mixtures | +0.00016 | +0.00078 | -0.00044 |
| Clean-reference SNR, four mixtures | +0.00092 dB | -0.03439 dB | -0.18185 dB |
| DNSMOS P.835 overall, seven fixtures | -0.00510 | +0.00296 | -0.01677 |
| DNSMOS P.808, seven fixtures | -0.00182 | +0.00438 | -0.01749 |

W7A8 is the more promising quality/speed tradeoff in this small suite. Its
PESQ and STOI deltas were positive on every controlled mixture, while ESTOI,
SNR and DNSMOS still show individual regressions. W8A7's worst P.835 overall
delta was -0.03834. These small, mixed shifts do not prove general quality
equivalence or superiority. No human listening verdict is claimed.

## Why reducing the range helps

The existing AVX2 W8A8 loop stores centered signed activations. Each group of
four products needs an activation absolute value and a weight sign transfer
before `vpmaddubsw`, then `vpmaddwd` and integer accumulation. Merely casting
the activations to unsigned bytes changes the result. Re-centering them into
0..254 without another change can saturate the intermediate signed word sum.

Both experimental grids permit direct unsigned-activation/signed-weight
products with provable pair-sum bounds:

- W8A7: `2 * 127 * 127 = 32258`.
- W7A8: `2 * 254 * 63 = 32004`.

Both fit signed 16-bit words. Each kernel removes the sign transfer and
activation absolute-value work, then applies `dot(u,w) - zero_point*sum(w)`.
The existing per-row/per-output scales and FP32 bias FMA follow. Weights and
activations still occupy one byte each; **this saves computation, not RAM**.
FP32 gates, normalization and recurrent state are retained.

W7A8 retains the original 254-step activation grid and halves weight
resolution. W8A7 retains weight resolution and halves activation resolution.
Neither is byte-exact with the current W8A8 model. Neither needs retraining to
run, but either needs broader quality evidence before becoming a release preset.

## Handwritten assembly

[`qdot4pair_avx2.S`](experiments/qdot4pair_avx2.S) implements the same packed
four-row/two-vector integer dot as the existing intrinsics. Eight YMM registers
hold accumulators; the remaining working registers hold weights, activation
sign/magnitude, word ones and two independent product chains. It uses only
caller-saved SysV registers, unaligned loads/stores, AVX2 instructions and
`vzeroupper`. It introduces no new ISA requirement.

This file is Linux x86-64 SysV research code. It is not added to either default
CMake target. It does not cover Windows x64 or ARM calling conventions, and
the measured gain does not establish an AMD or cross-compiler benefit.

## Evidence and scope

Screening uses three alternating-order continuous runs, 100 warmup and 600
timed recurrent frames per run. Final runs use four continuous pairs, four
paired 10 ms cadence runs, and two standalone cadence runs per implementation,
with 100 warmup and 1,000 timed frames. A separate preallocated-buffer diagnostic
uses three continuous pairs. All samples, including scheduling and GC peaks,
remain included. Paired cadence runs two models per hop; standalone runs one.

Measurements are single-thread i7-8700, GCC 12.2, Linux Docker/WSL2, unrestricted
CPU affinity. They time the spectral model plus the stated wrapper; FFT,
resampling, HushMic and PipeWire are excluded. The 50 ms audio delay is unchanged.
They establish neither real-time deadlines nor cross-machine speedups.

The quality suite compares freshly computed original ONNX, current W8A8,
W8A7 and W7A8 outputs. It contains three existing HushMic fixtures and four
controlled fan/typing mixtures. Clean-reference alignment is estimated once
from original ONNX and applied to every candidate. PESQ/STOI/ESTOI/SNR and
DNSMOS are complementary measurements, not listening-test certification or
a diverse-speaker non-inferiority study.

## Validation

- W8A7 and W7A8 each passed five release and five ASan/UBSan CTest contracts.
  The scalar quantization/dot oracle was adapted to each grid independently
  of SIMD packing. It covers extreme/zero inputs, zero columns, unaligned
  buffers, short widths and tile tails. Batch and paired projections match
  independently checked single rows.
- Both reproducible candidates matched their screening implementations over
  500 recurrent frames in FP32, selective FP16 and their own experimental INT8
  mode. This is **not** parity with baseline W8A8. Four distinct concurrent
  contexts match serial execution in each mode.
- Assembly passed all five release contracts and 500-frame byte-exact
  output/state comparisons against the production arithmetic in all three
  modes, plus four-context concurrency checks.
- Sanitizer executables use the previously documented `-no-pie` environment
  workaround. Handwritten assembly itself is not ASan-instrumented.

Evidence: [contract counts and library hashes](../results/range_contracts.json),
[assembly parity/concurrency](../results/range_asm48_validation.json),
[W7A8 reproducibility/concurrency](../results/range_w78_validation.json), and
[W8A7 reproducibility/concurrency](../results/range_u78_validation.json).
The final quality run uses the same W7A8 library hash as final latency.
Decoded input and output WAV samples also match the screening pass exactly.
The final quality report includes PCM hashes: floating-point WAV container
metadata can change file hashes between otherwise identical runs.

## Reproduce

Use the existing Linux development/quality images and pinned models described
in the parent README. Run from `native_inference/`. The preparation script
checks the normalized `int8.c` source hash and refuses to overwrite an existing
destination or generate inside the source tree.

```sh
python native/experiments/prepare_int8_variant.py baseline scratch/ranges/baseline
python native/experiments/prepare_int8_variant.py asm4 scratch/ranges/asm4
python native/experiments/prepare_int8_variant.py u7 scratch/ranges/u7
python native/experiments/prepare_int8_variant.py w7 scratch/ranges/w7
for variant in baseline asm4 u7 w7; do
  cmake -S scratch/ranges/$variant -B build/ranges_$variant \
    -DDPDF_EXTENDED_MODEL=ON \
    -DDPDF_GENERATED_MODEL="$PWD/models/rework8/generated_model.c" \
    -DDPDF_TEST_WEIGHTS="$PWD/models/rework8/weights.f32"
  cmake --build build/ranges_$variant -j 4
  ctest --test-dir build/ranges_$variant --output-on-failure
done
python native/latency_probe.py \
  --model models/dpdfnet8_48khz_hr.onnx --weights models/rework8/weights.f32 \
  --baseline-build build/ranges_baseline --candidate-build build/ranges_asm4 \
  --repeats 4 --paced-repeats 4 --standalone-paced-repeats 2 \
  --output results/local_asm4.json
python native/experiments/range_latency_probe.py \
  --model models/dpdfnet8_48khz_hr.onnx --weights models/rework8/weights.f32 \
  --baseline-build build/ranges_baseline --candidate-build build/ranges_w7 \
  --repeats 4 --paced-repeats 4 --standalone-paced-repeats 2 \
  --output results/local_w7.json
python native/experiments/range_quality_probe.py \
  --model models/dpdfnet8_48khz_hr.onnx --weights models/rework8/weights.f32 \
  --baseline-build build/ranges_baseline --u7-build build/ranges_u7 \
  --w7-build build/ranges_w7 --scratch scratch/ranges/audio \
  --dnsmos-dir models/dnsmos --output results/local_ranges_quality.json
```

The quality command needs the quality image, the existing local speech fixtures
and voice source, and DNSMOS files. No network access is needed. Add
`-DDPDF_SANITIZE=ON -DCMAKE_EXE_LINKER_FLAGS=-no-pie` in fresh build directories
for reduced-range sanitizer validation. `latency_validation.py` can compare
an exact candidate to baseline, or repeat a candidate against itself to test
concurrent contexts. Use `wrapper_latency_probe.py` for the allocating versus
preallocated diagnostic. Preserve baseline binaries; never rebuild them from
candidate source.

## Next decisions

1. Prioritize W7A8 for a larger, more varied speech/noise evaluation and blinded
   listening. If accepted, give it a distinct preset and reference outputs;
   the existing W8A8 preset must keep its arithmetic contract.
2. Keep the assembly candidate as an exact scheduling experiment until its
   cadence gain justifies the additional platform-specific maintenance.
3. Measure a VNNI tier on hardware that exposes it. This i7-8700 cannot validate
   that direction; see the earlier [ISA discussion](ASSEMBLY_FEASIBILITY.md).
4. A two-worker encoder remains a separate wall-latency experiment. The two
   branches are independent mathematically, but the current arena planner reuses
   storage according to sequential lifetimes. Parallel execution must first
   extend those lifetimes or allocate separate branch storage. It also occupies
   another core, so it cannot be counted as single-thread efficiency.

INT4, pruning, low-rank factorization and retraining remain architectural
research, with additional quality and deployment work. This pass establishes
measured alternatives on the current host without assuming those will win.
