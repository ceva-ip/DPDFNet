# Single-thread exact-output optimization of DPDFNet-8 and DPDFNet-2

Investigation: 2026-10-03. Models: `dpdfnet8_48khz_hr` and `dpdfnet2_48khz_hr`.
The first tables below describe DPDFNet-8; the DPDFNet-2 results follow. The retained profile is
`best_single`, using exactly one inference thread. Its reference is the fitted
W7A8 + pack32 + degree-5 GRU-gate profile already present in the repository.
Both retain the same weights, quantization grid, activation coefficients, FP32
recurrent state, 48 kHz sample rate, 480-sample hop and 50 ms model delay.

The accepted optimization reduces standalone 10 ms cadence inference time by
**5.9%, from 1.983 to 1.867 ms per hop**, with byte-identical tested output.
It is an isolated, reproducible research build; distributed integration
presets remain unchanged. All inference executes on the calling thread.

## Matched latency results

| Execution | Reference → optimized mean | Time reduction |
| --- | ---: | ---: |
| Continuous | 1.903 → 1.805 ms | 5.17% |
| Paired 10 ms cadence | 2.011 → 1.918 ms | 4.62% |
| Standalone 10 ms cadence | **1.983 → 1.867 ms** | **5.88%** |

Standalone tails and CPU use:

| Measurement | Reference → optimized |
| --- | ---: |
| p99 | 2.347 → 2.239 ms |
| Observed maximum | 3.318 → 2.659 ms |
| Whole-process CPU per hop | 1.991 → 1.875 ms |

Whole-process CPU fell by about 5.85%. Both implementations had **zero calls
above 10 ms across 12,000 timed calls each**. All cadence calls completed
before the next scheduled release, including wake lateness. Continuous and
paired-cadence maxima were slightly higher for the candidate despite lower
means/p99, so these observations establish no worst-case timing guarantee.

Each implementation has four 1,000-hop continuous runs, four paired 10 ms
cadence runs and four standalone cadence runs, with 100 warmup hops per run.
The benchmark uses preallocated caller-owned buffers, cached native pointers
and in-place state. Paired order reverses each hop; standalone order alternates.
All measured samples and scheduler stalls are included in the statistics.
Table means and p99 are medians of four run statistics; maxima are the largest
observation across those runs.

Measurements use an Intel i7-8700, GCC 12.2 and Linux Docker/WSL2 with
unrestricted affinity. They exclude FFT, overlap-add, resampling and the
HushMic audio-device pipeline. Native Windows and HushMic under desktop/audio
load require their own measurements.

## Exact-output evidence

The full saved suite passed:

- 50 EARS-WHAM mixtures from six speakers.
- Ten quiet mixtures with clean speech at -50 dBFS RMS.
- Two clean-speech controls.
- Three independent 120-second white, pink and mechanical noise streams.

Across **65 files, 1,339.149 seconds and 134,332 streaming hops**, every output
spectrum, the entire 90,228-float recurrent state at every hop, and aligned
FP32 PCM matched the live W7A8 reference byte for byte. All values remained
finite. Input hashes and the complete fixture-ID set match the saved evaluation
record. All 12 previously saved fitted-profile waveform hashes matched too.
The fresh reference library is byte-identical to the preserved fitted W7A8
library.

Identical PCM preserves the fitted profile's quality on these fixtures,
including its existing limitations. It does not establish a quality improvement
or a universal input proof. No additional approximation is introduced.

Additional validation passed:

- 1,000 recurrent frames with exact output/state in the existing precision
  compatibility modes, plus four independent contexts matching serial and
  concurrent replay.
- A separate 1,000-hop synthetic stream with quiet/high-level/silence
  transitions, reset replay and independent concurrent contexts.
- Release and ASan/UBSan: all five C contracts each.
- Scalar-only: all four applicable C contracts.
- All four rounding modes with denormal handling off/on after model creation:
  64 recurrent frames per mode, exact spectrum/state, both buffers in place.
- One OS thread during creation, all FP-control inference calls and destruction.

The independent scalar quantization oracle retains its established W7A8 grid
adaptation. It checks short widths, tile tails, unaligned buffers, extreme and
zero inputs, row/batched execution and paired matrices. Concurrent replay
checks use independent contexts; each model still executes on one caller.

## What changed

The retained profile combines five exact changes:

1. **Clamp after byte packing.** Saturating int32-to-int16 and int16-to-byte
   packs already clamp values into [0,255]. One unsigned byte min with 254
   replaces the earlier int32 min/max operations. Scales, zero points, rounding,
   packed order and tails are unchanged.
2. **Contiguous normalization loads.** Load four adjacent channels from each
   row, transpose a 4x4 tile, and retain the original ordered FP64 mean and
   variance reductions. No floating-point reduction is reassociated.
3. **Skip the initial zero-state integer projection.** Each intra-GRU direction
   starts with positive zero. Its first integer projection equals `+0 + bias`;
   that addition is retained to preserve signed-zero behavior.
4. **Align packed weights to 64 bytes.** Packed vectors no longer straddle
   cache-line boundaries. Keep the allocation owner for destruction and count
   the padding in owned-memory accounting.
5. **Remove tanh's redundant upper exponential clamp.** Its exponent input
   is `-2*abs(x)`. The lower clamp, range reduction, fitted coefficients,
   reconstruction and division are unchanged.

Only the reference and accepted profile are exposed by the reproduction
script. No new CPU ISA, global `-march=native`, fast-math, hop skipping, extra
buffering or retraining is introduced.

## Memory

Four fresh processes per implementation warmed 120 hops, unmapped the source
weights after creation and subtracted the common imported-runtime baseline.
Owned heap allocations and incremental process RSS are separate measurements.

| Profile | Owned model bytes | Warmed incremental RSS |
| --- | ---: | ---: |
| Fitted W7A8 reference | 6,738,951 | 7,901,184 bytes |
| Accepted single-thread profile | 6,763,588 | 7,923,712 bytes |

The candidate adds about 24.1 KiB of owned storage for weight alignment and
bookkeeping. Measured warmed RSS increased by 22,528 bytes.

The retained screening, quality, safety logs, memory, library/source hashes and
final timing are collected in [further_summary.json](../results/further_summary.json).
Source manifests retain C/header/CMake hashes; numerical measurements and
library identities are preserved.

## Reproduce

Use the existing Linux development image and saved evaluation fixtures. From
`native_inference/`, prepare the pinned graph and block fixtures if absent:

```sh
python download_model.py dpdfnet8_48khz_hr
python native/generate_extended.py models/dpdfnet8_48khz_hr.onnx models/rework8
python native/export_blocks.py models/dpdfnet8_48khz_hr.onnx models/native_blocks
python native/experiments/further_optimization.py build baseline best_single
python native/experiments/further_optimization.py screen best_single
python native/experiments/further_optimization.py asan best_single
python native/experiments/further_optimization.py scalar best_single
python native/latency_validation.py --model models/dpdfnet8_48khz_hr.onnx --weights models/rework8/weights.f32 --baseline-build build/further_baseline --candidate-build build/further_best_single --frames 1000 --output results/further_best_single_validation.json
python native/experiments/further_exact_validation.py --baseline-build build/further_baseline --candidate-build build/further_best_single --workers 4 --output results/further_best_single_audio.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python native/experiments/further_fp_environment.py --baseline-build build/further_baseline --candidate-build build/further_best_single --frames 64 --verify-single-thread --output results/further_selected_fp_environment.json
```

Run memory and latency jobs sequentially after correctness jobs finish:

```sh
python native/experiments/further_optimization.py memory baseline best_single
python native/experiments/further_optimization.py final best_single
python native/experiments/further_optimization.py summary best_single
```

The driver preserves guarded source copies under `scratch/further_optimization/`
and libraries under `build/further_*/`. Existing snapshots are reused for
subsequent builds. The accepted library SHA256 is
`992634818ee6e2f8301c63d33a3a279da8e8f054daed3c47eadcfe94926c3771`;
the reference SHA256 is
`bf71a93d37cfca5d69a1ce2ee55341e75c400a539e80425b961793396420e4ee`.

## DPDFNet-2 results

The same five accepted transforms apply to `dpdfnet2_48khz_hr`, with its own
unchanged generated graph and weights. No additional optimization or numerical
approximation was added. All inference remains on the calling thread.

| Execution | Fitted W7A8 reference → accepted mean | Time reduction |
| --- | ---: | ---: |
| Continuous | 0.838 → 0.813 ms | 2.92% |
| Paired 10 ms cadence | 0.990 → 0.944 ms | 4.61% |
| Standalone 10 ms cadence | **0.974 → 0.939 ms** | **3.58%** |

| Standalone measurement | Reference → accepted |
| --- | ---: |
| p99 | 1.176 → 1.176 ms |
| Observed maximum | 2.272 → 2.328 ms |
| Whole-process CPU per hop | 0.982 → 0.947 ms |
| Native owned bytes | 5,406,675 → 5,424,496 |
| Warmed incremental RSS bytes | 6,436,864 → 6,418,432 |

Timing and memory use the same protocols as DPDFNet-8. Both DPDFNet-2 builds
had zero calls above 10 ms across 12,000 timed calls each; all cadence calls
completed before the next release. The standalone maximum increased slightly
and p99 was nearly unchanged, so the lower mean is not a worst-case guarantee.
The accepted build owns 5.17 MiB and its warmed incremental RSS is 6.12 MiB.
Alignment adds 17,821 owned bytes; the small RSS difference is not evidence
of a memory reduction. Cross-model comparisons use separate timing sessions.

All 65 saved audio fixtures passed, totaling 1,339.149 seconds and 134,332 hops.
Every output spectrum, full recurrent state and aligned PCM sample was byte
identical to the live reference; all values remained finite and reset replay
matched. All six scored waveform hashes from
[dpdfnet2_overview_quality.json](../results/dpdfnet2_overview_quality.json) also
matched. Its model and weight hashes are checked against that saved report.
The rebuilt reference library is identical to the preserved fitted W7A8 build.
Thus the README's six-mixture quality scores remain valid; the 65-file output
regression does not add perceptual scores or establish equivalence to INT8/FP16.

Release and ASan/UBSan each passed five C contracts; scalar-only passed four.
The three compatibility precision modes passed 1,000 recurrent frames and
independent-context replay. The additional quiet/high-level/silence synthetic
stream passed 1,000 hops. All eight rounding/denormal settings passed 64 hops
with exact output/state and one observed OS thread throughout create/process/destroy.
Independent-context concurrency is a correctness check; it adds no inference workers.

Complete evidence, source/library hashes, safety logs and timing tails are in
[further2_summary.json](../results/further2_summary.json). The accepted library
SHA256 is `3669e2784dc5f3ae0dc85b2078dadd663f59773fae8a785ca5b7964a65d07f5f`;
the reference is `ba199d02ef9111007cff638eabb8070da01bb9593c9ac756fdffdd34b1c357c7`.

### Reproduce DPDFNet-2

From `native_inference/` in the same Linux environment, with saved fixtures:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python download_model.py dpdfnet2_48khz_hr
python native/generate_extended.py models/dpdfnet2_48khz_hr.onnx models/rework2
python native/experiments/further_optimization.py --model-size 2 build baseline best_single
python native/experiments/further_optimization.py --model-size 2 asan best_single
python native/experiments/further_optimization.py --model-size 2 scalar best_single
python native/latency_validation.py --model models/dpdfnet2_48khz_hr.onnx --weights models/rework2/weights.f32 --baseline-build build/further2_baseline --candidate-build build/further2_best_single --frames 1000 --output results/further2_best_single_validation.json
python native/experiments/further_exact_validation.py --model models/dpdfnet2_48khz_hr.onnx --weights models/rework2/weights.f32 --baseline-build build/further2_baseline --candidate-build build/further2_best_single --model-quality-report results/dpdfnet2_overview_quality.json --workers 4 --output results/further2_best_single_audio.json
python native/experiments/further_fp_environment.py --baseline-build build/further2_baseline --candidate-build build/further2_best_single --weights models/rework2/weights.f32 --frames 64 --verify-single-thread --output results/further2_selected_fp_environment.json
```

After correctness jobs finish, run the measurement phases sequentially:

```sh
python native/experiments/further_optimization.py --model-size 2 screen best_single
python native/experiments/further_optimization.py --model-size 2 memory baseline best_single
python native/experiments/further_optimization.py --model-size 2 final best_single
python native/experiments/further_optimization.py --model-size 2 summary best_single
```

Sources are guarded under `scratch/further_optimization2/`, libraries under
`build/further2_*/`, and retained results under `results/further2_*.json`.
The default driver still reproduces DPDFNet-8 under its original paths.
