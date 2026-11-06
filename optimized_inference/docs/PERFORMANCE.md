# Performance, memory and output quality

The current defaults were selected separately for models 2 and 8 on an
Intel Core i7-7700T. The older i7-8700 latency/quality records below describe
the historical package baseline; they were not remeasured for these defaults.

## i7-7700T final per-model selection (2026-10-27)

G replaces depthwise gather with bounded loads/permutations; V combines an
AVX2/FMA target on the generated graph with `-fno-math-errno`; B combines
model-8 AVX2 batch fusion with `quantize64`; S writes model-8 recurrent state
directly; J aligns branches. V and B are predefined groups, not claims about
each internal edit independently. Batch rows are internal frequency rows of
one frame; these gains do not require multiple microphones.

| Model | Selected factors | Baseline mean ± run SD, ms | Final mean ± run SD, ms | Time reduction | Adjusted interval |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | `GV` | 0.9536 ± 0.0108 | 0.9086 ± 0.0095 | +4.723% | [+4.065, +5.377]% |
| 8 | `GVBSJ` | 2.0015 ± 0.0179 | 1.8978 ± 0.0061 | +5.177% | [+4.799, +5.554]% |

These are independent confirmation measurements with 36 mirrored blocks,
two fresh processes per variant per block (72 runs), 120 warmup hops and
500 timed hops per process, paced every 10 ms. Both logical CPUs of physical
core 2 (CPU 2 and HT sibling CPU 6) were reserved through a cgroup v2 isolated
partition; only CPU 2 ran workers. Ordinary processes, including new ones,
were limited to CPU 0,1,3,4,5,7. IRQ/unbound work/watchdog placement was steered;
residual bound kernel work and package/firmware effects remain. Workers ran
without root capabilities. All settings were restored after each phase.

Final confirmation used performance governor/EPP with the original frequency
bounds. Discovery used a 3.6 GHz request range after the first identical-binary
control produced a marginal false gain and halted before any optimization
outcomes were collected. That pilot is retained. Both phases included A/A
controls, fixed counts and all slow observations. The statistical unit is a
complete mirrored block, never an individual hop. The t interval for mean
paired block log ratios, conditional randomization test and block-bootstrap
sensitivity retain the earlier validated statistical functions. Bonferroni
families were 32 for discovery and 40 for confirmation/guard comparisons.

Each retained factor was also removed from the final combination for a direct
comparison. These are conditional contributions in that combination; the
percentages below must not be added.

| Model | Factor | Time reduction from adding factor | Adjusted interval | Individually confirmed |
| --- | --- | ---: | ---: | --- |
| 2 | G | +2.211% | [+1.464, +2.952]% | yes |
| 2 | V | +2.271% | [+1.720, +2.819]% | yes |
| 8 | G | +1.179% | [+1.012, +1.346]% | yes |
| 8 | V | +1.563% | [+1.111, +2.012]% | yes |
| 8 | B | +1.114% | [+0.962, +1.266]% | yes |
| 8 | S | +1.132% | [+0.922, +1.341]% | yes |
| 8 | J | +0.883% | [+0.698, +1.068]% | yes |

| Model | Median run p99, baseline → final, ms | Observed maximum, baseline → final, ms | Calls >10ms | Late completions |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.1384 → 1.0971 | 1.5815 → 1.5217 | 0 → 0 | 0 → 0 |
| 8 | 2.2385 → 2.1421 | 3.8238 → 2.4383 | 0 → 0 | 0 → 0 |

A separate guard used the original `powersave` governor and
`balance_performance` EPP, with 12 blocks of 300 hops. Neither model's gain
passed both adjusted-interval and randomization-test criteria in that mode.
The model-8 point estimate was negative with a wide interval spanning zero.
The confirmed gains above apply to the measured performance policy; they are
not established for every power policy.

| Model | Original-power time reduction | Adjusted interval | Adjusted p | Confirmed |
| --- | ---: | ---: | ---: | --- |
| 2 | +6.700% | [+1.148, +11.940]% | 0.101 | no |
| 8 | -1.542% | [-8.246, +4.747]% | 1.000 | no |

Final review identified a discreteness limitation in the original 12-block
guard: its conditional permutation test could not reach adjusted p<0.05 with
family size 40 (even an unequal assignment has at most 792 permutations,
40/792>0.05; a balanced assignment's symmetric extremes give 80/924).
The original guard is retained. After that review, a separate independent
36-block original-power follow-up was fixed before its collection, using the
same binary, 300 timed hops per process and new randomization. It is a
post-plan follow-up, not part of the initial plan. Across main confirmation,
original guard and this follow-up, 21 contrasts use the unchanged family-40
correction; the unused simplification rounds are cancelled. No further
repeat-until-significant measurements are planned.

| Model | Follow-up original-power time reduction | Adjusted interval | Adjusted p | Confirmed |
| --- | ---: | ---: | ---: | --- |
| 2 | +5.142% | [+2.556, +7.658]% | 0.0020 | yes |
| 8 | +1.508% | [-1.240, +4.181]% | 1.0000 | no |

Final validation exposed a static PGO-generation link error: static consumers
were missing GCC's profiling runtime link option. That option now propagates
only for static GENERATE builds. A fresh shared PGO release build after this
build-rule fix is byte-identical (including its ELF hash) to the library used
for all confirmation timings. Failed logs and the before/after CMake hashes
are retained; the measured runtime implementation did not change.

After selection, the unused global/model-2 branch-alignment experiments and
all alignment switches were removed from the production build. Alignment is
now fixed to model 8 only. Fresh PGO release compilation and the available
release contracts passed, and the resulting library is byte-identical to the
measured binary. Archived experiments remain separate from production sources.

The final build's VNNI objects for both models are byte-identical to baseline.
Branch alignment is cancelled specifically for VNNI sources; the refined
production implementation, not the broader discovery prototype, received the
independent confirmation. AVX-VNNI runtime dispatch, AVX2/FMA baseline, seven
public exports and libc/libm-only shared dependencies are preserved. CPUs
without AVX2/FMA continue to be rejected safely; this is not a scalar fallback.

Every spectrum and complete state matched baseline bitwise over 65 independent
EARS/stress streams per model, including all four supported in-place modes and
reset replay. This is distinct from the unavailable historical 65-file corpus
below and does not add new perceptual scores. Release, ASan/UBSan, static,
no-VNNI and installed-consumer checks passed; native VNNI execution is skipped
on this CPU. The available QEMU TCG also rejects AVX-VNNI: all six direct
VNNI tests skip there; the two dispatch tests select AVX2. VNNI execution is
not covered by this host or emulator. Object identity and dispatch preservation
are verified; full emulator outcomes are recorded separately. S removes copying,
not the allocated arena size. FFT, synthesis, I/O, creation and the unchanged
50 ms algorithmic delay are outside the neural-call latency figures.

The [portable result summary](../validation/i7_7700t_final.json) records exact
intervals, per-factor outcomes, test counts, CPU-mask/emulation results and
source/library hashes. The [paired block observations](../validation/i7_7700t_blocks.json)
include all 57 completed contrasts, including the initial failed control, with
block log ratios, assignment orders, analysis seeds and source hashes. Complete raw journals, plans and artifacts are local
under `.local/i7-7700t-final-2026-10-27` and the generated final HTML report.

## Historical package measurements

### Latency and memory

Intel **Core i7-8700**, GCC 12.2, Linux x86-64 under Docker/WSL2; one unpinned
calling thread, one hop every 10 ms. Inputs are precomputed causal spectra from
six held-out EARS/WHAM 48 kHz speech/noise mixtures. FFT, synthesis, audio I/O,
resampling and context creation are excluded. The algorithmic delay remains
**50 ms**.

| Final model | Mean ± SD / hop ↓ | Median run p99 ↓ | Observed maximum ↓ | Warmed incremental RSS ↓ | Context-owned heap ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dpdfnet2_48khz_hr` | **0.774 ± 0.045 ms** | 0.955 ms | 1.657 ms | **5.201 MiB** | **3.800 MiB** |
| `dpdfnet8_48khz_hr` | **1.670 ± 0.058 ms** | 1.953 ms | 3.530 ms | **6.594 MiB** | **5.077 MiB** |

Each model has **32 fresh-process runs**, with 120 warmup hops and 1,000 timed
hops per run. The same preallocated C benchmark, library and six held-out
spectrum files were used throughout. Twelve previously verified package runs
form the first chronological batch; twenty additional runs form two further
batches of ten per model, with randomized model/input order. Every run is
retained, including slow runs. These are three batches on the same host, not
independent days or machines.

**Mean ± SD** is the arithmetic mean and sample standard deviation (n−1) of
those 32 run means. It describes run-to-run variation; it is not the SD of
individual hops, a confidence interval or a worst-case bound. P99 is the median
of the 32 per-run p99 values; maximum is the largest recorded call. There were
**zero calls over 10 ms and zero late cadence completions across 32,000 timed
calls per model**. Finite observed maxima do not bound future peaks.

RSS is the median resident memory increase above a fresh C worker's baseline
after input/timing/state buffers were allocated and before model creation.
It includes faulted-in state/code, allocator retention and context storage.
Source weight buffers are freed before sampling. Total warmed worker RSS was
**11.041 MiB** for model 2 and **12.445 MiB** for model 8, including preloaded
spectra and ordinary process memory. These totals are not whole-application
memory requirements. RSS differs from exact owned allocations and model files.

The [32-run evidence](../validation/repeated_timing.json) records every run,
per-batch statistics, protocol and input/library identities. It supersedes the
initial 12-run package summary for the overview table.

#### Benchmark history and comparisons

The separate [packaging comparison](../validation/package_timing.json) used
balanced ABBA/BAAB order against the accepted reference binaries, with 12
package runs per model. In that first batch, model-8 reference/package means
were 1.624/1.601 ms and model-2 means were 0.720/0.721 ms. That matched check
supports preservation of the accepted runtime's performance. Its reference
rows must not be paired with the later 32-run aggregate, which includes other
collection batches.

The earlier optimization audit's **32 tests were comparison/control blocks**:
16 per model, four processes per block, 128 processes overall. Each candidate
had 24 comparison runs. The current table instead reports 32 actual package
runs per model; these counts describe different designs.

Earlier audit results used a Python/ctypes caller, with different warmup and
collection runs. Absolute numbers across those methods/sessions are not
directly comparable. The unchanged accepted model-8 reference measured 1.744 ms
in that audit and 1.624 ms in the first C benchmark batch. These records do not
separate caller overhead from machine/session effects. Mean ± SD improves the
description of variability; it does not establish an optimization gain.

Exact context-owned allocations are **3,984,320 bytes** and **5,323,412 bytes**,
respectively, including the 16-byte public API wrapper. Caller state requires
another 225,744 or 360,912 bytes, plus spectral/audio buffers and stack. Weight
files contain 10,329,776 or 14,532,272 bytes of float32 data; creation repacks
them into the optimized representation and temporarily needs both forms.

This RSS protocol belongs to the C worker and should not be mixed with
imported-Python-runtime memory measurements.

### Historical output quality

Scores are means over the same **six EARS/WHAM mixtures** with clean references:
`00033`, `00046`, `00084`, `00133`, `00200`, `00364`. These are a small quality
screen, not a full EARS test-set benchmark.

| Final model | PESQ-WB ↑ | STOI ↑ | SI-SNR (48 kHz, dB) ↑ | SIGMOS SIG ↑ | SIGMOS NOISE ↑ | SIGMOS OVRL ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dpdfnet2_48khz_hr` | 2.44130 | 0.935874 | 16.0841 | 3.86312 | 4.61457 | 3.28124 |
| `dpdfnet8_48khz_hr` | 2.52099 | 0.940449 | 16.6446 | 3.84506 | 4.63650 | 3.35663 |

Audio, SI-SNR and **SIGMOS** use native 48 kHz data. PESQ-WB requires resampling
to 16 kHz; STOI internally uses 10 kHz. There is no native-48-kHz PESQ score in
this table. SIGMOS supplies the fullband signal, noise and overall MOS estimates.

The scores are retained from the fitted W7A8 reference evaluation. The rebuilt
package reproduces all scored waveforms byte for byte, so rescoring unchanged
PCM would not evaluate a new output. The [quality record](../validation/quality.json)
contains unrounded means, clip IDs, scorer/model hashes and source-study hashes.

W7A8 quantization and fitted gates change the model relative to original FP32
ONNX. Exact regression here means preservation of the **fitted W7A8 reference**;
it does not imply lossless conversion from ONNX.

### Historical broader regression and safety checks

Both models reproduce every saved spectrum, complete recurrent-state history
and aligned PCM hash across **65 files per model**: 50 noisy mixtures, ten
approximately -50 dB speech clips, two clean clips, and three two-minute noise
files (white, pink and mechanical). That is **22.3 minutes / 134,332 hops per
model**, including flush hops. Reset replay and independent concurrent contexts
also match. This broader check preserves outputs; its 65 files were not all
scored for perceptual quality.

All **19 contract tests** pass in the release, ASan/UBSan and static builds.
They cover the public API, both model lifetimes, weight ownership, reset,
in-place processing, graph layouts, scalar numerical oracles and guarded
assembly boundaries/FP control modes. Separate installed shared/static CMake
consumers build and run. The shared kernel consolidation also matches the
accepted per-model C sources after preprocessing.

See [release evidence](../validation/release.json),
[reference hashes](../validation/reference_streams.json), and
[numerical provenance](../validation/provenance.json).

## Measure your application

```sh
build/dpdfnet_benchmark 2 models/dpdfnet2_48khz_hr/weights.f32 1000 paced
build/dpdfnet_benchmark 8 models/dpdfnet8_48khz_hr/weights.f32 1000 paced
```

Without an input file these are deterministic synthetic smoke tests. For real
audio, append a raw little-endian float32 spectrum stream as the sixth argument:
one `[481,2]` frame per hop, using the analysis convention in [API.md](API.md).
The example's `spectra()` helper can produce these frames from 48 kHz mono PCM.
Use `continuous` instead of `paced` to measure back-to-back throughput.

For a repeatable mean ± SD summary in fresh processes:

```sh
python3 tools/benchmark.py --runs 32 --spectra speech_frames.f32
```

This standard-library driver alternates model order and saves all run summaries
under `validation/local/`. Omit `--spectra` for synthetic inputs or pass multiple
spectrum files to rotate real clips. `--model 2` or `--model 8` selects one model.

Run benchmarks without competing builds or tests. Use multiple fresh processes,
balanced order and the same input/warmup/cadence for comparisons. Measure the
complete application pipeline separately when budgeting callback deadlines.

### Paired optimization experiments

The paired comparison, kernel comparison and stream comparison Python tools
require Python 3.9 or newer. These optional development tools are separate
from the Python 3.8+ build driver and from the dependency-free C runtime.

`tools/compare_benchmarks.py` compares two existing builds in randomized,
balanced ABBA/BAAB blocks. Each letter starts a fresh process; all four runs in
a block use the same model, input, 120-hop warmup and measurement length. Pin
both variants to the same logical CPU and keep other builds, regression tests
and profilers idle during collection:

```sh
python3 tools/compare_benchmarks.py \
  --baseline /path/to/reference-build --candidate /path/to/changed-build \
  --models 2 8 --blocks 24 --hops 500 --mode paced --cpu 2 \
  --family-size 2 --seed 43000 --label comparison \
  --spectra speech.f32 quiet.f32 silence.f32 \
  --output validation/local/comparison.json
```

Choose the inputs, sample count, statistical family and acceptance rule before
collection. Repeat the same comparison with identical builds as a noise control.
Exploratory candidate selection should be followed by a separate confirmation
experiment; a favorable result selected from many trials is not confirmation.

The driver retains every run, raw per-hop double-precision timings, library and
input hashes, process statistics and before/after CPU telemetry. It does not
remove slow runs. Run means are summarized with sample SD. The inferential unit
is a complete four-process block: hops from the same process are correlated and
must not be counted as independent replications. The primary interval is a
Student t interval for the mean paired block log time ratio, transformed to
percentage time reduction. `--family-size` applies a Bonferroni correction to
the interval; multiply the reported raw randomization p-value by the same
family size, capped at one. A block bootstrap interval is a sensitivity check.
`tools/analyze_comparison.py` can reanalyze an immutable journal separately.

`DPDF_BENCH_SAMPLES=/path/to/timings.f64` also enables raw timing output in an
individual C benchmark run. The file contains one native-endian binary64
millisecond duration per timed hop. Writing happens after the timed region.
Per-run p99, observed maximum, calls exceeding 10 ms and late cadence completions
remain separate from the mean-throughput comparison. Short runs and a finite
observed maximum cannot establish a worst-case latency bound.

### Hot kernels and hardware counters

Configure with `-DDPDF_BUILD_KERNEL_BENCHMARKS=ON` to build the private kernel
workers. `tools/compare_kernels.py` uses balanced fresh-process blocks and at
least 100 ms of same-kernel warmup. These workers measure amortized hot-cache
call time, not per-call tail latency. Preserve end-to-end model measurements:
a large speedup in a small kernel need not materially speed up the model.

`tools/profile_cpu.py` collects Linux perf events only after benchmark warmup,
using perf's FIFO enable/disable protocol. For example:

```sh
python3 tools/profile_cpu.py --build-dir /path/to/build \
  --model 8 --cpu 2 --hops 5000 --spectra speech.f32 \
  --kind record --events cycles:upp --output validation/local/hotspots
```

Precise events depend on CPU/kernel support. Use `--kind stat` for hardware
counter totals; choose event groups that fit the available counters and check
their running percentage. Repeat on the same inputs for both builds. Sampling
percentages describe a share of the measured profile, not absolute saved time.
Do not use timings gathered under a profiler as the latency experiment.

Where perf permissions require elevation, `--sudo` elevates only the collector.
Its workload drops to the invoking user's UID/GID and clears capabilities.
Neither this tool nor the benchmark changes global perf permissions, frequency
policy or security mitigations. `--perf` and `--perf-library-path` support an
isolated perf extraction without installing an application runtime dependency.

### Independent stream regression

`tools/make_cpu_fixtures.py` creates a separately documented 48 kHz regression
suite from official EARS demonstration audio and deterministic stress signals.
It requires development-only NumPy and FFmpeg. This suite does not reconstruct
the historic 65-file release corpus described above and does not add perceptual
quality scores. `tools/compare_streams.py` compares multiple library builds on
every spectrum and complete recurrent state at every hop, including in-place
buffers and reset replay. Keep these correctness jobs separate from timing.
