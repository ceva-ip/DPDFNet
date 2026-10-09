# Performance, memory and output quality

These measurements describe the final combined library built from this folder,
with the recommended model-8 compiler profile. They are host-specific
observations, not portable latency or worst-case execution-time guarantees.

## Latency and memory

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

### Benchmark history and comparisons

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

## Output quality

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

## Broader regression and safety checks

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
