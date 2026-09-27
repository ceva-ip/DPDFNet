# W7A8 latency follow-up: packing and fitted GRU activations

Investigation dated 2026-09-27. Target: `dpdfnet8_48khz_hr`, using the previously
quality-tested [W7A8 research build](INT8_RANGE_EXPERIMENTS.md) as the reference.
These experiments keep the 48 kHz, 480-sample streaming hop and FP32 recurrent
state. They do not change the model's 50 ms algorithmic audio delay.

## Results

**Best new candidate: 32-value activation packing plus a fitted degree-5
exponential for the GRU gates.** Across the longer runs it reduces mean inference
time by **3.6–4.9% beyond W7A8**, while passing the original activation-error
tolerance. It is not byte-exact; the speech/noise screen below shows small
quality changes. The separate, exact packing improvement saves **1.2–2.7%**
and preserves all 65 previously evaluated audio outputs byte for byte.

| Candidate | Execution | Matched W7A8 reference → candidate, mean ms/hop | Reduction |
| --- | --- | ---: | ---: |
| Exact packing | Continuous | 1.961 → 1.937 | 1.21% |
| Exact packing | Paired 10 ms cadence | 2.090 → 2.056 | 1.59% |
| Exact packing | Standalone 10 ms cadence | 2.043 → 1.987 | 2.70% |
| Packing + fitted gates | Continuous | 1.941 → 1.849 | 4.74% |
| Packing + fitted gates | Paired 10 ms cadence | 2.071 → 1.970 | 4.86% |
| Packing + fitted gates | Standalone 10 ms cadence | 1.992 → 1.921 | 3.56% |

Each candidate has its own matched reference runs; absolute times from different
experiments should not be compared as though they were simultaneous.
Preallocated-buffer diagnostics retain the gain: **1.898 → 1.809 ms/hop
(4.71%)** for fitted gates, and **1.919 → 1.861 ms/hop (3.01%)** for exact packing.

**Tail latency did not improve uniformly.** For fitted gates, paired-cadence
p99 improved from 2.472 to 2.312 ms, but standalone-cadence p99 increased from
2.383 to 2.575 ms; the standalone maximum rose from 3.035 to 3.977 ms. Exact
packing also had a larger continuous-run maximum, 7.263 versus 3.406 ms.
Neither selected candidate nor its matched reference exceeded 10 ms in these
final runs (10,000 timed calls per implementation per comparison).

The subsequent [peak-latency investigation](TAIL_LATENCY_INVESTIGATION.md)
traces 30,000 more calls and adds a reusable-buffer streaming runner. It removes
observed in-call allocation/GC events, but C-only tests still show occasional
slow calls and CPU affinity does not consistently lower the maximum.

The production W8A8 implementation and integration presets are unchanged.
Candidate libraries live in separate research build directories and are not
drop-in updates to a shipped preset. Owned model allocation remains
**6,738,951 bytes** in the selected candidates.

## Directions considered, in initial priority order

| Priority | Direction | Decision after investigation |
| --- | --- | --- |
| 1 | Dynamic activation packing and INT8 output processing | Implemented 32-value packing and fused dot-product epilogues; retained packing as the simpler exact candidate. |
| 2 | GRU sigmoid/tanh arithmetic | Tested reciprocal refinement, a shorter Taylor polynomial, then fitted coefficients. The fitted degree-5 polynomial is the strongest new candidate that passes the existing activation tolerance. |
| 3 | INT8 register tiles, loop scheduling and cache traversal | Tested eight-row tiles, two/four-way dot-loop unrolling and column-first traversal. No convincing improvement over the selected candidates. |
| 4 | Compiler optimization | Tested LTO, combinations, and PGO trained on synthetic, clean, quiet and noise inputs. Benefits did not reliably add to packing. |
| 5 | Remaining convolution/layout work | Profiled. Convolutions are a smaller opportunity now; prior convolution/register/layout optimizations already apply. No new convolution rewrite in this pass. |
| 6 | Two persistent workers for the two encoder branches | Potentially a larger wall-time improvement, but consumes another core and needs separate scratch storage plus synchronization. Not implemented or measured here. |
| 7 | Fixed activation scales, lower precision, sparsity | Could remove more work, but changes quantization/quality assumptions. Static ranges are especially questionable for the existing −50 dBFS cases. Not tested in this pass; W8A7 remains excluded as requested. |
| 8 | VNNI/AVX-512, GPU/NPU backends | A separate hardware/backend experiment. This CPU exposes AVX2/FMA/F16C, but no AVX-VNNI, AVX-512-VNNI or AVX-512F. No speedup claim for untested hardware. |
| 9 | Pruning, distillation, fewer recurrent blocks/channels | Requires a different trained model and renewed quality evaluation. Not a kernel-only update. |
| 10 | Batching/skipping audio hops | Not pursued: batching adds buffering latency, while skipping changes the streaming computation and can damage quiet speech. |

Handwritten assembly remains an option for direction 3. The preceding assembly
experiment produced only a small gain, so this pass first reduced arithmetic
and data packing. No new assembly kernel is claimed in this report.

## Profile and screening

The refreshed, instrumented W7A8 profile attributes approximately **1.44 ms**
to the DPRNN blocks, **0.25 ms** to convolutions, and **0.085 ms** to remaining
Gemm/MatMul operators. Within DPRNN, matrix operations account for about
**0.95 ms**, gates **0.35 ms**, and normalization **0.058 ms** per hop.
Instrumentation overhead makes these diagnostic attribution numbers, not
release-build latency measurements.

The gates remain material after W7A8 removes the signed-byte conversion work
from the integer dot products. This is why activation arithmetic was revisited
instead of spending the entire pass on increasingly elaborate integer loops.

Positive values mean lower mean wall time than W7A8. These are initial screens,
not the final cadence measurements above.

| Experiment | Mean reduction | Result |
| --- | ---: | --- |
| 32-value activation packing | +3.00% | Selected exact candidate |
| Fused recurrent dot-product epilogue | +1.06% | Small gain; did not add reliably to packing |
| Reciprocal refinement in gates | −0.69% | Slower and changed arithmetic; dropped |
| Dot-loop unroll ×2 | +0.20% | Too small to select |
| Dot-loop unroll ×4 | +0.44% | Too small to select |
| Gate-loop unroll ×4 | +1.55% | Tested in combinations |
| LTO alone | +1.82% | Tested in combinations |
| Eight-row integer tile | −2.13% | Slower; dropped |
| Column-first cache traversal | +0.08% | Negligible |
| Degree-5 Taylor gates | +4.32% | Fails existing activation tolerance |
| Packing + epilogue | +1.46% | No additive gain |
| Packing + epilogue + LTO | +1.86% | No clear advantage over simpler packing |
| Packing + gate unroll | +1.34% | No additive gain |
| Packing + gate unroll + LTO | +3.02% | Essentially tied with initial packing screen |
| Packing + epilogue + gate unroll + LTO | +2.47% | No clear advantage |
| Previous combination + PGO | +2.84% | No clear advantage despite added build complexity |
| Packing + degree-5 Taylor gates | +5.44% | Retained as negative numerical-contract result |
| Packing + fitted degree-5 gates | +4.88% | Selected faster research candidate |

Screening uses three balanced AB/BA continuous runs of 600 timed hops, plus
100 warmup hops per run. Every exact candidate also compares output and state
byte for byte for 500 recurrent frames. Approximate candidates carry independent
states and report numerical differences instead. All 18 candidate builds pass
the five existing C contracts. Those C tests do **not** imply that a changed
activation approximation passes the separate Python activation-error contract.

Small screen differences should not be treated as reliable rankings. In
particular, packing's initial 3% result became a smaller improvement in the
longer continuous run. Combining individually beneficial changes did not
consistently improve performance; compiler scheduling/code layout can change
with the combination. No individual percentage savings were added together.

## Selected implementations

### Exact: 32-value activation packing

The previous quantizer converted eight floats at a time, extracted two 128-bit
halves, packed them down to bytes, and stored eight bytes. The candidate converts
32 floats, packs them with AVX2, permutes the four-byte groups into the correct
order, and stores 32 bytes. It retains the original multiply/add, rounding,
clamping, scale, zero-point and eight-value tail arithmetic.

The independent scalar INT8 oracle covers short widths, tails, unaligned buffers,
zero/extreme inputs, batched calls and paired matrices. The candidate also passed:

- **1,000 recurrent frames in each of FP32, FP16 and W7A8**, with byte-identical
  output/state relative to the preserved W7A8 library in the corresponding mode.
- Four independent contexts run serially and concurrently, matching in each mode.
- All five contracts under **ASan + UBSan**.
- **65 complete 48 kHz audio files / 22.32 minutes / 134,332 model hops**, matching
  the previously saved W7A8 PCM hashes exactly: all 50 EARS-WHAM mixtures,
  ten quiet mixtures, two clean controls and all three two-minute noise streams.

The existing quality scores on those exact waveforms are therefore unchanged;
there was no need to recompute their PESQ/STOI/SIGMOS scores.

### Faster research candidate: packing plus fitted degree-5 exponential

The original vector exponential reduces the argument to approximately
`[-ln(2)/2, ln(2)/2]` and evaluates a degree-7 Taylor polynomial with FMA.
The candidate retains range reduction, clamping, exponent reconstruction,
division, and FP32 gates, but uses a degree-5 Chebyshev interpolation polynomial.
Coefficients are computed in FP64, rounded to FP32, and emitted as exact C hex
literals. This removes two dependent polynomial FMAs per exponential evaluation.
It does not use `-ffast-math` or change the W7A8 quantization grid.

The direct degree-5 Taylor experiment was faster but failed the original
activation tolerance. It is retained as a negative result, not the recommended
approximation. Reciprocal refinement was slightly slower and was also dropped.

The fitted candidate passes the **unchanged `atol=2e-7, rtol=2e-7` activation
contract**. A separate 2,000,008-value sweep over `[-100,100]`, dense near zero,
measured maximum absolute errors of **1.01e-7 for sigmoid** and **1.29e-7 for
tanh**, relative to FP64 reference formulas. These are tested bounds, not a
formal proof over all FP32 bit patterns.

The fitted candidate is **not byte-exact** relative to W7A8. Small activation
changes can cross a later dynamic-quantization rounding boundary; passing the
activation tolerance alone is not evidence of unchanged speech quality.

## Fitted-candidate quality and long-stream checks

The follow-up uses six existing EARS-WHAM mixtures, one per test speaker, two
−50 dBFS speech cases including `low_00525`, one clean control, and all three
120-second noise fixtures. Each file starts from a fresh state and then runs
continuously, including drain hops. All output/state values were finite.

Mean changes below are **candidate minus preserved W7A8**, measured again on
the same signals. This is a small engineering screen, not a full 50-clip
non-inferiority study or a listening test.

| Metric | Six mixtures | Two quiet cases | One clean control |
| --- | ---: | ---: | ---: |
| SI-SNR, native 48 kHz (dB) | −0.00467 | −0.02865 | +0.00241 |
| PESQ-WB, explicitly resampled to 16 kHz | +0.00129 | −0.00124 | −0.00202 |
| STOI, algorithm internally uses 10 kHz | −0.0000236 | −0.0000431 | −0.0000114 |
| SIGMOS SIG, native 48 kHz | −0.00395 | +0.00016 | +0.00112 |
| SIGMOS OVRL, native 48 kHz | −0.00031 | +0.01269 | −0.00100 |

The worst mixture SI-SNR change was −0.01132 dB. The known quiet-speech outlier
changed by −0.05982 dB SI-SNR and −0.00275 PESQ; this does not resolve the
pre-existing level sensitivity shared with original ONNX.

Across the three long noise files, attenuation changed by **−0.07255 to
+0.01713 dB**. No 20 ms window amplified its input. The change in final-30-second
attenuation stayed within 0.0053 dB of W7A8. Speech-reference metrics are not
assigned to speech-free noise.

## Measurement scope and next priorities

Measurements are single-thread spectral-model calls on the i7-8700 under
Linux/Docker/WSL2 with GCC 12.2, unrestricted CPU affinity, the same weights,
and the same development ABI. FFT, overlap-add, device I/O and resampling are
excluded. Host scheduling still affects tails. These are not Windows-native
deployment results or worst-case deadline guarantees.

Final timing uses four 1,000-hop continuous runs, four 1,000-hop paired
10 ms cadence runs, and two standalone cadence runs per implementation, each
with 100 warmup hops. Paired runs reverse call order each hop; standalone runs
reverse implementation order between repeats. Thread CPU time, context switches,
all outliers and calls above 10 ms are retained. Table percentiles are the mean
of run percentiles, not pooled percentiles. Separate preallocated-buffer tests
help distinguish kernel gains from allocation/GC behavior.

The next larger latency direction is branch-level parallelism: the generated
graph has independent magnitude and complex DPRNN branches before fusion. The
current arena reuses addresses across them, so simply adding threads would
introduce races. A correct experiment needs private temporary regions and a
persistent worker, followed by cadence-tail and total CPU-use measurements.
The profile suggests a potentially larger wall-time opportunity than these
remaining single-core micro-optimizations, but no parallel speedup has been
measured here.

For the fitted candidate, the next quality step is the complete 50-clip suite
and all quiet/clean controls before promotion. The exact packing candidate
already has full PCM equivalence on those fixtures.

## Artifacts and reproduction

- [Machine-readable summary](../results/w7_followup_summary.json), including
  library/source hashes and all screen/final timing summaries.
- Final [exact packing timing](../results/w7_followup_pack32_final.json) and
  [fitted-gate timing](../results/w7_followup_pack_fit5_final.json), with all
  per-run distributions and context-switch records.
- [Instrumented profile](../results/w7_followup_profile.json).
- [Exact audio regression](../results/w7_followup_pack32_audio.json) and
  [recurrent/concurrency validation](../results/w7_followup_pack32_validation.json).
- [Fitted-candidate audio/noise results](../results/w7_followup_pack_fit5_audio.json)
  and [activation contracts, including the rejected Taylor variant](../results/w7_followup_activation_contracts.json).
- [ASan/UBSan results](../results/w7_followup_sanitizers.json).
- [Isolated source preparation and screening](experiments/w7_followup.py),
  [final checks/timing](experiments/w7_followup_finalize.py),
  [audio checks](experiments/w7_followup_audio.py), and
  [summary generator](experiments/summarize_w7_followup.py).

Selected artifacts are `build/w7_followup_pack32/libdpdf_full.so` and
`build/w7_followup_pack_fit5/libdpdf_full.so`, with sources under
`scratch/w7_followup/pack32` and `scratch/w7_followup/pack_fit5`.

From `native_inference` in the existing development container, on a fresh
experiment tree with the preserved models/builds available:

```sh
python native/experiments/w7_followup.py profile pack32 epilogue reciprocal unroll2 unroll4 gate4 lto batch8 column_first poly5 combined combined_lto pack_gate pack_gate_lto exact_all pgo pack_poly5 pack_fit5
python native/experiments/w7_followup_finalize.py checks
python native/experiments/w7_followup_finalize.py timing
python native/experiments/w7_followup_finalize.py fit_timing
```

In the existing fullband image, with the previously downloaded fixtures:

```sh
python native/experiments/w7_followup_finalize.py audio
python native/experiments/w7_followup_finalize.py fit_audio
```

Then run `python native/experiments/summarize_w7_followup.py`. Keep quality,
sanitizer and build jobs separate from timing. Preparation refuses to overwrite
existing experiment source directories so earlier binaries/results remain
available for comparison.
