# Exact four-row quantization refinement

Status: full registered qualification completed, 2026-11-10. Both experimental
build options remain OFF by default. Complete regenerated spectral/state/PCM
equivalence and red AVX2 portability passed for both models. The final
72-condition study confirms incremental native model-8 gains on the fast
Ryzen core at steady frequency: continuous +0.810%, paced10 +0.863%. Global
model-2/AVX2/full-audio/Boost-paced superiority remains unqualified. The selected
native path remains CM-only (`8ad669a`). See the
[final qualification](../validation/final_qualification_20261110.json).
The earlier native quantize4 experiment is `1e2cb70`.

## Implementation and selection

`src/common/quantize_refined.inc` implements the shared helpers and candidate V.
`quantize_legacy.inc` preserves the selected baseline; `int8_kernels.inc`
selects between them at compile time. There is no new runtime branch or ABI.
The R and refactor-only diagnostic sources are retained in the local study.
Independent numerical contracts check the refined AVX2 and VNNI helpers even
when the baseline library is selected. The frozen oracle is intentionally
independent of both implementations.

Enable `DPDF_ENABLE_QUANTIZE4_REFINEMENT=ON` for the AVX2 refinement. Additionally
enable `DPDF_ENABLE_VNNI_QUANTIZE4=ON` to apply it to native row-major VNNI
calls. The latter requires the former. Keep a fresh model-8 PGO profile per
configuration. These are experimental build controls, not performance presets.

The [short refinement study](../validation/quantize4_refinement_20261110.json)
contains 160 unprofiled processes / 56,000 calls, 16 helper discovery processes
covering all masks, and 40 separate hardware profiling jobs. CPU 3 was isolated
and sibling 15 parked; settings were restored. The helper's active/all-empty
time means decreased by 12.198%/22.431%; R regressed on some mixed masks.
Those microbench results do not establish a graph-level improvement.

Continuous graph gain estimates for AVX2 models 2/8 were +0.113%/-0.226%, and
for native models 2/8 +0.298%/+0.588%. Every adjusted interval included zero;
paced improvements were also unconfirmed. Therefore neither default changes.
The final gated libraries are byte-identical to the measured candidate when
enabled, and to the original selected baseline when disabled, for both ISAs.

The [isolated-L3 follow-up](../validation/quantize4_lab_20261110.json) uses the
mandatory [benchmark laboratory](BENCHMARK_LAB.md). Native model-8 batch meets
its predeclared gate: +0.832%, adjusted interval [0.580,1.084]%, with matching
A/A precision qualified. No other condition meets all criteria. Do not combine
this verified non-boost result with earlier Boost-enabled percentages or promote
a global VNNI switch from one model/cadence under a different power policy.

The [extended 48-condition matrix](../validation/quantize4_regimes_20261110.json)
tests continuous calls, groups16/64 and 5/10/20 ms cadence under both registered
power profiles, with eight A/A pairs and eight comparison pairs per condition.
One family of 48 contrasts is used. Model-8 VNNI passes all gates for steady
batch16 (+0.971%, interval [0.653,1.288]%), steady batch64 (+0.769%,
[0.447,1.089]%), steady paced10 (+0.835%, [0.480,1.189]%), and isolated-Boost
continuous (+0.739%, [0.516,0.962]%). No other condition passes all gates.
Unqualified/too-small effects are not evidence of zero true benefit.

The earlier short numerical stream check covered 16 fixtures / 32 model jobs rather
than the proposed 65 fixtures / 130 jobs, following the requested shorter run.
It includes 46,487 frames per model and the 17,996-frame recurrent history.
No historical PCM/perceptual corpus or other core/CPU was remeasured. The
confirmation used two previously known inputs, not a new held-out input corpus.

## Objective and evidence

Remove repeated range scans when one or more K64 rows are degenerate, while
sharing range and byte-packing code with the existing quantizer. Preserve exact
floating-point divisions, quantized bytes, scale bits and zero points.

The completed [factorial recheck](../validation/ryzen_vnni_retest_20261110.json)
found a useful diagnostic: ordinary batch40 improved with the old native
quantize4 experiment, but all-zero batch40 regressed by about 1.408%/3.146%
for models 2/8. These are kernel observations, not predicted improvements
from this design. At that stage, incremental full-graph benefit over CM was
unconfirmed.

## Final ready-binary qualification

The [final record](../validation/final_qualification_20261110.json) retains
72 preregistered contrasts: CM versus pre-CM, refined quantize4 versus CM-only
native, and refined versus legacy AVX2; two models, three workloads, two Ryzen
core groups and two power profiles. Counts were fixed at eight A/A pairs and
sixteen comparison pairs per condition, family size 72, using two held-out
timing inputs. The fixed-layout cohort is separate from earlier randomized
layout studies. Total: 3,456 unprofiled processes, 811,008 calls and 288 separate
PMU jobs.

The native model-8 fast/steady continuous gain is +0.810%, adjusted interval
[0.683,0.937]%; paced10 is +0.863%, [0.513,1.213]%. Both clear A/A precision
and useful-effect gates. Compact/Boost continuous model 8 meets practical
noninferiority at the predeclared 0.5% loss margin, but its useful-gain gate
does not pass. No model-2, AVX2 or full-audio quantize4 condition passes every
superiority gate. Only 10/72 A/A conditions qualify at 0.5%; failed precision
is retained as inconclusive, not zero benefit or evidence of a regression.

All 65 regenerated public/stress fixtures (147,711 hops per model) pass
every-spectrum/state, reset/replay and four-alias checks for pre-CM and four
candidates. Another 130 complete PCM jobs verify exact FFT fixture reproduction
and bitwise equal synthesized PCM across all five libraries. Red i7-7700T
passes 130 stream jobs across four libraries; all 130 cross-host diagnostic
digests match. Its single shared L3 and lack of VNNI limit it to numerical
AVX2 checks; all eight service health endpoints remain HTTP 200.

The original historical reference corpus was unavailable and is not certified
by these regenerated fixtures. Complete host callbacks were measured, but
device I/O, resampling, audio-server/service concurrency and hardware loopback
were not. The observed maximum 2.787454 ms and zero missed 10 ms deadlines are
finite observations, not WCET guarantees. A secondary direct cumulative
comparison uses existing nonadjacent block observations; it is explicitly
exploratory, not an added acceptance test or a sum of separate gains.

All settings pass independent final restoration readback. Frozen binaries,
measurement sources, plans, every raw sample, call/process SD, intervals,
subgroups, counters and preparation-only failures are retained under
`.local/final-qualification-20261110/`. After collection, a separately tested
error-path fix preserves controller failure/rollback JSON in future logs;
it does not change the normal measurement path or numerical kernels.

## Existing behavior and its cost

The selected legacy `quantize4_64` in `src/common/quantize_legacy.inc` computes ranges row
by row. On the first `low == high`, it calls the original quantizer for all
four rows and returns. If the first degenerate row is index e (0 through 3),
this performs e+1 preliminary scans plus four fallback scans: five through
eight K64 scans. Computing every range once needs four. For an all-zero
group the scan count therefore falls from five to four, not eight to four.
This is work counting, not a proportional runtime prediction.

The reduction includes zero in every range. A positive constant row has
range [0,c]; a negative constant row has [c,0]. Neither is degenerate.
Here, "empty" means the existing reduced `low == high` condition, not a
zero-length input, silence inferred from the source audio, or `scale == 0`.
Keep the current signed-zero, unordered-min/max and FTZ/DAZ behavior.

## Shared internal structure

Use private, compile-time helpers inside the shared AVX2/VNNI implementation:

1. `range(x,k)` returns the existing low/high reduction, unchanged in its
   lane accumulation and horizontal reduction order.
2. `pack(x,out,k,inv,zp)` retains the existing multiply-then-add sequence,
   conversion, saturation to [0,254], and byte permutation.
3. The original single-row quantizer composes these helpers and retains its
   scalar scale/zero-point calculations and degenerate early return.
4. The four-row quantizer computes four ranges once and forms an empty mask.
   It then uses the same packing helper only for nonempty rows.

These are conceptual names, not a new public API. Keep the constant-K64
wrapper so GCC can specialize/unroll it. Do not force the generic K<=512
wrapper to unroll or inline everywhere. Check actual assembly for duplicated
code and spills before imposing always-inline attributes. Extract helpers in
a separate change so source refactoring and new empty handling can be
compared independently.

No heap allocation, context fields, persistent range cache, function-pointer
calls, CPU probes or new public configuration belong in this path. Reuse is
within this invocation: inputs change between frames. Do not cache ranges
across calls or introduce `restrict` without proving existing alias contracts.

## Three group paths

| Empty mask | Work |
| --- | --- |
| 0000 | Existing three packed exact divisions, then pack all four rows |
| 1111 | Write 256 zero bytes, four exact scales of 1 and four zero points of 0; no parameter divisions or packing |
| Other 14 masks | Reuse the four ranges; zero only empty rows; calculate and pack active rows without a rescan |

Use one group test for all-active and one for all-empty. Initially keep the
four-row loop and conditional stores simple; benchmark before introducing
16-way dispatch, bit scans, manual unrolling, or branch-likelihood hints.
Do not add a separate input pass to detect silence. Compute the mask from
the ranges that are already required.

## Mixed-group implementations to compare

First implement a conservative mixed path: for each active row, compute the
original scalar parameters from its already available low/high, then pack it.
Empty rows get their original constants immediately. This establishes range
reuse without changing mixed-row division scheduling. Call this candidate R.

Then compare candidate V, which keeps the three packed exact divisions even
in mixed groups. For empty lanes only, select a neutral low of +0 and span
of exactly 254 before the first division. For active lanes, retain the
original low and the exact `high-low` result.

Conceptually, with selects performed using the original empty mask:

```text
L = select(empty, +0, low)
D = select(empty, 254, high - low)
S = D / 254
Z = clamp(round_MXCSR((-L) / S), 0, 254)
I = 1 / S
for each row:
    if empty: output[0:64] = 0; scale = 1; zp = 0
    else:     pack_using_existing_order(input, I, Z); scale = S; zp = Z
```

An empty lane now executes 254/254, signed-zero/1 and 1/1. All are exact;
it never executes a spurious 0/0 or 1/0. Do not calculate invalid divisions
first and blend their results afterward. With AVX2 there is no per-lane
arithmetic suppression from a final blend. Packing an empty row is skipped
entirely: do not use zero multiplied by an infinite inverse as a substitute
for the explicit zero output.

Build the active-lane numerator with the original scalar unary-negation
semantics (sign-bit XOR is a candidate); verify against the oracle in every
rounding mode. The old packed helper's `0-low` must not silently become the
definition of scalar signed-zero behavior. Preserve active-lane overflow,
underflow and special-value behavior rather than "repairing" the scale.
In particular, a nonempty tiny range may produce a zero scale under some
FTZ/DAZ settings. That is not a reason to mark the row empty.

R versus V is an empirical choice: fewer division instructions can lose to
extra selects, branches, register pressure or code layout. Sparse groups
may favor R. Choose one measured mixed implementation; do not add dynamic
CPU/vendor policy or mask-popcount thresholds without separate evidence.

## Numerical contract and tests

Required equivalence is bitwise output bytes, scale bits and zero points
against an independent frozen pre-change quantizer, under the supported
masked-exception environment. Preserve MXCSR control bits. The existing
contract does not prove identical sticky exception flags or trapping order:
`nearbyintf` and packed integer conversion can differ in flag behavior, and
batching changes operation order. Record flag differences diagnostically;
do not claim a new exception/trap contract or broaden it without a separate
decision. Do not replace the independent oracle with the refactored helper.

Extend the current quantize4 test (which emphasizes first-row/all-row empty
cases) with the following matrix before speed measurement:

- All 16 empty masks, each row position and mixed +0/-0 empty rows.
- Positive/negative constant nonzero rows, random rows, one nonzero element
  in each of the 64 positions, and extrema in every SIMD lane.
- Four rounding modes times four FTZ/DAZ combinations; subnormal boundaries,
  representable tiny nonzero ranges and scale underflow cases.
- Values near zero-point and packing rounding thresholds, saturation edges,
  and large finite ranges. Diagnostic NaN/infinity cases preserve defined
  baseline behavior; avoid treating undefined C float-to-int conversions as
  a portable numerical oracle.
- Guarded input/output placements at both ends; canaries around the four
  scales/zero points; input immutability. Retain network alias/reset tests.
- AVX2 and native VNNI builds for both models; ISA gates, sanitizer and static
  builds, fresh model-8 PGO, and all 130 regenerated model/stream jobs.

The baseline scalar output oracle must remain independent of extracted range
and packing helpers: otherwise the same refactoring bug can affect both sides.
Keep `-ffp-contract=off`; no fast-math, reciprocal approximation, Newton step,
changed reduction tree or unsolicited AVX-512 extension.

## Measurement and selection

First record the real empty-mask distribution in a separate diagnostic
driver on regenerated streams, including reset/startup and silence-like
fixtures. No counters or instrumentation in timed libraries. Source-audio
silence does not prove that recurrent intermediate rows are empty.

Discovery microbenchmarks compare old helper, refactor-only, R and V using
all 16 masks, identical active rows and balanced randomized fresh processes.
Include batch40/pair40/pair48 and unchanged row64 controls. Retain isolated
helper measurements as diagnostics; final decisions use actual call sites.
Inspect division instructions, branches, text size and stack spills, then
collect hardware counters separately from unprofiled timing. Include branch
misses and frontend/L1I events because earlier sub-percent results were
sensitive to code shape. Test all-active performance explicitly: mixed-row
improvements must not conceal overhead in the common path.

Freeze the selected discovery candidate, binary/PGO hashes and independent
confirmation plan before collecting confirmation data. The relevant final
contrast is **CM + refined quantize4 versus CM-only**, not a comparison that
credits it with CM's already accepted benefit. Old quantize4 + CM is an
additional diagnostic comparator. Check both models, continuous and 10 ms
cadence separately; retain AVX2 as a regression target.

Reserve a physical core and its SMT sibling, keep the sibling idle/offline,
record residual kernel activity, and restore CPU/IRQ/workqueue settings.
Keep A/A controls, fixed sample counts, balanced order, held-out fixtures
and fresh processes. Report individual-call SD separately from run-mean SD,
paired block confidence intervals, P95/P99/max and deadline misses. Never
treat serially dependent calls as independent replicates or remove slow calls.
Set multiplicity correction, A/A precision requirement, minimum useful gain
and noninferiority margins before confirmation. An inconclusive guard is
inconclusive; do not extend the sample until it becomes favorable.

Select only after numerical gates pass, mixed/all-empty regressions are
resolved, and the added full-graph benefit over CM clears those predefined
criteria. Preserve an experiment if only a microbenchmark improves. Validate
other core types/CPUs before making a broader claim.

## Implementation order

1. Freeze the old independent oracle and add complete mixed-mask coverage.
2. Extract shared range/packing helpers; run correctness and inspect codegen.
3. Implement R with one range scan per row and explicit empty outputs.
4. Implement V separately; validate neutral lanes and exact divisions.
5. Measure diagnostic mask frequencies and run controlled discovery.
6. Freeze one candidate and run independent CM-relative confirmation.

Steps 1–6 and the full ready-binary qualification above are complete. The
regenerated 65-fixture stream/PCM corpus, compact-core coverage and red AVX2
portability are no longer pending. Global promotion, qualified model-2/AVX2/
full-audio/Boost-paced benefit, unavailable historical samples and production
device/service-load behavior remain outside the accepted result. The selected
runtime, model weights, ABI, dispatch and service deployment are unchanged.
