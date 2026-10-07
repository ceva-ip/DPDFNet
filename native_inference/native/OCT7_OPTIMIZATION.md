# Further exact optimization of both 48 kHz models

Investigation: 2026-10-07. The selected profile, `combo_asm_norm`, reduces
standalone 10 ms cadence inference time by **7.67% for DPDFNet-8** and
**4.05% for DPDFNet-2**, against fresh measurements of the accepted
[October 3 `best_single` builds](FURTHER_EXACT_OPTIMIZATION.md).
Every output spectrum, complete recurrent state and PCM sample matches those
builds across all 65 saved audio fixtures per model.

This remains an isolated research profile. It preserves the W7A8 grid, fitted
degree-5 GRU gates, weights, FP32 temporal state, one calling thread, 48 kHz
audio, 480-sample hop and 50 ms algorithmic delay. The assembly component
targets Linux x86-64 SysV; native Windows performance has not been measured.

## Historical times and the fresh matched comparison

The README's previous **1.867 / 0.939 ms** results are valid October 3
measurements. Those same libraries measured **1.963 / 0.958 ms** when rerun
in the new session. Their binary hashes are unchanged. The difference between
sessions is timing variability, not a source or binary regression; its specific
cause was not isolated. CPU clocks, scheduling and Windows/WSL host activity
are not held constant by this benchmark.

| Model | Oct3 build, historical session | Same Oct3 build, rerun Oct7 | New Oct7 profile | Fresh matched reduction |
| --- | ---: | ---: | ---: | ---: |
| `dpdfnet8_48khz_hr` | 1.867 ms | 1.963 ms | **1.813 ms** | **7.67%** |
| `dpdfnet2_48khz_hr` | 0.939 ms | 0.958 ms | **0.919 ms** | **4.05%** |

The reductions use the two columns measured in the new session. Comparing
1.867 directly with 1.813 gives about 2.9%, but mixes sessions and is not the
matched estimate. The same limitation applies to comparisons against older
ONNX, FP16 and INT8 rows. All four standalone candidate run means were below
their corresponding reference run means for each model; raw paired run values
are retained in [oct7_summary.json](../results/oct7_summary.json).

## Directions investigated

The initial priorities were recurrent integer scheduling and GRU dependency
chains, then block memory traffic, strided convolution and ordered
normalization. Variants were first tested separately, then in combinations.
The two model sizes were measured independently because their bottlenecks
differ. There are **27 screened model/build combinations** in this round.

Each screen has three balanced continuous runs of 600 timed hops, 100 warmup
hops, and a separate 1,000-frame byte-exact output/state check. Positive
reduction means less inference time. Small differences are screening signals;
only the selected combination received the longer cadence confirmation.

| Individual direction | DPDFNet-8 reduction | DPDFNet-2 reduction | Decision |
| --- | ---: | ---: | --- |
| Stage reset/update gates before candidate gates | 3.38% | 2.21% | Retain |
| Fully unrolled K=64 AVX2 assembly dot | 3.66% | 1.04% | Retain |
| Borrow/direct block input and output | 0.98% | Not separately screened | Retain in combination |
| Strided depthwise deinterleave | 0.58% | 4.96% | Retain in combination |
| Eight-row ordered FP64 normalization | 0.58% | Not separately screened | Retain in combination |
| Gate loop unroll 1 / 4 / 8 | 1.33% / 0.05% / −0.11% | Not screened | Staged gates were stronger |
| Two-row / 32-output integer tiles | −2.57% | Not screened | Reject |
| Pre-expanded activation broadcasts | 0.14% | Not screened | No convincing benefit |
| Duplicate blocked row weights | 2.35% | Not screened | Reject: adds 1,191,248 owned bytes |
| Repack only sequential recurrent matrices | 1.45% | −3.92% | Reject as common profile |
| Direct intra-GRU state stores | −0.67% | Not screened | Reject |
| Both block copy transforms | 1.08% | Not screened | Keep input/output transform only |
| Stride-2/3 depthwise gathers | 0.25% | Not screened | Prefer deinterleave variant |

Combination screens:

| Profile | Included changes | DPDFNet-8 reduction | DPDFNet-2 reduction |
| --- | --- | ---: | ---: |
| `combo_asm` | Staged gates + assembly | 6.09% | Not screened |
| `combo_asm_io` | Above + block input/output | 6.87% | 6.12% |
| `combo_asm_conv` | Above + depthwise deinterleave | 6.44% | Not screened |
| `combo_asm_norm` | Above + eight-row normalization | **7.29%** | **7.00%** |
| `combo_layout` | Staged gates + recurrent repacking + block input/output + depthwise | 4.06% | 4.89% |

Each percentage uses that screen's own live reference. Baseline drift between
screens prevents reliable rankings from absolute times or adding individual
percentage savings. DPDFNet-2's longer confirmation was weaker than its initial
7.00% screen: **4.05% at standalone cadence**, which is the reported result.
Neighboring combinations were not all confirmed at cadence, so the evidence
does not prove that each retained component contributes independently.

All individual means, p99, maxima, allocation counts, parity and artifact hashes
are available in the `oct7_*_screen.json` reports and consolidated summary.
Several screen maxima increased despite lower means. For example, DPDFNet-8
`combo_asm_io` maximum was 3.524 → 6.570 ms and `combo_asm_norm` was
3.109 → 4.736 ms. Those samples were retained.

## Confirmed latency, CPU and tails

| Model | Execution | Oct3 reference → Oct7 mean | Reduction | p99 reference → Oct7 | Maximum reference → Oct7 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | Continuous | 1.891 → 1.751 ms | 7.44% | 2.498 → 2.335 ms | **3.664 → 4.346 ms** |
| 8 | Paired 10 ms cadence | 2.011 → 1.889 ms | 6.04% | 2.849 → 2.679 ms | 5.061 → 4.526 ms |
| 8 | Standalone 10 ms cadence | **1.963 → 1.813 ms** | **7.67%** | **2.463 → 2.134 ms** | **3.834 → 3.027 ms** |
| 2 | Continuous | 0.823 → 0.794 ms | 3.51% | 1.105 → 1.043 ms | 2.333 → 1.581 ms |
| 2 | Paired 10 ms cadence | 0.954 → 0.929 ms | 2.69% | 1.184 → 1.136 ms | 1.712 → 1.688 ms |
| 2 | Standalone 10 ms cadence | **0.958 → 0.919 ms** | **4.05%** | **1.305 → 1.153 ms** | **1.910 → 1.549 ms** |

Standalone whole-process CPU per hop fell from **1.973 → 1.822 ms** for model 8
and **0.966 → 0.927 ms** for model 2. There are 12,000 timed calls per
implementation per model: four 1,000-hop runs in each execution mode, with
100 warmup hops per run. Means and p99 are medians of four run statistics;
maximum is the largest observed call across those runs. Samples are unfiltered.

Every inference call was below 10 ms. All standalone cadence calls completed
before their next scheduled release. Model 2 also had zero paired late
completions. Model 8's candidate had **one paired late completion**: inference
took 4.526 ms, process CPU was 4.550 ms, and completion was 10.140 ms after
scheduled release. That measurement includes wake delay, bookkeeping and the
other implementation when it runs first; the inference duration alone does
not account for the entire completion time. It does not identify a unique
cause. The reference had no paired late completions.

The lower standalone tails are useful observations, not a worst-case guarantee.
The continuous model 8 maximum increased. OS/host interruptions, power states
and competing activity still require measurement in the consumer audio pipeline.
The [prior tail investigation](TAIL_LATENCY_INVESTIGATION.md) describes these
limits and the allocating versus preallocated calling paths.

The machine is an Intel i7-8700, GCC 12.2, Linux Docker/WSL2, with unrestricted
affinity. Calls use preallocated caller buffers, cached native pointers and
in-place state. Paired order reverses each hop; standalone order alternates
between runs. No build, correctness or memory jobs overlap these benchmarks.
FFT, synthesis, resampling, Python bookkeeping outside the C call and audio
device I/O are excluded. Cross-model ratios come from separate sessions.

## What the selected profile changes

1. **Stage GRU gates.** Calculate 64 reset and 64 update values first, then
   candidates and next state. Two temporary float arrays expose independent
   work to the processor. Formulas, coefficients and each element's FP/FMA
   operations remain unchanged.
2. **Specialize the K=64 row dot in assembly.** Fully unroll 16 input groups,
   maintain eight integer accumulator vectors and schedule four independent
   product temporaries. Fold weight loads into `vpmaddubsw`. The existing C
   quantizer, zero-point correction, scaling and bias FMA remain unchanged.
   Other K sizes and unsupported platforms use the original intrinsic path.
3. **Remove block input/output copies.** Borrow frequency-major input until
   its final residual consumer and normalize directly into frequency-major
   output. Channel-major layouts retain transpose scratch. Writes begin after
   the input is consumed, including when input/output alias. State aliasing
   remains supported. The slower direct intra-state-store change is excluded.
4. **Vectorize strided depthwise convolution.** Specialize depthwise 1x3
   stride-2/3 downsamplers with eight-output vectors. Stride 2 uses contiguous
   loads and deinterleaving; stride 3 and boundaries use gathers. Keep tap order,
   separate FP32 multiply/add, skipped padding and scalar fringes. An explicit
   bound checks the extra unused element loaded by the deinterleave path.
5. **Interleave eight normalization rows.** Overlap two independent four-row
   FP64 reduction chains. Each row still sums channels 0 through 63 in the
   original order. Variance, sqrt/division, conversions and residual arithmetic
   are unchanged. Existing four-row tails remain available.

There is no new activation approximation, precision mode, hop batching, worker
thread or retraining. Compiler contraction remains disabled except explicit
existing intrinsics. The assembly is private and uses the existing AVX2/FMA
capability dispatch; no AVX-512/VNNI requirement is added.

## Memory

| Model | Warmed incremental RSS, Oct3 → Oct7 | Native owned bytes, Oct3 → Oct7 |
| --- | ---: | ---: |
| 8 | 7.525 → 7.531 MiB | **6,763,588 → 6,763,588** |
| 2 | 6.135 → 6.078 MiB | **5,424,496 → 5,424,496** |

Four fresh processes per implementation warm 120 hops and unmap source weights
before sampling. RSS subtracts the common imported-runtime baseline and includes
library pages, stream buffers and allocator retention. It is distinct from
owned model allocations or total application RAM. These small RSS differences
do not establish memory savings; retained model allocations are unchanged.
The older README RSS numbers remain dated historical observations.

The blocked duplicate-weight alternative adds about 1.14 MiB for model 8 despite
a modest screen gain. Repacking only the sequential recurrent matrices avoids
duplicate retained weights but adds 2,776 / 2,008 bookkeeping bytes, incurs
creation-time temporary storage and performs worse on model 2. Neither was
selected. Staged gates add transient stack arrays, not model heap allocations.

## Output quality and safety evidence

Each model passed the entire saved **65-file / 1,339.149-second / 134,332-hop**
suite: 50 EARS-WHAM mixtures, 10 mixtures with speech at −50 dBFS RMS, 2 clean
controls and 3 independent 120-second noise streams. Every spectrum, complete
state and aligned FP32 PCM sample matched the live Oct3 reference byte for
byte, including signed zero. All values were finite and reset replay matched.

The summary additionally checks **all 65 spectrum/state stream hashes and PCM
hashes against the saved Oct3 candidate**, with identical model, weight and
input hashes. That earlier evaluation links the fitted profile to the scored
waveforms. Consequently existing PESQ, STOI, SI-SNR and SIGMOS scores carry
forward unchanged. Scorers were not rerun, and the scored comparison remains
the same six-mixture subset; 65 exact-output files do not enlarge it or establish
equivalence to ONNX/FP16/INT8. Existing quiet-speech behavior is preserved too.

Additional selected-profile checks passed for **both** model sizes:

- Release and ASan/UBSan: five C contracts each; scalar-only: four each.
- Three precision compatibility configurations: 1,000 recurrent frames each,
  plus four independent contexts matching serial and concurrent replay.
- 6,144 byte-exact block calls per model across F=40/F=48, four precision modes,
  all layouts, input/output and state aliases, unaligned buffers, signed-zero
  weights/features/states and quiet/high-level inputs.
- 108 additional convolution boundary shapes per model in release and sanitizer
  builds, including widths 18, 19, 34, 35. The original independent scalar oracle
  and transpose/normalization checks remain intact.
- All four rounding modes with denormal handling off/on after creation:
  64 recurrent frames per setting, exact finite output/state, in-place spectrum
  and state. One observed OS thread throughout creation/process/destruction.
  Floating-point exception flags are not compared.
- A separate 1,000-hop quiet/high-level/silence stream, reset replay and
  independent-context concurrency. Correctness workers own separate models;
  no inference worker threads are added to an individual model.

**Assembly requires separate bounds evidence:** ASan does not instrument
the handwritten `.S` memory operations. A direct scalar int32 oracle passed
**795 calls / 50,880 integer outputs** with extreme, zero and random W7A8
patterns. Guard pages immediately before/after payloads, read-only inputs,
unaligned buffers and output canaries were checked. The tested assembly source
hash is identical for both selected model builds.

Static bounds: reads 64 activation bytes and 4,096 packed weight bytes, writes
256 output bytes. The final accesses end at offsets 63, 4095, 255 respectively.
The byte-pair sum bound is `2*254*63=32004`, so signed 16-bit saturation cannot
change the result; the K=64 full dot fits int32. SysV caller-saved RAX/YMM0–13
only, unchanged stack/callee-saved registers, CFI, `vzeroupper`, position
independent code and a non-executable stack declaration are retained.
These checks do not certify another ABI, CPU or worst-case latency.

## Artifacts and reproduction

Selected libraries and SHA256:

| Model | Build | SHA256 |
| --- | --- | --- |
| 8 | `build/oct7_8_combo_asm_norm/libdpdf_full.so` | `5398d9cb8d3412cee7e8e6e5b5bd4b941bd7bf4ebd59ff53564da31786dccdb5` |
| 2 | `build/oct7_2_combo_asm_norm/libdpdf_full.so` | `c26483de98b13e3468a961d9f7702ea587e8ae3520851fc6c9f2e5c654412be9` |

All reports, source/binary identities, historical-output links and safety logs
are indexed by [oct7_summary.json](../results/oct7_summary.json).
Guarded transforms live under [experiments/](experiments/oct7_optimization.py).
Frozen generated sources are under `scratch/oct7/{8,2}/combo_asm_norm/`;
builds/scratch and downloaded audio remain local generated artifacts.

From `native_inference/` in the same offline Linux development environment,
with the saved models, block weights and evaluation fixtures:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python native/experiments/further_optimization.py --model-size 8 build baseline best_single
python native/experiments/further_optimization.py --model-size 2 build baseline best_single
for size in 8 2; do
    python native/experiments/oct7_optimization.py build combo_asm_norm --size "$size"
    python native/experiments/oct7_optimization.py asan combo_asm_norm --size "$size"
    python native/experiments/oct7_optimization.py scalar combo_asm_norm --size "$size"
done
python native/experiments/oct7_quant/run_asm64_contract.py
```

Run the precision, alias, boundary and FP-control checks before timing:

```sh
for size in 8 2; do
    if [ "$size" = 8 ]; then
        oct7_baseline=build/further_best_single
    else
        oct7_baseline=build/further2_best_single
    fi
    oct7_candidate=build/oct7_${size}_combo_asm_norm
    python native/latency_validation.py --model models/dpdfnet${size}_48khz_hr.onnx --weights models/rework${size}/weights.f32 --baseline-build "$oct7_baseline" --candidate-build "$oct7_candidate" --frames 1000 --output results/oct7_${size}_compatibility.json
    python native/experiments/further_fp_environment.py --baseline-build "$oct7_baseline" --candidate-build "$oct7_candidate" --weights models/rework${size}/weights.f32 --frames 64 --verify-single-thread --output results/oct7_${size}_fp_environment.json
    python native/experiments/oct7_graph/block_copy_oracle.py --baseline "$oct7_baseline/libdpdf_dprnn.so" --candidate "$oct7_candidate/libdpdf_dprnn.so" --weights models/native_blocks/df_0.f32 models/native_blocks/erb_0.f32 --output results/oct7_${size}_block_alias.json
    python native/experiments/oct7_graph/stride_boundary_check.py --build "$oct7_candidate" --output results/oct7_${size}_stride_boundary.json
    python native/experiments/oct7_graph/stride_boundary_check.py --build "${oct7_candidate}_asan" --sanitize --output results/oct7_${size}_stride_boundary_asan.json
done
```

The [graph experiment README](experiments/oct7_graph/README.md) and
[integer experiment README](experiments/oct7_quant/README.md) explain the focused
checks. Then run the audio regressions:

```sh
python native/experiments/further_exact_validation.py --model /bench/models/dpdfnet8_48khz_hr.onnx --weights /bench/models/rework8/weights.f32 --baseline-build /bench/build/further_best_single --candidate-build /bench/build/oct7_8_combo_asm_norm --workers 4 --output results/oct7_8_combo_asm_norm_audio.json
python native/experiments/further_exact_validation.py --model /bench/models/dpdfnet2_48khz_hr.onnx --weights /bench/models/rework2/weights.f32 --baseline-build /bench/build/further2_best_single --candidate-build /bench/build/oct7_2_combo_asm_norm --model-quality-report /bench/results/dpdfnet2_overview_quality.json --workers 4 --output results/oct7_2_combo_asm_norm_audio.json
```

After all build/correctness jobs finish, measure sequentially:

```sh
for size in 8 2; do
    python native/experiments/oct7_optimization.py final combo_asm_norm --size "$size" --frames 1000 --repeats 4
    python native/experiments/oct7_memory.py --size "$size"
done
python native/experiments/oct7_summary.py
```

The RSS driver refuses to overwrite an existing report; use `--output` for
new runs or explicitly `--overwrite`. Timing reports are per profile and
phase; preserve existing records before repeating them. The consolidated
summary expects all named compatibility/FP-control/block/stride reports too.

## Next directions, in priority order

1. **Port and measure in the native Windows/HushMic pipeline.** The four shared
   C transforms retain platform fallbacks, but the assembly uses SysV. A
   Windows version needs its calling convention, vector-register preservation
   and unwind/build support. Real audio load and end-to-end completion tails
   are more informative than another small isolated Linux gain.
2. **Fuse the sequential K=64 dot epilogue.** The present assembly writes an
   integer scratch tile that C reloads for correction/dequantization. Keeping
   that epilogue in registers may reduce stores and call overhead. Preserve
   quantization, rounding and explicit FMA behavior, then repeat the same
   independent integer, FP-control and streaming checks. This is unimplemented.
3. **Reduce remaining scalar nonlinear work.** Vector libm or another fitted
   activation for the non-DPRNN paths could matter proportionately more to
   model 2. Any new approximation needs fresh perceptual scoring, quiet-speech
   and long-noise tests, rather than exact-output score carry-forward.
4. **Explore structured sparsity/pruning or narrower quantization with
   retraining.** These could exceed scheduling gains but change the model and
   require representative quality evaluation. Skipping/batching hops changes
   streaming behavior, and extra inference threads conflict with the current
   integration constraint; neither is included in the selected profile.
