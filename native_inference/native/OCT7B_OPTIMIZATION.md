# Further optimization against the committed Oct7 profile

Investigation: 2026-10-07. Both selected profiles retain byte-identical output and complete recurrent state across the saved 65-file suite per model. They reduce retained model allocations by **1,440,192 bytes each**. These are isolated research builds; consumer integration presets are unchanged.

The reference is the committed `combo_asm_norm` profile from [the preceding investigation](OCT7_OPTIMIZATION.md). Its historical **1.813 / 0.919 ms** standalone means are dated observations. New reductions below use fresh matched reference and candidate calls in this session. Cross-session subtraction does not estimate the measured benefit.

| Model | Historical Oct7 reference | Same reference, rerun | Selected new profile | Matched reduction |
| --- | ---: | ---: | ---: | ---: |
| DPDFNet-8 | 1.813 ms | 1.907 ms | **1.776 ms** | **6.87%** |
| DPDFNet-2 | 0.919 ms | 0.942 ms | **0.878 ms** | **6.82%** |

## Selected changes

- DPDFNet-8: `combo_exact8` — `compiler_pgo`, `quant_fused192_init_inline`, `dense_int8_pad8`, `graph_gru_fuse`, `conv_row_pair`, `quant_pair64`.
- DPDFNet-2: `combo_wide2` — `quant_fused192`, `dense_int8_pad8`, `graph_gru_fuse`, `graph_bias_relu`, `conv_row_pair`, `quant_fused256`, `quant_pair64`.

The integer row kernels fuse correction, conversion, the separately rounded FP32 scale product and the existing bias FMA. Fixed paired kernels reuse four activation broadcasts for two output tiles. Eight-column INT8 dense padding removes unneeded projections and output scratch. Exact generated-GRU fusion keeps scalar libm gates and original arithmetic; exact convolution helpers preserve tap and multiply/add order. PGO, where selected, is trained on procedural spectral streams in a separate process and adds no inference thread.

DPDFNet-8 selects PGO, correction-first small-projection fusion and the inlined K64 quantizer. DPDFNet-2 selects ordinary small-projection correction order, bias/ReLU fusion and the wider K256 external-GRU projection kernel. The saved combination screens document those model-specific choices; the final confirmation validates each selected composition as a whole.

Different component choices reflect independent screens for the two model sizes. Individual reductions cannot be added: fused kernels, compiler decisions, instruction footprint and memory traffic interact. Neighboring combinations have their own live reference runs; screens do not prove the globally optimal composition or independent value of every retained component. No new approximation is selected.

## Confirmed timing and tails

| Model | Execution | Reference → candidate mean | Reduction | p99 reference → candidate | Maximum reference → candidate |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 | Continuous | 1.762 → 1.666 ms | 5.45% | 1.998 → 1.885 ms | 3.086 → 2.777 ms |
| 8 | Paired 10 ms cadence | 1.921 → 1.781 ms | 7.29% | 2.162 → 2.024 ms | 4.743 → 2.486 ms |
| 8 | Standalone 10 ms cadence | 1.907 → 1.776 ms | 6.87% | 2.189 → 2.015 ms | 3.249 → 2.818 ms |
| 2 | Continuous | 0.795 → 0.748 ms | 5.85% | 1.059 → 0.987 ms | 1.584 → 1.365 ms |
| 2 | Paired 10 ms cadence | 0.961 → 0.882 ms | 8.19% | 1.170 → 1.103 ms | 1.670 → 1.553 ms |
| 2 | Standalone 10 ms cadence | 0.942 → 0.878 ms | 6.82% | 1.157 → 1.135 ms | 1.696 → 2.177 ms |

Four 1,000-hop runs in each mode, with 100 warmup hops per run, produce 12,000 timed calls per implementation per model. Means and p99 are medians of run statistics; maxima are largest observed calls. No samples, outliers or scheduler stalls are filtered from the statistical aggregation. Saved JSON retains every per-run summary, context-switch count and late-call event; ordinary per-hop sample arrays are not stored. Paired order reverses each hop, standalone implementation order reverses between repeats.

| Model | Standalone process CPU, reference → candidate | Calls above 10 ms, reference / candidate | Paired late completions, reference / candidate | Standalone late completions, reference / candidate |
| --- | ---: | ---: | ---: | ---: |
| 8 | 1.916 → 1.785 ms | 0 / 0 | 0 / 0 | 0 / 0 |
| 2 | 0.950 → 0.886 ms | 0 / 0 | 0 / 0 | 0 / 0 |

Observed p99/maxima and deadline counts are measurements, not a hard worst-case guarantee. Completion after scheduled release also includes wake delay and, in paired mode, the other implementation. OS scheduling, CPU power states and host activity remain variable. No build, validation, scoring or memory jobs overlapped latency runs. FFT, synthesis, resampling, Python work outside the C call and audio device I/O are excluded. Windows/HushMic pipeline performance remains unmeasured. The environment is Intel i7-8700, GCC 12.2, Linux Docker/WSL2, one calling thread and unrestricted CPU affinity.

A separate saved real EARS-WHAM mixture (clip00033) confirmation used four balanced 1,000-hop continuous runs per implementation and 100 warmup hops. FFT preparation stayed outside the timed call.
DPDFNet-8: 1.753 → 1.682 ms (4.05% reduction), p99 2.160 → 1.979 ms, maximum 2.971 → 4.685 ms, with exact spectrum/state. [oct7b_8_speech_timing.json](../results/oct7b_8_speech_timing.json).
DPDFNet-2: 0.775 → 0.732 ms (5.47% reduction), p99 0.958 → 0.896 ms, maximum 2.656 → 1.422 ms, with exact spectrum/state. [oct7b_2_speech_timing.json](../results/oct7b_2_speech_timing.json).
The DPDFNet-8 real-clip maximum increased despite lower mean/p99. This single-clip confirmation does not establish uniformly better peak latency or replace the broader cadence tests.

## Memory

| Model | Native owned bytes, reference → candidate | Reduction | Warmed incremental RSS, reference → candidate |
| --- | ---: | ---: | ---: |
| 8 | 6,763,588 → **5,323,396** | **21.29%** | 7.564 → 6.479 MiB |
| 2 | 5,424,496 → **3,984,304** | **26.55%** | 6.107 → 4.791 MiB |

RSS uses four balanced fresh processes per implementation, 120 warmup hops, unmapped source weights and subtraction of the common imported-runtime baseline. It includes library pages, buffers and allocator retention, and differs from owned model allocations and total application RAM. The owned reduction comes from dense padding and scratch removal; it is exactly 1,440,192 bytes for each model.

## Screened directions

There are **50 model/build screens** in this round. The first table contains all 28 initial model/direction screens (14 directions × two model sizes). Each has a separate 1,000-hop correctness check and three balanced 600-hop continuous timing runs. Positive percentages mean lower inference time. Approximate rows measure numerical drift and require fresh quality evidence.

| Direction | DPDFNet-8 reduction | DPDFNet-2 reduction | Arithmetic |
| --- | ---: | ---: | --- |
| Generated GRU with approximate AVX2 gates (`graph_gru_approx`) | +1.60% | +5.89% | Approximate; not accepted |
| Fuse complete K64/N192 row projection (`quant_fused192`) | +3.41% | +3.78% | Exact on 1,000 tested hops |
| Eight-column INT8 dense padding (`dense_int8_pad8`) | +1.80% | +5.24% | Exact on 1,000 tested hops |
| Synthetic-trained PGO (`compiler_pgo`) | +3.51% | +2.24% | Exact on 1,000 tested hops |
| Fuse generated GRU with exact scalar libm gates (`graph_gru_fuse`) | +1.41% | +3.32% | Exact on 1,000 tested hops |
| Approximate generated Sigmoid/Tanh (`graph_activations_approx`) | +1.64% | +2.88% | Approximate; not accepted |
| Fuse bias and ReLU loops (`graph_bias_relu`) | -0.20% | +4.27% | Exact on 1,000 tested hops |
| Above, initialize integer correction before dot (`quant_fused192_init`) | +2.05% | +1.68% | Exact on 1,000 tested hops |
| IPO + disable semantic interposition (`compiler_ipo_nointerpose`) | +2.20% | +1.11% | Exact on 1,000 tested hops |
| Correction-first N192 fusion + inline quantizer (`quant_fused192_init_inline`) | +3.73% | -1.90% | Exact on 1,000 tested hops |
| Inline specialized K64 quantizer (`quant_inline64`) | +2.33% | -1.82% | Exact on 1,000 tested hops |
| Fuse K64 integer dot and FP32 epilogue (`quant_fused64`) | +2.02% | -1.98% | Exact on 1,000 tested hops |
| Align selected hot functions/loops (`compiler_hot_align`) | +1.50% | -2.11% | Exact on 1,000 tested hops |
| PGO + IPO + disable interposition (`compiler_pgo_ipo_nointerpose`) | -0.05% | -0.75% | Exact on 1,000 tested hops |

Additional focused screens follow. A blank experiment can still be present in a selected combination without an individual measurement; no separate gain is inferred.

| Direction | DPDFNet-8 reduction | DPDFNet-2 reduction |
| --- | ---: | ---: |
| Paired output rows in exact convolution (`conv_row_pair`) | +2.32% | +3.82% |
| Fuse convolution transpose/ReLU (`conv_transpose_relu`) | -1.18% | -0.35% |
| Fixed K64 four-row/two-tile assembly (`quant_pair64`) | +1.24% | +2.95% |
| Fixed K64/K96 paired assembly (`quant_pair64_96`) | +0.59% | -2.61% |
| Fuse external-GRU K256/N768 projection (`quant_fused256`) | +0.55% | +3.10% |

| Combination | Components | DPDFNet-8 reduction | DPDFNet-2 reduction |
| --- | --- | ---: | ---: |
| `combo_approx2` | quant_fused192, dense_int8_pad8, graph_gru_approx, graph_bias_relu, conv_row_pair, quant_pair64 | Not separately screened | +5.60% |
| `combo_core` | quant_fused192_init_inline, dense_int8_pad8, graph_gru_fuse | +2.49% | Not separately screened |
| `combo_core192` | quant_fused192, dense_int8_pad8, graph_gru_fuse | Not separately screened | +4.69% |
| `combo_core192_bias` | quant_fused192, dense_int8_pad8, graph_gru_fuse, graph_bias_relu | Not separately screened | +5.19% |
| `combo_core192_pgo` | compiler_pgo, quant_fused192, dense_int8_pad8, graph_gru_fuse, graph_bias_relu | Not separately screened | +2.93% |
| `combo_core_ipo` | compiler_ipo_nointerpose, quant_fused192_init_inline, dense_int8_pad8, graph_gru_fuse | +2.96% | Not separately screened |
| `combo_core_pgo` | compiler_pgo, quant_fused192_init_inline, dense_int8_pad8, graph_gru_fuse | +5.28% | Not separately screened |
| `combo_exact2` | quant_fused192, dense_int8_pad8, graph_gru_fuse, graph_bias_relu, conv_row_pair, quant_pair64 | Not separately screened | +5.28% |
| `combo_exact8` | compiler_pgo, quant_fused192_init_inline, dense_int8_pad8, graph_gru_fuse, conv_row_pair, quant_pair64 | +6.28% | Not separately screened |
| `combo_min2` | quant_fused192, dense_int8_pad8, conv_row_pair, quant_pair64 | Not separately screened | +0.61% |
| `combo_wide2` | quant_fused192, dense_int8_pad8, graph_gru_fuse, graph_bias_relu, conv_row_pair, quant_fused256, quant_pair64 | Not separately screened | +7.34% |
| `combo_wide8` | compiler_pgo, quant_fused192_init_inline, dense_int8_pad8, graph_gru_fuse, conv_row_pair, quant_fused256, quant_pair64 | +2.24% | Not separately screened |

Full per-run statistics, parity, memory counts, source manifests and library hashes for every screen are retained in [oct7b_summary.json](../results/oct7b_summary.json). Percentages within this table use each screen’s own contemporaneous reference; absolute times across separate screens do not rank close variants.

## Quality and exactness

DPDFNet-8 passed **65 files / 1339.149 seconds / 134,332 hops**. Every spectrum, entire recurrent state and aligned FP32 PCM sample matched the live Oct7 reference byte for byte, including signed zero. All 65 stream/PCM hashes also match the saved Oct7 candidate with identical input, model and weight hashes. [oct7b_8_combo_exact8_audio.json](../results/oct7b_8_combo_exact8_audio.json).

DPDFNet-2 passed **65 files / 1339.149 seconds / 134,332 hops**. Every spectrum, entire recurrent state and aligned FP32 PCM sample matched the live Oct7 reference byte for byte, including signed zero. All 65 stream/PCM hashes also match the saved Oct7 candidate with identical input, model and weight hashes. [oct7b_2_combo_wide2_audio.json](../results/oct7b_2_combo_wide2_audio.json).

The 65 files include 50 EARS-WHAM mixtures, ten mixtures with speech at −50 dBFS RMS, two clean controls and three independent 120-second pure-noise streams. Existing PESQ/STOI/SI-SNR/SIGMOS scores therefore carry forward unchanged on those scored waveforms. The quality comparison remains the existing six-mixture subset; exactness over 65 files does not enlarge its perceptual scoring coverage or prove universal equivalence to ONNX/FP16/INT8.

The separate `graph_gru_approx` experiment replaces scalar generated GRU gates with fitted AVX2 gates and changes output. It is **not accepted**. These six-mixture candidate-minus-Oct7-reference deltas are exploratory; full 50-mixture scoring, quiet/clean/noise controls and listening evidence are still required before adoption.

| Metric | DPDFNet-8 paired mean delta | DPDFNet-2 paired mean delta |
| --- | ---: | ---: |
| PESQ-WB (16 kHz resampling) | +0.0005330 | -0.0001825 |
| STOI | +0.0000015 | +0.0000167 |
| SI-SNR (48 kHz, dB) | +0.0004071 | -0.0001180 |
| SIGMOS coloration | -0.0023652 | +0.0008049 |
| SIGMOS discontinuity | -0.0005597 | +0.0079075 |
| SIGMOS loudness | -0.0039846 | +0.0042746 |
| SIGMOS noise | -0.0043834 | -0.0000024 |
| SIGMOS reverberation | -0.0015111 | -0.0016162 |
| SIGMOS signal | -0.0009819 | -0.0037450 |
| SIGMOS overall | -0.0009865 | -0.0016476 |

SIGMOS and SI-SNR use native 48 kHz PCM. PESQ-WB explicitly resamples to 16 kHz; it is not a 48 kHz PESQ metric. Standard STOI accepts the 48 kHz input and internally resamples to 10 kHz. All seven SIGMOS dimensions, per-clip deltas, paired extrema and waveform differences are retained in [oct7b_8_graph_gru_approx_quality.json](../results/oct7b_8_graph_gru_approx_quality.json), [oct7b_2_graph_gru_approx_quality.json](../results/oct7b_2_graph_gru_approx_quality.json).

## Safety evidence and artifact identities

| Model | Release contracts | ASan/UBSan contracts | Scalar-only contracts |
| --- | ---: | ---: | ---: |
| 8 | 9 | 9 | 6 |
| 2 | 10 | 10 | 6 |

Both selected binaries passed four precision compatibility configurations with 1,000 recurrent hops each, independent-context serial/concurrent replay, and all four rounding modes with denormal handling off/on after model creation. One observed OS thread was maintained during creation/process/destruction. Exception flags are not compared. Correctness workers own independent streams; they add no worker to one model’s inference.

New fused/paired/wide assembly, when present, has a dedicated direct scalar-oracle CTest with protected mappings, read-only inputs, unaligned buffers and canaries. ASan does not instrument handwritten assembly memory accesses; those direct checks and documented static bounds are separate evidence. Sanitizer/scalar PGO variants use plain builds; release-binary exactness is checked separately. Detailed test names, logs/hashes, source manifests, training/counter identities and supplemental oracles are retained in the JSON summary.

| Model | Library | SHA256 |
| --- | --- | --- |
| 8 | baseline | `5398d9cb8d3412cee7e8e6e5b5bd4b941bd7bf4ebd59ff53564da31786dccdb5` |
| 8 | candidate | `bdfc013b5c42c0a6f454d8c84465a616bb8be910b136423f366d50d018f7986b` |
| 2 | baseline | `c26483de98b13e3468a961d9f7702ea587e8ae3520851fc6c9f2e5c654412be9` |
| 2 | candidate | `03fd96a37ade162fceeb70743204ca8e3b5c4d45a3f86851f6bf359e2c6c682b` |

## Reproduce offline

Use the existing `dpdfnet-native-dev` image, cached models/weights and saved audio fixtures. Preserve `scratch/oct7/{8|2}/combo_asm_norm` and `build/oct7_{8|2}_combo_asm_norm`: these are the reference sources and binary identities. All new builds use isolated snapshots. No download, tool installation or network access is required.

The commands below describe a full replay in a fresh, isolated `native_inference` workspace. First regenerate or preserve the Oct7 reference using the preceding report, with identical source and library hashes. The summary also requires the retained screen and approximate-quality evidence from this investigation. Existing frozen binaries and reports can instead be verified by the summary driver without rebuilding or retraining.

From `native_inference/` in PowerShell, define an offline runner:

```powershell
function Invoke-Native([string]$Command) {
  docker run --rm --network none -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 `
    --mount "type=bind,source=${PWD},target=/bench" --workdir /bench `
    --entrypoint sh dpdfnet-native-dev -c $Command
  if ($LASTEXITCODE -ne 0) { throw "Native stage failed: $Command" }
}
```

On a fresh output set, invoke stages sequentially. Do not repeat the release PGO build against the retained counters:

```powershell
Invoke-Native 'python native/experiments/oct7b_optimization.py build combo_exact8 --size 8'
Invoke-Native 'python native/experiments/oct7b_optimization.py asan combo_exact8 --size 8'
Invoke-Native 'python native/experiments/oct7b_optimization.py scalar combo_exact8 --size 8'
Invoke-Native 'python native/experiments/oct7b_optimization.py final combo_exact8 --size 8 --frames 1000 --repeats 4'
Invoke-Native 'python native/experiments/oct7b_speech_timing.py combo_exact8 --size 8'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_exact8 --size 8 --stage contracts'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_exact8 --size 8 --stage compatibility'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_exact8 --size 8 --stage fp'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_exact8 --size 8 --stage dense'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_exact8 --size 8 --stage audio'
Invoke-Native 'python native/experiments/oct7b_memory.py combo_exact8 --size 8'
Invoke-Native 'python native/experiments/oct7b_optimization.py build combo_wide2 --size 2'
Invoke-Native 'python native/experiments/oct7b_optimization.py asan combo_wide2 --size 2'
Invoke-Native 'python native/experiments/oct7b_optimization.py scalar combo_wide2 --size 2'
Invoke-Native 'python native/experiments/oct7b_optimization.py final combo_wide2 --size 2 --frames 1000 --repeats 4'
Invoke-Native 'python native/experiments/oct7b_speech_timing.py combo_wide2 --size 2'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_wide2 --size 2 --stage contracts'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_wide2 --size 2 --stage compatibility'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_wide2 --size 2 --stage fp'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_wide2 --size 2 --stage dense'
Invoke-Native 'python native/experiments/oct7b_validate.py combo_wide2 --size 2 --stage audio'
Invoke-Native 'python native/experiments/oct7b_memory.py combo_wide2 --size 2'
Invoke-Native 'python native/experiments/oct7b_summary.py --selected8 combo_exact8 --selected2 combo_wide2'
```

Build/validation/scoring/RSS stages must not overlap latency. PGO builds are immutable after training: repeating a release build against retained counters is rejected. The CLI has no variant alias or alternate output-root option; independent replay needs a fresh isolated workspace. Assembly targets Linux x86-64 SysV with AVX2/FMA; fallback sources are preserved elsewhere.
