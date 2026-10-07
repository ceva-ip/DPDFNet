# Exact graph and convolution experiments, 2026-10-07

These isolated transforms start from the accepted October 3 `best_single`
sources for both `dpdfnet8_48khz_hr` and `dpdfnet2_48khz_hr`. They preserve the
current fitted W7A8 arithmetic and use one calling thread. They do not change
production sources, generated distribution artifacts or existing benchmarks.

The common driver is `../oct7_optimization.py`. Source application, correctness
and timing are separate stages. A successful guarded source transform is not
evidence that a variant is faster or passes runtime validation. Results belong
in the saved per-variant reports and the final investigation report.

The [completed investigation](../../OCT7_OPTIMIZATION.md) retains `block_io`,
`depthwise_deinterleave` and `norm8` in `combo_asm_norm`. Both model sizes passed
6,144 byte-exact block/layout/alias calls and 108 additional boundary shapes
in release and sanitizer builds, plus the complete streaming regression.
The report distinguishes initial screens from the longer cadence results.

| Variant | Work targeted | Numerical and memory contract |
| --- | --- | --- |
| `block_io` | Frequency-major input/output copies inside each DPRNN block | Borrow input through the first residual norm; write the final norm directly into frequency-major output. Channel-major paths retain their transpose scratch. |
| `intra_store` | Copying each 64-float intra-GRU state into the bidirectional result | Write into the correct result slice, then use that slice as the next old state. Retain the positive-zero initial state and initial bias arithmetic. |
| `block_copies` | Both preceding copies | Compose the two independent transforms. Fewer copies can still change cache locality and compiler scheduling. |
| `depthwise_stride` | Scalar stride-2/3 depthwise convolution loops | Gather eight outputs and use separate FP32 multiply/add in original tap order; preserve skipped padding and scalar fringes. |
| `depthwise_deinterleave` | Stride-2 gather overhead | Use bounded contiguous loads and shuffles for stride 2; retain gathers for stride 3 and boundary cases. |
| `norm8` | Ordered FP64 mean/variance dependency chains | Interleave two independent groups of four rows. Each row still reduces channels 0 through 63 in the original order. Preserve the original four-row tail. |

The four encoder depthwise downsamplers are shared by both model sizes. They
use 1x3 kernels, strides 2/3 and widths 480, 160, 80 or 96. The prior AVX2
convolution specialization handles stride 1; these strided paths therefore
offer a distinct optimization opportunity. DPDFNet-2 has fewer DPRNN blocks
and the same 219 generated dense objects and 21 convolution objects, making
this common graph a larger part of its remaining work.

`block_copies` would eliminate approximately 660 KiB of copied float payload
per hop in DPDFNet-8 and 132 KiB in DPDFNet-2. These counts are payload sizes,
not measured latency, RSS savings or CPU traffic. Local transpose buffers
remain on the stack. The direct intra-state variant also changes which
memory supplies the next recurrent input, so correctness does not imply a
performance gain.

## Safety and exactness review

- `block_io` consumes the input residual completely before writing final
  output. Thus `input == output` remains valid for all four layout-flag
  combinations. Temporal `state == state_out` follows the existing gate
  behavior; other buffer overlap remains outside the block API contract.
- A strided vector is allowed only when every tap and lane lies within the
  input row. The upper test implies `(ow+7)*stride+1 <= width-1`; the lower
  test excludes left padding. Depthwise channel multipliers retain the
  original `output_channel / channels_per_group` input mapping.
- The stride-2 contiguous implementation needs `p[0]` through `p[15]`, though
  only even elements through `p[14]` affect output. It explicitly checks the
  extra element and falls back to a gather when that contiguous read would
  leave the input row.
- The original stride-2/3 path uses separate multiply/add. Generated and
  kernel targets retain `-ffp-contract=off`; this specialization contains no
  FMA. The fixed supported shape guards preserve the generic fallback.
- `norm8` retains independent FP64 channel order, mean/variance precision,
  square root and division, FP32 conversion and residual arithmetic. It does
  not reassociate reductions. Supported DPRNN frequencies 40/48 are multiples
  of eight; contract sizes with four remaining rows use the original code.
- FP control modes must be compared with independently advanced states.
  These transforms preserve arithmetic under a fixed caller FP mode; they
  do not promise an identical order of floating-point exception traps.

## Focused validation

`block_copy_oracle.py` compares a live reference against a candidate for both
frequency sizes and scalar/FP32/FP16/W7A8 modes. It covers all layout flags,
both independent in-place buffer choices, unaligned buffers, ordinary/quiet/
high inputs, signed-zero state/features and synthetic signed-zero weights.
Comparison uses raw float bits rather than numerical equality.

The existing layout contract independently covers convolution taps/padding,
channel multipliers and ordered normalization. Useful extra isolated width
cases are **18, 19, 34 and 35**: stride-2 vectors can have their last required
element inside the row while an additional contiguous element lies outside.
Those cases exercise the explicit deinterleave bound and gather fallback.
`stride_boundary_check.py --build <candidate-build> --output <report.json>`
generates those width fixtures from the original oracle in a temporary folder,
compiles with `-ffp-contract=off`, then retains the log and source/library hashes.

For the selected combination, retain release and ASan/UBSan contracts,
scalar fallback, compatibility precision modes, all eight caller rounding/
denormal settings and the full 65-file exact streaming regression. The
quality scores carry forward only when complete PCM hashes match the
fitted reference. Benchmark combinations against that same accepted baseline
after correctness work finishes; individual screening gains must not be added.
