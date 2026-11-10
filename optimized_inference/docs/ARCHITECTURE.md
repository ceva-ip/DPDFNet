# Architecture and maintenance

The public runtime selects one of two fixed graph backends. Each backend is
compiled from shared kernels, a model-specific graph, and its final assembly
specializations. Backend symbols are isolated at compile time; neither graph
can accidentally bind to the other model's kernels.

## Numerical implementation

Quantized matrices use signed **7-bit weights** and dynamically scaled **8-bit
activations**, with integer dot products and FP32 scale/bias application.
The weight restriction prevents saturation in the AVX2 packed multiply path.
DPRNN, dense/grouped projections and eligible 1x1 convolutions use this path.
Other convolutions, normalization, complex filtering and recurrent state
remain FP32. DPRNN gates use the validated fitted degree-5 approximation.

Adjacent DPRNN blocks retain frequency-major features, avoiding repeated
transposes. Matrix tiles use packed 32-byte operands; kernels specialize the
fixed graph's common 64-input projections. Model 2 also uses fused paired
projections, a 256-input assembly projection, and AVX2 arithmetic surrounding
unchanged scalar libm calls in its generated GRU cells. Model 8 uses its
accepted fixed-row kernels and compiler profile.

The K64N64 channel-major pointwise path supports both AVX2 and AVX-VNNI.
It reads and writes frequency positions directly, avoiding the eight eligible
projections' input/output float transposes in each graph. The wrapper reuses
the matrix's existing affine implementation selection; dimensions, bounded
row counts and non-overlapping buffers remain explicit eligibility checks.
The shared quantization and 4x8 output stores preserve floating-point order.
VNNI dot products use VEX-encoded instructions and eight resident integer
accumulators, without requiring AVX-512 or adding per-context state.

Model-specific choices in shared C sources are selected by the private
`DPDF_MODEL_SIZE` compile definition. These branches preserve the accepted
operation order of each graph. Do not replace explicit operations with FMA
or change quantization rounding without new numerical and audio evidence.

## Source boundaries

- `runtime.c`, `backend.h`, and `model_backend.c` implement the public API and
  select the fixed W7A8 configuration.
- `common/int8.c`, `avx2.c`, and assembly perform quantization, projection,
  activation and normalization operations.
- `common/dpdf_dprnn.c` manages the fixed DPRNN blocks; `extended_ops.c` manages
  dense/grouped/convolution storage and graph helpers.
- `src/models/*/generated_model.c` contains explicit fixed graph operations,
  weight offsets, state layout, and normalization seeds. These files belong to
  the supplied model/weight hashes; they are not a general ONNX interpreter.
- `private/` holds kernel declarations and namespace mappings. These headers
  are not installed and do not define a consumer ABI.

Private scalar and precision helpers remain as numerical oracles/fallback
building blocks for contract tests. The public API exposes only the final
optimized configuration; there are no experimental runtime presets or
alternative model packages.

## Validation expectations

Run release and sanitizer contracts after kernel changes. Preserve guarded
assembly tests: sanitizers cannot instrument the `.S` files. Use the saved
stream hashes to compare every spectrum, full state history and synthesized
PCM across the 65 fixtures, rather than checking only the last frame.

The fixed graphs are paired with `models/*/manifest.json`. A changed model or
weight layout requires new graph generation, manifests and reference outputs.
Source reorganization, symbol changes and new compiler flags require fresh
model-8 PGO training and a latency check on the target machine.

Keep benchmark jobs isolated from builds and correctness tests. Use balanced
process order, equal input/warmup/cadence, and same-session comparisons for
speedup claims. Per-hop samples are not independent experiments. An observed
maximum is a finite measurement, not a worst-case execution-time bound.

### Quantization and channel-major experiments

Treat row-major four-row quantization and channel-major K64N64 execution as
separate factors. They partly overlap: accepting the channel-major path bypasses
pointwise row-major projections whose quantization could otherwise benefit from
the four-row helper. Its channel-major quantizer already works across eight
frequency positions. Adding separately measured percentages is therefore invalid.

The selected legacy four-row helper vectorizes scale divisions without
reciprocal approximations. Its degenerate-row fallback recomputes all four rows;
zero-row microbenchmarks must accompany ordinary inputs when evaluating wider
VNNI adoption. Any refinement should reuse the exact range/packing logic,
preserve FP order and empty-row behavior, and repeat the numerical and ISA gates.
The [exact quantize4 refinement design](QUANTIZE4_DESIGN.md) specifies shared
range/packing helpers, complete empty-mask coverage, scalar versus neutral-lane
mixed-group candidates, and an independent CM-relative acceptance study.
The refinement is now implemented in `quantize_refined.inc`: ranges and packing
are shared, all-empty groups bypass divisions, and mixed groups use safe
neutral lanes before exact divisions and skip empty-row packing. The scalar
mixed alternative was tested separately and regressed on some diagnostic masks.
Both experimental CMake options default OFF after an inconclusive short
graph-level study. Compile-time selection retains `quantize_legacy.inc`;
final default/candidate libraries match their respective measured binaries
byte for byte. No callback dispatch, persistent cache or ABI was introduced.
Do not add public presets, per-context state, or a vendor-specific dispatcher
on the basis of one CPU's microbenchmark.

The [Ryzen recheck](../validation/ryzen_vnni_retest_20261110.json) supports
retaining the selected CM-only VNNI branch. It found no SIMD stack spills in
the CM integer K loop of either the CM-only or combined fresh PGO libraries.
The combined variant beat the baseline in continuous mode, but its increment
over CM did not pass the predefined A/A robustness guard. Quantize4 remains a
numerically validated experiment; lower source-level division counts alone do
not establish a full-graph benefit.

New performance claims must use the [reproducible benchmark laboratory](BENCHMARK_LAB.md).
The [isolated-L3 non-boost follow-up](../validation/quantize4_lab_20261110.json)
supports a conditional model-8 VNNI batch benefit (+0.832%, adjusted interval
[0.580,1.084]%) with matching A/A precision qualified. Ordinary-power and paced
acceptance remain outstanding, so this does not change the global defaults.
The common dlopen measurement worker keeps CPU-time syscalls outside measured
series and distinguishes amortized groups from individual-call latency.

The subsequent [48-condition matrix](../validation/quantize4_regimes_20261110.json)
confirms model-8 VNNI gains in steady batch16/batch64/paced10 and isolated-Boost
continuous calls, under the planned A/A/effect gates and family correction.
Boost-paced results remain too noisy for acceptance. Model-2 and AVX2 conditions
do not pass all gates. This is no basis for a global dispatch/default change:
the two models retain distinct eligible quantization work after CM, and actual
clock/wakeup behavior changes across cadences despite a performance governor.

The [final ready-binary qualification](../validation/final_qualification_20261110.json)
adds full regenerated 65-fixture spectral/state/PCM equivalence for both models,
red AVX2 portability and a 72-condition independent fixed-layout timing matrix.
It retains 3,456 processes / 811,008 calls and 288 separate PMU windows, spanning
fast/compact Ryzen, steady/Boost, spectral continuous/paced and full FFT/NN/IFFT
callbacks. Model-8 native quantize4 passes useful-gain gates for fast/steady
continuous (+0.810%) and paced10 (+0.863%); no global default promotion follows.
The selected CM implementation retains confirmed fast/steady benefits for both
models. Most A/A cells are too noisy for the 0.5% precision contract, especially
Boost audio. Exact numerical success does not imply universal timing gains.
The original historical reference corpus and real device/service-load behavior
are not certified. Secondary cumulative binary comparisons are exploratory;
do not add separately estimated percentages. All laboratory settings are
independently verified restored, and production services remain unchanged.
