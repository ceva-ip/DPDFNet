# Further generated-graph candidates on the frozen Oct7 profile

These guarded source transforms apply to isolated copies of
`scratch/oct7/{8,2}/combo_asm_norm`. They preserve the model dimensions,
weights, retained model allocation and single calling thread. Source generation
is separate from building, validation, timing and quality scoring.

The five non-DPRNN GRU cells are shared by both model sizes. Each cell has
256 channels and two 256-by-768 projections. Outside DPRNN, the generated
graph still evaluates **2,560 scalar sigmoids and 1,280 scalar tanhs** in those
cells, plus a 480-wide output sigmoid and a 960-wide output tanh. This gives
**5,280 scalar nonlinear evaluations per hop** as a new optimization target.
Model 2 has fewer DPRNN blocks, so this common work can be a larger fraction
of its inference time. This is a hypothesis; no speedup is established here.

| Variant | Change | Numerical contract |
| --- | --- | --- |
| `graph_gru_fuse` | Keep both affine outputs in separate local 768-float arrays; replace six gate split copies and ten elementwise passes with one helper per cell. | Exact candidate: original scalar `expf`/`tanhf`, operand order and separate rounded FP32 operations. |
| `graph_bias_relu` | Fuse eight adjacent generated bias-add/ReLU chains. | Exact candidate: the same addition followed by `value > 0 ? value : 0.0f`, including signed-zero behavior. |
| `graph_activations_approx` | Replace all 17 generated sigmoid/tanh loops with separate entry points reusing the existing fitted `sigmoid8`/`tanh8`. | Approximate: unchanged fitted coefficients, but these graph nodes previously used scalar libm. Quality must be rescored. |
| `graph_gru_approx` | Fuse the five GRUs using the fitted nonlinearities, and replace the two final output activations. | Approximate: surrounding GRU multiply/add remains separate, without the DPRNN gate FMA. Quality must be rescored. |

The exact GRU helper consumes gate projections in contiguous reset, update,
candidate order. Its arithmetic follows the original generated sequence:
`b_reset + a_reset`, `b_update + a_update`, `b_candidate * reset`,
`a_candidate + product`, `old - candidate`, `difference * update`,
`product + candidate`. It retains the final output's original arena location
and the whole-model deferred state concat. The two local projection buffers
use 6 KiB of transient stack storage, which is not retained model memory.

The approximate variants use AVX2 only when the generated model's existing
dispatch selects AVX2. Scalar builds and scalar tier retain scalar libm.
The pointwise helper supports scalar tails, although the fixed graph counts
are multiples of eight. These candidates do not change quantization, weights,
precision selection, fitted polynomial coefficients or thread count. They
cannot use the Oct7 byte-exact quality-score carry-forward.

The GRU helper's focused C contract is copied into fused candidate snapshots
and added to CTest. Its independent oracle keeps the original staged array
passes and checks raw float bits over ordinary, signed-zero, subnormal, quiet,
large and activation-boundary fixtures. It covers disjoint output, output
equal to old state, and output equal to each of the six input gate slices,
with byte canaries and all supported caller rounding/FTZ/DAZ settings.
The exact scalar helper remains present in approximate fused builds so its
contract can still verify fallback semantics. Approximate AVX2 output needs
separate finite/range and speech-quality evaluation.

For exact candidates, also retain whole-model scalar/FP32/FP16/W7A8 contracts,
in-place processing, precision compatibility, streaming recurrent replay and
caller FP-control comparisons against the frozen Oct7 profile. Operation
scheduling changes do not promise identical floating-point exception trap
order or flags. Preserve `-ffp-contract=off`; do not add fast-math. Benchmark
variants and combinations only after correctness jobs finish.
