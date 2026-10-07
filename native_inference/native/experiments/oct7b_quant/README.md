# Exact fused quantized row candidates after Oct7

These transforms start from frozen `scratch/oct7/{2,8}/combo_asm_norm` sources.
They guard both `int8.c` and the retained `qdot64_fixed_avx2.S` hashes, edit
only the root driver's isolated copy, and expose `VARIANTS` and `apply`.
No model weights, integration preset, existing source snapshot or result is
modified. No CPU build, correctness check or timing has run for this pass.

Suggested screening order:

| Variant | Change | Main uncertainty |
| --- | --- | --- |
| `quant_fused192_init` | One K=64,N=192 assembly call; initialize integer accumulators with the zero-point correction; emit FP32 scales/FMA directly. | More pointer arguments and code size may offset eliminated intermediate stores and calls. |
| `quant_fused192` | Same whole-row call, retaining correction after the integer dot. | Separates fusion gains from the exact integer reordering. |
| `quant_fused64` | Fuse each 64-column tile's dot and epilogue, retaining the original C tile loop. | Extra argument setup for each tile may erase gains. |
| `quant_fused192_init_inline` | Whole-row correction-initialized fusion plus an always-inline fixed-K64 copy of the original quantizer. | Inlining can increase register pressure and instruction footprint. |
| `quant_inline64` | Only specialize and inline the original quantizer for K=64. | Prior generic inlining was unhelpful; this isolates fixed-shape quantization. |

The assembler retains eight integer accumulators through zero-point
correction, integer-to-FP32 conversion, the separately rounded
`activation_scale * weight_scale`, and the original bias FMA. It writes final
floats directly. For N=192 it reuses scale and zero-point broadcasts across
three sequential 64-column tiles inside one leaf call.

Initializing with the correction is an exact integer reordering. W7A8 pair
products remain within +/-32,004 before `vpmaddwd`. A K=64 dot is bounded by
+/-1,024,128, as is the zero-point correction; the conservative intermediate
bound +/-2,048,256 fits int32 comfortably. There is no floating-point
reassociation, reduced division precision, new gate approximation or new ISA.

The private Linux x86-64 SysV function takes six pointer arguments, one float
and two integers. The float arrives in XMM0 and is broadcast into YMM15 before
YMM0 becomes an accumulator. Zero point and tile count arrive at `[rsp+8]`
and `[rsp+16]`; tile count is restricted by C dispatch to 1 or 3. Only caller
saved registers are used. The function changes no stack pointer, includes
unwind CFI/non-executable stack metadata, and preserves caller FP controls.
Other dimensions and platforms retain their accepted Oct7 paths.

`fused_contract.c` is copied into each fused candidate and adds one direct C
contract, independent of the model runtime. It checks scalar int32 sums and
an explicitly rounded FP32 scale product followed by scalar bias FMA. It
uses one/three tiles, all four rounding modes with FTZ/DAZ off/on, exact float
bits including signed zeros, extreme/tiny values, mmap guard boundaries,
unaligned valid pointers, read-only inputs and padding canaries. The original
scalar quantization oracle also exercises row shapes and tails through the
modified model kernel.

Assembly itself is not ASan instrumented. Guard pages and the independent
contract complement instrumentation of the surrounding C, and do not prove
universal input equivalence or timing bounds. Full recurrent/audio/FP-control
checks and matched sequential timing remain required before acceptance.
