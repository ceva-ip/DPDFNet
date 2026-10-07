`quant_fused256` fuses each external GRU's K256/N768 row projection into one
private Linux SysV AVX2/FMA call. It keeps the existing W7 packed layout and
quantizer, runs 12 tiles of 64 outputs, and directly writes the final FP32 values.
There are no intermediate integer-output stores/reloads, extra owned weights,
process allocations, or threads. Other shapes, batches, and platforms retain
the frozen Oct7 paths.

The source guard accepts untouched Oct7 `int8.c` and the known exact preceding
`oct7b_quant` transformations. It removes those recognized additions, hashes the
remaining text against the committed baseline, and verifies both baseline and
preceding fused64 assembly. Apply this transform after any fused64/inline64
transform when combining variants.

The leaf uses only caller-saved registers. Activations occupy 256 bytes; each
64-output weight tile occupies 16,384 bytes; the 12-tile call reads exactly
196,608 weight bytes and 3,072 bytes from each sum/scale/bias array, and writes
3,072 output bytes. The unsigned activation grid is 0..254 and weights are
-63..63, so a saturated 16-bit pair stays within +/-32,004. Raw dot products
and zero-point corrections are each at most 4,096,512 in magnitude; even the
conservative sum of their absolute bounds remains below 8.2 million. Integer
correction precedes conversion, the activation/weight scale multiplication is
a separate FP32 operation, and bias uses the original FMA operand order.

`wide_contract.c` links directly against the generated assembly. It uses an
independent scalar corrected-integer dot, volatile FP32 scale product and
`fmaf` oracle. The cases cover 1/12 tiles, all four rounding modes with FTZ/DAZ
both off/on, signed zeros, extreme and seeded random inputs, exact-length guard
pages at both ends, unaligned operands, read-only source pages, and output
canaries. Expected coverage is 1,536 calls and 638,976 FP32 outputs. ASAN does
not instrument the assembly itself; these guard pages and independent oracle
are necessary alongside whole-model and sanitizer contracts.

This is an unmeasured research candidate. Assembly code and contract source
have only been implemented and reviewed; root coordinates builds, correctness,
quality and latency jobs separately.
