# Narrow INT8 dense output padding

`dense_int8_pad8` changes only precision-8 dense construction: `np` becomes
`ceil(n/8)*8` instead of `ceil(n/32)*32`. FP32/FP16 still use the original
32-column padding. Activation K padding, source/tiled packing, quantizer, real
column weights/biases, integer reduction, correction, scaling and bias FMA remain
unchanged. The transform checks the preserved Oct7 `extended_ops.c` hash shared
by both model sizes. Apply before another transform of that file.

The INT8 kernel accepts any N divisible by eight. Its four-row/16-column path
requires N divisible by sixteen; other shapes use its existing eight-column
batch loop and row tails. Consequently N=8/24 may lose the 16-column paired
schedule despite computing fewer columns. Model-specific latency evidence is
required, particularly for the many N=16 projections.

With `kp=ceil(k/8)*8`, old/new padded widths `np0/np1`, exact owned-byte savings
are:

```text
(np0-np1)*(kp+12)
  + 192*((np0!=n ? np0 : 0) - (np1!=n ? np1 : 0))
```

The first term covers bias and retained INT8 weights/scales/sums. The second
covers the 48-row output scratch. K scratch and allocation owners are unchanged.
For N=16, the output scratch disappears as well as the 16 padded columns.
Creation-time FP32 packing allocations also shrink before being freed.

`run_contract.py` compiles an independent C executable that loads baseline and
candidate libraries with `dlopen`. It compares raw float bits and exact expected
owned-memory deltas through the public dense API. It covers N=1/7/8/16/24/31/
32/64/192; K=1/7/8/17/64/127/512; rows=1/2/4/5/47/48/49/96/97; FP32/FP16/INT8;
ordinary/quiet/high/signed-zero inputs; zero weight columns and signed-zero bias;
unaligned buffers, canaries and unchanged disjoint inputs; tiled source packing;
and all four rounding modes with denormal handling off/on after construction.

INT8 alias checks cover <=48 rows, or N<=K for larger in-place chunks, because
each input chunk is quantized before writing output and cannot clobber future
chunks in those cases. The dense header does not promise arbitrary overlapping
input/output; FP32/FP16 aliases and expanding multi-chunk aliases are therefore
outside this oracle's claims. Whole-model arena aliasing still needs the normal
end-to-end exact-state tests.

```sh
python native/experiments/oct7b_dense/run_contract.py \
  --baseline build/oct7_8_combo_asm_norm/libdpdf_full.so \
  --candidate build/oct7b_8_dense_int8_pad8/libdpdf_full.so \
  --output results/oct7b_8_dense_int8_pad8_contract.json
```

Pass `--sanitize` with both sanitizer libraries to instrument the oracle too.
The runner saves compile/run logs and compiler, harness, executable and library
hashes. Unsupported FP16/INT8 modes are skipped on scalar-only builds. No builds
or tests have been run by the implementing agent.
