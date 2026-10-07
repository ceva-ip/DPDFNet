# Oct 7 exact W7A8 kernel candidates

`transforms.py` applies each candidate to an isolated copy of the accepted
Oct 3 `best_single` sources. It checks the untouched `int8.c` source hash
before editing. It does not build or benchmark, and does not modify the
distributed runtime. It exposes `VARIANTS` and `apply(name, source: Path)`.

The [completed investigation](../../OCT7_OPTIMIZATION.md) selects `quant_asm64`
as part of `combo_asm_norm`, with validated cadence gains for both models.
The other candidates remain recorded alternatives. Both model sizes use the
same shapes in the DPRNN code; no model ID appears in a kernel.

| Candidate | Hypothesis | Cost or limitation |
| --- | --- | --- |
| `quant_broadcast` | Expand a four-row block's byte activations into vector broadcasts once, then reuse those vectors for each output tile. The existing dot repeats these broadcasts for every column tile. | Adds 2 KiB transient storage at K=64, up to 16 KiB at K=512; wider activation loads could increase cache pressure. |
| `quant_2x32` | Two rows by four output vectors retain eight accumulators but replace four activation broadcasts and two weight loads per K step with two broadcasts and four weight loads. | More weight loads and different register pressure may erase the saving. Original dimension fallbacks remain. |
| `quant_blocked_row` | K-major packing of 64-column tiles turns eight 512-byte-strided K=64 weight streams into one sequential 256-byte stream per K step. | Duplicates only K=64,N=192 packed matrices; owns another 12,351 bytes per qualifying matrix plus structure fields. Reported owned bytes include all allocations. |
| `quant_asm64` | Fully unroll the K=64 row dot; four independent memory-folded multiply chains use constant displacements and remove dynamic stride/loop bookkeeping. | Linux x86-64 SysV only; larger code footprint, and the compiler may already schedule the intrinsic loop sufficiently well. Other shapes/platforms retain the original path. |
| `quant_recurrent_layout` | Reorder only the two intra-GRU recurrent matrices' existing packed allocations into K-major 64-column tiles at creation. | Temporary 12,288-byte creation buffer is freed; only a layout flag adds retained metadata. Other matrix layouts remain unchanged. Multi-row/mixed paired calls use exact single-row fallback. |

All candidates retain the original activation quantizer, W7 weight integers,
zero-point correction, per-row/output FP32 scale product, final bias FMA, and
fitted GRU gates. `vpmaddubsw` intermediates satisfy the same
`2 * 254 * 63 = 32004` signed-word bound; K<=512 keeps complete sums in signed
int32. No extra threads, new ISA, approximation or input-dependent skip is
introduced. There is no floating-point reassociation.

The selected profile passed scalar-oracle, compatibility, FP-control and
end-to-end exact-output checks before the final measurements were accepted.
The assembly body is not instrumented by ASan. Its fixed 64-byte input,
4096-byte weight-tile and 256-byte output accesses passed a separate direct
integer oracle with guard pages, read-only inputs, unaligned buffers and
canaries: 795 calls, checking 50,880 integer outputs. The assembly source hash
matches both frozen model builds.

After building both selected snapshots, run the direct check from
`native_inference/` in the Linux development container:

```sh
python native/experiments/oct7_quant/run_asm64_contract.py
```

The runner records source, compiler, executable, library and log hashes in
[`oct7_asm64_contract.json`](../../../results/oct7_asm64_contract.json).
It performs correctness checks only; run it separately from latency jobs.

`quant_recurrent_layout` adds a sixth AVX2 C contract. It checks prepared and
ordinary matrices on 1/2/4/5/48 rows, a mixed prepared/unprepared pair in either
position, both prepared matrices, idempotent preparation, unaligned pointers,
signed-zero inputs/biases, extreme/tiny values, unchanged owned-byte accounting,
and unsupported dimensions rejected without changing their output. The original
independent scalar quantization oracle still uses the ordinary packing.
