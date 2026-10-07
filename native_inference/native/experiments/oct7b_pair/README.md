# Fixed-K paired row/tile integer kernels

`quant_pair64` replaces only K=64 calls inside the existing four-row/two-tile
dot wrapper; `quant_pair64_96` also replaces K=96. All other K widths/platforms
keep the original intrinsic function. A SHA guard checks the entire original
`qdot4pair` block rather than the whole file, allowing this transform after the
independent Oct7b quantized-row fusion/inline transforms. Those other transforms
must come first because their own whole-file guard requires untouched sources.

Both assembly functions fully unroll increasing four-element K groups, with
eight int32 accumulator vectors, one word-ones vector, two shared weight vectors,
four activation broadcasts and one multiply scratch vector. They need no stack
allocation or callee-saved register. Each activation broadcast feeds both weight
tiles. Input is unsigned A8 [0,254], weights signed W7 [-63,63]; paired products
are bounded by 2*254*63=32,004 and cannot saturate signed int16. K=96 complete
sums are at most 1,536,192, safely inside int32. Correction/scaling/bias remain C.

Linux x86-64 SysV ABI: RDI points to four K-wide activation rows, RSI/RDX each
point to one packed K-by-eight weight tile, RCX points to 64 int32 outputs.
Outputs retain the original order: tile0 rows0..3, then tile1 rows0..3. All YMM
registers are caller-saved; `vzeroupper`, CFI and a non-executable stack note are
included. A larger instruction footprint may lose against the intrinsic loop;
only a matched model screen can establish a speed benefit.

Static bounds per invocation:

| K | Activation bytes | Bytes per weight pointer | Output bytes |
| --- | ---: | ---: | ---: |
| 64 | 256 | 512 | 256 |
| 96 | 384 | 768 | 256 |

For each J=0,4,...,K-4, a row broadcast reads [R*K+J,R*K+J+3], R=0..3;
weight loads read [J*8,J*8+31]. The last end addresses are 4*K-1 and 8*K-1.
Eight output stores cover exactly bytes 0..255. There are no dynamic address
loops, additional ISA requirements or retained model allocations.

The added CTest contract calls the assembler directly without linking the
runtime. It packs weights independently and compares scalar int32 dots for nine
zero/extreme/sign/order patterns and 256 deterministic random patterns at three
placements: immediately before an inaccessible guard page, immediately after
one, and separate unaligned gaps. Both weight inputs and activations are
read-only; surrounding canaries and untouched payloads are checked.

```sh
python native/experiments/oct7b_pair/run_contract.py \
  --source scratch/oct7b/8/quant_pair64_96 \
  --library build/oct7b_8_quant_pair64_96/libdpdf_full.so \
  --output results/oct7b_8_quant_pair64_96_contract.json
```

The runner checks frozen-source manifests and records compiler, source,
executable, optional library and log hashes. `--sanitize` instruments the C
harness; assembly memory operations remain covered by the independent guard
pages/static bounds. No build, test or benchmark ran from the implementing agent.
