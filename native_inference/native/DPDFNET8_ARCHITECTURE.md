# `dpdfnet8_48khz_hr` architecture and optimization map

This note records the architecture-first analysis used for the next native
optimization pass. The audited artifact is
`dpdfnet8_48khz_hr.onnx`, SHA-256
`7b3afbb260a08fe9af3d16e3bda992971be1e7e951d1dee7c2d235f5c43f5631`.
The graph topology and tensor shapes below come from that artifact and the
repository's streaming exporter, not from model-name assumptions. In
particular, this downloaded artifact has stale `profile=dpdfnet2_48khz_hr`
metadata even though its state size and sixteen DPRNN blocks identify the
DPDFNet-8 topology.

Primary references are the
[DPDFNet paper](https://arxiv.org/html/2512.16420v3), the repository's
[`DPDFNet48HR` implementation](../../onnx_model/dpdfnet_48khz_hr.py), and its
[`DPRNNBlock` implementation](../../onnx_model/layers.py).

## Streaming contract

- One 48 kHz mono hop is 480 samples / 10 ms.
- Model input and output are one 960-point real FFT frame:
  `[1, 1, 481, 2]`, or 962 FP32 values.
- The recurrent state contains 90,228 FP32 values.
- Feature normalization consumes the first 577 state values (481 magnitude
  and 96 complex-spectrum bins).
- The neural graph predicts a full-band magnitude mask and five complex deep
  filter taps for the lowest 96 bins, with two frames of look-ahead.

## Encoder and dual-path topology

The high-resolution variant differs from the paper's generic 32-ERB diagram:
its first encoder branch consumes 480 per-bin log-magnitude features. Causal
convolutions reduce that axis `480 -> 160 -> 80 -> 40`. The complex branch
consumes the lowest 96 complex bins and reduces `96 -> 48`. Both produce
64-channel tensors.

Each branch then executes eight DPRNN blocks before the branches are flattened,
projected, concatenated, and passed to the original DeepFilterNet2-style GRU
encoder and two decoders:

```text
full magnitude [480] -> conv stack -> [40 x 64] -> 8 DPRNN blocks --+
                                                                  +-> fusion/GRUs -> mask + DF coefficients
complex low band [96] -> conv stack -> [48 x 64] -> 8 DPRNN blocks +
```

Every DPRNN block is residual and has two stages:

1. A bidirectional 64-unit GRU traverses frequency within the current frame,
   followed by a `128 -> 64` projection and LayerNorm.
2. A shared unidirectional 64-unit GRU advances time independently at every
   frequency position, followed by a `64 -> 64` projection and LayerNorm.

Only the inter-stage hidden vectors persist between hops. The eight magnitude
blocks own `8 * 40 * 64 = 20,480` state values and the eight complex blocks own
`8 * 48 * 64 = 24,576`, for 45,056 DPRNN state values total.

## Compute and storage concentration

For a branch width `F`, one DPRNN block performs the following dense MACs per
hop. Bias, gates, normalization, residuals, and layout copies are additional.

| Work in one block | MACs |
| --- | ---: |
| Two intra input projections and two recurrent projections | `4 * F * 64 * 192` |
| Intra output projection | `F * 128 * 64` |
| Inter input and recurrent projections | `2 * F * 64 * 192` |
| Inter output projection | `F * 64 * 64` |

This is 3,440,640 MACs for `F=40` and 4,128,768 for `F=48`.
Across all sixteen blocks, DPRNN work is **60,555,264 MACs per hop**, about
84.5% of the published 7.17 G MAC/s model total at 100 hops/s. The blocks hold
1,400,832 parameters including FP32 biases and LayerNorm parameters.

The remaining generated model contains:

- ten dense 256-unit GRU projections with 1,966,080 matrix weights;
- 200 grouped-FC contexts with 211,968 matrix weights;
- nine pointwise convolutions with 32,868 weights;
- twenty-one other convolutions with 5,312 weights.

The grouped and convolution weights are small; their many small invocations and
layout work matter more than raw multiply count.

## Measured cost map

The checked-in instrumented FP32 profile averages 300 recurrent hops. It is a
diagnostic build, so its 3.647 ms total is not a release latency number, but the
relative concentration is useful:

| ONNX/native node family | Mean per hop | Share |
| --- | ---: | ---: |
| 16 native DPRNN blocks | 2.540 ms | 69.6% |
| 10 dense GRU `Gemm` nodes | 0.409 ms | 11.2% |
| 30 convolutions | 0.321 ms | 8.8% |
| 19 materialized transposes | 0.147 ms | 4.0% |
| 10 grouped `MatMul` nodes | 0.055 ms | 1.5% |
| All remaining elementwise/state operations | 0.175 ms | 4.8% |

The final selective INT8 mode already compresses and quantizes DPRNN and dense
GRU matrices while leaving grouped FC and non-pointwise convolution paths FP32.
The earlier ablation showed that generic im2col-based reduced-precision
convolution is slower, so broadening the precision mask is not the next step.

## Ranked optimization directions

The next experiments are ordered by expected gain and semantic risk:

1. **Retile the AVX2 INT8 affine kernel for single-row GRU recurrence.** The
   current output-major loop reloads and broadcasts each four-byte activation
   group once per eight output channels. Holding several output accumulators
   while walking `K` preserves integer accumulation order and exact results,
   while reusing each activation broadcast. This directly targets the 1,408
   sequential intra-GRU recurrent projections executed per hop.
2. **Quantize shared inputs once.** The same frequency-major input feeds both
   directions of every intra GRU. A prequantized-row API can remove the second
   min/max and conversion pass without changing weights or arithmetic.
3. **Keep DPRNN stacks frequency-major between blocks.** The current block API
   transposes `C x F -> F x C` at entry and reverses it at exit for every block.
   Each branch needs only one conversion at the stack boundary. This is exact
   and also aligns the last stack output with the following flatten operation.
4. **Vectorize/fuse LayerNorm plus residual for reduced-precision modes.** This
   removes scalar double-precision passes over 64 channels. It is not bit-exact,
   so it requires recurrent parity and the full quality gate.
5. **Fuse known transpose/pointwise-convolution pairs.** Two of the largest
   non-DPRNN nodes are decoder layout conversions. Generating the consumer's
   preferred layout can avoid those copies without changing model math.
6. **Optional two-worker encoder mode.** The 40-bin and 48-bin DPRNN stacks are
   independent until fusion and can run concurrently. This improves wall-clock
   latency at the cost of another occupied core, so it must be reported
   separately from one-thread efficiency.

Weight-only INT4, low-rank factorization, structured pruning, and dynamic block
skipping remain later research candidates. They change the learned function or
add unpack/control overhead on AVX2, so they should follow the exact
transformations above rather than be mixed into the first performance result.

## Implemented exact pass

The first three directions were implemented, plus a specialized one-row entry
path for the recurrent projection. On this AVX2-only i7-8700 host, the final
selective INT8 candidate improved the preserved implementation as follows:

| Measurement | Preserved | Optimized | Saved |
| --- | ---: | ---: | ---: |
| Continuous median run mean | 2.940 ms | 2.710 ms | 7.8% |
| Continuous median frame p50 | 2.865 ms | 2.626 ms | 8.3% |
| Paced median run mean | 3.053 ms | 2.891 ms | 5.3% |
| Paced median frame p50 | 3.011 ms | 2.838 ms | 5.7% |

The FP16 path benefits only from stack layout elision: continuous mean improved
1.4%, while paced mean changed by 0.1%. Both FP16 and INT8 candidates matched
their preserved outputs and complete recurrent states bit-for-bit over 500
frames. All four release contracts and all four ASan/UBSan contracts passed.

Controlled ablations retained in `results/` show that bidirectional input
fusion saves about 1.4% paced mean latency, the specialized one-row entry saves
about 1.0%, and the 64-output tile plus fusion saves 1.8% versus the otherwise
identical 32-output/unfused control. The full gain is larger because layout,
tiling, fusion, and single-row specialization compose across all sixteen blocks.

## Acceptance gates

- Compare candidate and preserved baseline libraries in rotating order on the
  same frames; report continuous and 10 ms paced distributions.
- Exact transforms must be bit-identical to the current selective INT8 output.
- Numerically approximate transforms must pass recurrent synthetic parity,
  independent-stream tests, ASan/UBSan, and the seven-mixture quality suite.
- Keep original ONNX and native FP32 as reference points. Do not attribute
  multi-core latency gains to single-thread kernel efficiency.
