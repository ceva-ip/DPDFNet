# Exact convolution and pointwise-layout candidates after Oct7

These guarded transforms start from isolated copies of the frozen Oct7
`combo_asm_norm` sources for either model. They do not modify that baseline,
weights, existing reports or integration code. They preserve one calling
thread and the arithmetic selected by each precision/tier configuration.

The new diagnostic profiles attribute roughly 0.209 ms/hop (model 8) and
0.227 ms/hop (model 2) to nodes labelled `Conv`. This includes generated
pointwise paths which already use transpose/dense/transpose. Their largest
node, `convt1.1`, accounts for roughly 0.049/0.056 ms; this is not a standalone
benchmark and instrumentation changes scheduling. The next large pointwise
nodes are decoder `convt2.1` and encoder `erb_conv1.1`.

The accepted mask7 configuration leaves all 21 `dpdf_convop` objects FP32.
Mask8 controls their reduced precision; therefore optimizing generic FP16/
INT8 patch extraction would not directly accelerate the accepted mask7 path.
These candidates instead target two operations actually exercised there.

| Variant | Work targeted | Exactness contract |
| --- | --- | --- |
| `conv_transpose_relu` | Nine pointwise output transpose/ReLU chains, including encoder, decoder and the 10-channel DF projection. | Copy each output bit and apply the original positive comparison during the bounded 8x8 transpose tile. Scalar tiles/tails remain; no affine/quantization/activation approximation changes. |
| `conv_row_pair` | Two DF temporal convolutions with CI=32, KH=5, CO=5, width96. | Two output channels share each input load. Each output retains its original ordered FMA chain for full eight-frequency groups and separate FP32 operations in scalar tails. |

Transpose/ReLU maps negative values, both zero signs and NaNs to positive
zero exactly as the generated `value > 0 ? value : 0.0f`. Positive values retain
their original bits. Its input/output must be disjoint; the guarded generated
rewrite checks their full intervals. Final arena locations and the model's
deferred state concat remain unchanged. No general arena compaction or
decoder arithmetic fusion is introduced.

The paired convolution handles only unpadded unit-stride, single-output-row,
width-preserving 1-wide kernels. Group/channel divisibility and at least two
outputs per group are checked. It uses two outputs by four frequency vectors
(eight accumulators), loading four input vectors once for both channels.
Eight-frequency remainders use the same FMA; final scalar frequencies use
separate multiply/add. Odd output channels reuse the preserved original
row kernel. Unsupported shapes retain the original dispatch/fallback.

Both transforms add one shared CTest oracle to candidate snapshots. It checks
raw bits and canaries using valid unaligned float pointers. Transpose fixtures
cover all pairs of 17 dimensions around 8/16/32-element boundaries, the actual
graph dimensions, signed zeros, subnormals, infinities and NaNs. Convolution
fixtures independently model the original FMA/scalar-tail distinction across
14 widths, odd/even output channels, groups, temporal heights, signed-zero and
quiet inputs, and present/absent bias. Existing whole-model precision, alias,
FP-control and streaming comparisons are still required. No performance
gain is established by candidate generation or this source review.
