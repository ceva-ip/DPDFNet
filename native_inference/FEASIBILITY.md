# Feasibility: a specialized runtime for DPDFNet8 48 kHz HR

**Follow-up implemented:** the [native DPRNN prototype](native/README.md)
now replaces all 16 blocks in a working ONNX hybrid and measured 5.53 to
3.61 ms/hop on this host. FP32 numerical checks, public audio comparisons,
concurrency, scalar-only fallback and sanitizer tests passed. This document
records the earlier investigation; its statements about unimplemented native
code describe that initial stage, not the current prototype.

Examined 2026-09-18. Local DPDFNet base: `9bd9844a227bb6aa57e55588d8d0e961fcff1c46`.
Reference snapshots: [HushMic ede5673](https://github.com/Fovty/HushMic/tree/ede5673734f4bd8f3676c1195464bb0b146ddb8a),
[faster-enhancer.c 6838816](https://github.com/kdrkdrkdr/faster-enhancer.c/tree/6838816e9ea5369f94f64c57d900796aa9e6a2d5).

## Decision

Proceed with a bounded native recurrent-block prototype. The architecture has
fixed streaming shapes, repeated small recurrent kernels and substantial
tensor rearrangement, which are suitable for specialization. Preserve the
trained model first; quantization is a separate, lossy optimization.

A complete C runtime would require a DPDFNet-specific weight converter,
execution schedule, recurrent/state layout, operators, SIMD dispatch and
parity suite. Replacing ONNX Runtime with C alone is not a speedup mechanism:
its existing CPU kernels are already native and optimized. The opportunity is
fusion, fixed layouts, less dispatch/copying, and better kernels for these
specific shapes, followed by selective reduced precision.

## HushMic determines the first deployment target

The app is currently Linux/PipeWire. Its release assets and bundled ORT are
x86-64; native Linux x86-64 should be the initial acceptance platform. ARM64
Linux is a useful later target, not a verified current HushMic release target.
Windows and macOS portability can be preserved in the DSP library without
assuming that HushMic itself runs there. See the pinned
[release workflow](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/.github/workflows/release.yml)
and [asset provisioning](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/scripts/setup-assets.sh).

HushMic already runs a Rust wrapper with single-thread CPU ORT and Level3
graph optimization. Its graph interface is FP32 `spec[1,1,481,2]` plus
`state_in[90228]`, producing the corresponding spectrum and updated state.
The wrapper copies returned outputs into its own buffers. Binding preallocated
outputs is a small integration experiment, not a replacement for kernel work.
[Model implementation](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/crates/hushmic-denoiser/src/model.rs).

Current HushMic runs inference on a worker, outside the PipeWire callback. It
also has adaptive quality/light/raw switching, with a second model warmed or
evaluated during transitions. A replacement must preserve these protections
and permit independent engine instances; process-global mutable state is a
poor fit. See [worker integration](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/crates/dpdfnet-ladspa/src/lib.rs)
and [adaptive engine](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/crates/dpdfnet-ladspa/src/adaptive.rs).

The reported failures are not simply an average-throughput problem. For
example, [issue #14](https://github.com/Fovty/HushMic/issues/14) describes cutouts
on an Intel i5-1235U Linux laptop during screen sharing/CPU spikes and logs an
overflowed worker input ring. Test tail latency and scheduling under competing
load, including different CPU core types and power modes. A faster kernel
provides headroom but cannot guarantee that the OS schedules the worker.

Keep the current alignment: 480 samples/hop at 48 kHz; 960-point Vorbis FFT;
50 ms algorithmic delay (10 ms framing plus 40 ms model delay); another 30 ms
worker margin at the designed quantum, for 80 ms plugin latency. Faster
computation does not automatically remove these delays.
[Latency constant](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/crates/hushmic-denoiser/src/lib.rs),
[alignment ledger](https://github.com/Fovty/HushMic/blob/ede5673734f4bd8f3676c1195464bb0b146ddb8a/crates/dpdfnet-ladspa/src/align.rs).

## Architecture and optimization transfer

The local [model](../onnx_model/dpdfnet_48khz_hr.py) and
[layers](../onnx_model/layers.py) define two DPRNN branches with eight blocks
each. The branches operate on 40 and 48 frequency positions, with 64 channels.
Each block runs a bidirectional frequency GRU, projection, layer normalization,
residual, then a per-frequency temporal GRU, projection, normalization and
residual. The frequency scan starts afresh each audio frame; the temporal
hidden state persists. Bidirectionality across frequency does not imply
future-audio lookahead.

The downloaded graph contains 937 nodes: 16 bidirectional GRUs, 58 Gemms,
26 MatMuls, 32 LayerNormalizations, 30 Convs and 131 Transposes, among others.
The temporal GRU cells are decomposed into matrix/gate operations: 16 cells
with hidden size 64 plus five with hidden size 256 elsewhere in the network.
Do not mistake the count of GRU operators for the count of all recurrent work.

The 16 DPRNN blocks alone require approximately 6.06 billion dense MACs per
second: `8 * (40+48) * 21 * 64^2 * 100`. This counts both directions' GRU
matrices, temporal GRU matrices and the two projections; it excludes biases,
nonlinearities, normalization and other model stages. Specialization still has
to execute this workload unless precision or architecture changes.

The persistent interface state occupies 360,912 bytes (about 352.5 KiB), not
the total runtime working set. Preserve the 577 metadata-seeded normalization
entries; initializing everything to zero changes behavior. State includes
recurrent, convolution, spectrum and coefficient buffers.

| Reference technique | DPDFNet applicability |
| --- | --- |
| Fixed graph, prepacked weights, preallocated scratch | Strong fit; use per-instance state and shared immutable weights. |
| Fused GRU matrix/gate updates | Strong candidate for temporal cells. Frequency bidirectional scans need additional sequence logic and distinct state handling. |
| Dynamic int8 GEMM, per-row weight scales | Candidate for larger projections/GRU matrices, subject to measured quality and shape-specific benefit. |
| Fused transpose/pack, normalization/residual passes | Strong candidate; choose persistent layouts and fuse only where operator ordering permits. |
| FP16 recurrent/skip storage | Defer. Recurrence can accumulate error; investigate separately from int8 weights. |
| Winograd dense convolutions | Lower priority. DPDFNet uses many separable/grouped convolutions and has a different compute distribution. |
| Attention kernels | No corresponding DPDFNet attention block. |
| Reference FFT and streaming front end | Not interchangeable: reference uses 1024-point FFT/320-sample hops; HushMic uses 960/480. Retain HushMic's FFT first. |

These comparisons follow the reference's
[kernel implementation](https://github.com/kdrkdrkdr/faster-enhancer.c/blob/6838816e9ea5369f94f64c57d900796aa9e6a2d5/src/nn/fe_gru.c),
[optimization notes](https://github.com/kdrkdrkdr/faster-enhancer.c/blob/6838816e9ea5369f94f64c57d900796aa9e6a2d5/docs/optimizations.md),
and [architecture](https://github.com/kdrkdrkdr/faster-enhancer.c/blob/6838816e9ea5369f94f64c57d900796aa9e6a2d5/docs/architecture.md).
Its [README](https://github.com/kdrkdrkdr/faster-enhancer.c/blob/6838816e9ea5369f94f64c57d900796aa9e6a2d5/README.md)
reports ARM measurements and explicitly does not time its x86 tiers. Its
accuracy results concern FastEnhancer, not DPDFNet. They cannot establish
DPDFNet performance or quality on Linux x86-64.

An exporter must also handle GRU conventions explicitly: PyTorch uses gate
order `r,z,n`, while ONNX GRU uses `z,r,h`; this graph sets
`linear_before_reset=1`. Keep the candidate hidden bias inside the reset-gate
multiplication. A silent ordering/bias error can produce plausible but wrong
audio. [ONNX GRU specification](https://onnx.ai/onnx/operators/onnx__GRU.html).

## Measurements from this investigation

Intel i7-8700, Docker Desktop/WSL2 Linux x86-64, ORT 1.27.0, one thread.
Times below are median-of-repeat mean graph times; p99 ranges cover individual
repeats. See [method and commands](README.md) and JSON for full details.

| Probe | Mean ms/hop | RTF | p99 ms across repeats | Result |
| --- | ---: | ---: | ---: | --- |
| Original FP32 | 5.594 | 0.559 | 8.43–9.77 | Baseline |
| Original FP32, paced at 10 ms | 5.682 | 0.568 | 6.78–7.99 | Seven calls exceeded 10 ms across 3,000 timed hops |
| FP32 with output binding | 5.439 | 0.544 | 6.01–8.68 | Identical spectrum and state on 1,200 frames |
| Lowered FP32 intermediate, one repeat | 5.339 | 0.534 | 5.93 | Identical spectrum and state on 1,200 frames |
| Partial dynamic int8 | 5.055 | 0.505 | 5.50–6.98 | Output differs; quality unvalidated |

The partial-int8 model shrank from 14,857,107 to 7,354,286 bytes. The observed
mean reduction is about 9.6% relative to the original graph, or 5.3% against
the single-repeat lowered FP32 intermediate. Sequential runs in a shared VM
are susceptible to clock/load/order effects; these are exploratory results,
not statistically established gains. The small binding/lowering differences
in particular should not be sold as a production improvement.

The paced baseline had p99 start lateness of 0.22–0.24 ms and a worst observed
graph call of 14.18 ms. These measurements had no deliberate competing load;
they do not reproduce the screen-sharing workload reported by HushMic users.

Partial int8 spectrum relative RMS difference was 0.299; state relative RMS
difference was 0.00414. The model heavily suppresses this synthetic input, so
the spectrum reference energy is small. These values neither prove acceptable
quality nor quantify perceptual degradation. Evaluate real speech before
using this artifact. Quantization is not lossless; see
[ORT quantization guidance](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

In the separate instrumented Linux profile, GRU accounted for 30.7% of node
duration, Gemm 15.9%, Transpose 9.9%, LayerNormalization 7.9%, Reshape 7.2%,
Add 7.2% and Conv 3.6%. These suggest recurrent and layout work as the first
targets. Instrumentation can substantially inflate tiny operators, especially
views such as Reshape; these percentages are not direct compute attribution.

The Windows baseline is also retained (median mean 6.139 ms/hop), but its
initial run overlapped Docker image preparation and is not a clean OS
comparison. No native C implementation, real-audio quality evaluation,
bare-metal Linux PipeWire test, ARM run or low-power laptop test was performed.

## Implementation sequence and acceptance gates

1. **Freeze the streaming oracle.** Use this exact model hash and HushMic's
   causal grid, not the offline centered-STFT path. Extend the current probes
   with speech/noise fixtures, impulse/latency tests, long silence, abrupt
   levels, reset/restart, and multiple independent streams. Keep output and
   selected intermediate state traces. Reuse HushMic's public audio/parity
   fixtures with their recorded provenance.
2. **Prototype one complete DPRNN block in FP32.** Cover both frequency lengths
   and both recurrent paths; prepack weights and fix layouts. Check each stage
   against ONNX before optimizing. Then integrate the block into an otherwise
   unchanged graph (custom operator) to measure whole-model benefit. Test
   whether expressing decomposed temporal cells as fused GRUs helps ORT too;
   this was not tested in the current probe.
3. **Choose between a hybrid backend and a full native engine.** A custom
   recurrent operator retains ORT dependencies but reduces initial port scope.
   If measurements justify a standalone runtime, implement the remaining
   grouped linears, convolution/subpixel stages, norms, mask and complex
   five-tap deep filtering. Expose an opaque per-instance C context with
   create/reset/process-spectrum/destroy and explicit errors. Keep the Rust
   STFT, attenuation blending, bypass/mute, worker and alignment around it.
4. **Add SIMD and selective precision.** Start portable FP32 plus runtime-selected
   AVX2/FMA kernels; evaluate AVX-VNNI on suitable laptops and later NEON/ARM
   dot-product. Keep recurrent state, normalization accumulators and complex
   filtering FP32 initially. Quantize one layer family at a time. Retain a
   working baseline/fallback on CPUs without optional instructions.
5. **Prove quality and deployment benefit.** Compare long streaming outputs and
   state, then aligned clean-reference metrics (PESQ/STOI/SI-SNR where suitable),
   DNSMOS and listening tests on diverse speech/noise and difficult microphones.
   Evaluate the existing DPDFNet evaluation set with explicit rate/alignment
   handling. Set numerical and quality tolerances before accepting results;
   same-family bit equality is useful, but cross-CPU FP32 equality is not assumed.
   Benchmark unprofiled graph and full Rust hop separately, raced and paced,
   then actual PipeWire during browser screen sharing and model transitions.

Suggested performance gate, **not a forecast**: pursue a full port if a
representative native prototype shows at least 1.5x whole-model improvement
without a quality regression; aim for p99 below 5 ms at a 10 ms hop on the
selected minimum Linux hardware under a defined load. Record wake-up lateness,
worker backlog, xruns, memory and power as well as average RTF. A low average
with missed deadlines is insufficient.

## Portability and packaging

Build first for Linux x86-64 with GCC/Clang and a C ABI usable by Rust. Compile
ISA variants in separate objects, select them at initialization with CPU and
OS-state checks, and avoid globally enabling `-march=native` for distributable
binaries. Keep a baseline path: faster-enhancer.c rejects x86 CPUs lacking
AVX2/FMA3/F16C, which should not silently become DPDFNet's support policy.
Its CMake also rejects MSVC/clang-cl; direct reuse would not provide universal
Windows compiler support. [Reference build configuration](https://github.com/kdrkdrkdr/faster-enhancer.c/blob/6838816e9ea5369f94f64c57d900796aa9e6a2d5/CMakeLists.txt).

Build release artifacts against HushMic's existing Ubuntu 22.04/glibc floor,
then test supported distributions. Check other libc variants separately; a
glibc binary does not establish musl compatibility. Docker can make builds
and parity runs repeatable, but it should not be part of the user's audio
processing path and cannot replace bare-metal scheduling measurements.

No reference C code was copied into this investigation. Any later source
reuse must retain the reference MIT copyright/license notices and relevant
NOTICE material alongside this repository's Apache-2.0 notices. Keep model
provenance/checksums and third-party notices with distributed weight blobs.
