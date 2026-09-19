# FC/CNN precision and memory experiments

This follow-up extends [the complete C model](FULL_MODEL.md) beyond DPRNN
weights. The FP32 model and earlier timing binaries are preserved. The new
generated target has independent controls for:

| Mask bit | Operator family | Scope |
| --- | --- | --- |
| 1 | Dense GRU/FC | Ten 256→768 projections in the remaining GRU cells |
| 2 | Grouped FC | All ten grouped MatMul operators (200 group contexts) |
| 4 | 1×1 CNN | Nine dense pointwise convolution operators |
| 8 | Other CNN | Twenty-one remaining convolutions, including depthwise and spatial/temporal kernels |

FP16 means compressed weights loaded through F16C with FP32 accumulation;
this host does not execute native half-precision arithmetic. INT8 uses
per-output weight scales and per-row dynamic asymmetric activation scales,
integer dot products, and FP32 dequantization. Bias, normalization, nonlinear
activations, recurrent state and complex filtering retain FP32 storage.

The extended constructor accepts a DPRNN tier separately, so the remaining
families can be evaluated individually. The evaluation combines each FP16
family with FP16 DPRNN and each INT8 family with INT8 DPRNN. A mask of 7
compresses dense/grouped FC and pointwise CNN; 15 includes every CNN.

## Measured results

Intel i7-8700, Linux x86-64 under Docker/WSL2, one inference thread. Times are
the median of three run means (1,000 timed hops after 100 warmup hops each),
with configuration order rotated. Paced runs submit one hop every 10 ms;
reported time covers spectral model inference, excluding FFT and audio I/O.
Memory is the median fresh-process RSS increase in decimal MB.

| Configuration | Continuous ms | Paced ms | Resident increase MB | Initialization peak MB |
| --- | ---: | ---: | ---: | ---: |
| Previous complete C FP32 | 3.470 | 3.627 | 25.00 | 36.88 |
| Previous DPRNN FP16 | 3.219 | 3.355 | 27.82 | 39.74 |
| New FP16 DPRNN + dense FC | 3.274 | 3.454 | 15.09 | 27.02 |
| New FP16 DPRNN + dense/grouped FC + 1×1 CNN | 3.249 | 3.406 | 14.59 | 26.50 |
| Previous DPRNN INT8 | 3.190 | 3.353 | 26.57 | 38.47 |
| New INT8 DPRNN + dense FC | 2.948 | 3.173 | 12.46 | 24.34 |
| New INT8 DPRNN + dense/grouped FC + 1×1 CNN | 2.979 | 3.130 | 11.08 | 22.92 |

The combined selective INT8 configuration reduces paced latency by **6.6%**
and resident increase by **58.3%** versus the previous INT8 implementation.
Selective FP16 reduces resident increase by **47.6%** versus previous FP16,
but does not improve timing (paced mean is 1.5% higher). Memory improvements
combine lower precision with removal of duplicate FP32 weights; they must
not be attributed solely to quantizing additional operator families.

All paced configurations recorded zero inference calls above 10 ms across
3,000 timed hops each. This is a sample, not a real-time deadline guarantee.
Selective INT8 paced p99 ranges from 3.59 to 4.95 ms across runs.

The broader operator ablation measured all-CNN FP16 at 6.15 ms and all-CNN
INT8 at 7.92 ms continuous, with 22.37/18.93 MB of owned allocations.
Those generic convolution paths are unsuitable for the fast preset. Individual
family results are retained in [the screening data](../results/extended_ablations.json).

Use selective FP16 when memory and minimal numerical change matter most.
Selective INT8 is the more promising speed/memory candidate, pending broader
quality validation. Keep other convolutions FP32. Neither experimental mode
is enabled automatically, and these results do not validate native Windows,
macOS/ARM or other HushMic deployment targets.

Raw results: [final timing and memory](../results/extended_final.json),
[computed summary](../results/extended_summary.json).

## Actual memory ownership

Three changes reduce live allocations instead of merely adding compressed caches:

- The generated graph retains only 28,548 bytes of directly accessed constants.
  Matrices and biases are copied once into the operator that owns them, and
  DPRNN source weights are not retained again by the graph.
- FP16/INT8 dense operators release their temporary FP32 matrices after packing.
- FP16/INT8 DPRNN blocks release the full FP32 matrix copies, retaining only
  1,536 FP32 bias/normalization values per block plus the compressed matrices.

`dpdf_model_owned_bytes()` accounts for requested live heap bytes, including
contexts, matrices, scales, biases, scratch and alignment slack. It excludes
allocator metadata, code, stack, caller buffers and the input weight blob.
`memory_probe.c` independently measures Linux RSS in a fresh native process per
configuration: mmap weights, construct model, unmap input weights, process 120
hops, then sample RSS. Peak RSS includes initialization. No Python or ORT is
loaded in these memory-test processes. Three repetitions expose process variance.
Peak uses `/proc/self/status` VmHWM; `getrusage` was unsuitable because the
spawned process inherited its Python parent's historical high-water mark.
Legacy libraries lack owned-byte accounting (zero in raw data means unavailable).

The input artifact is still the **14,532,272-byte FP32 export** for every mode;
these changes reduce resident model memory, not package size. Serialization
of directly loadable FP16/INT8 weights remains separate work. The 4.90 MB FP32
graph arena and caller's 90,228-float state have not been compressed.

## Why not quantize every convolution?

The reduced-precision non-pointwise CNN experiment forms small patches and
uses the same matrix kernels. It works, but pads narrow outputs and performs
extra data rearrangement. That is particularly inefficient for depthwise
convolutions. Its larger scratch/packed arrays can consume more memory than
the tiny original weights. This result characterizes **this implementation**,
not a fundamental limitation of FP16 or INT8 depthwise kernels. A dedicated
depthwise SIMD kernel could behave differently.

These slow variants remain explicit experiments. They do not replace the
direct FP32 convolution implementation or change `AUTO` defaults.

## Quality and validation

The seven-mixture check uses one public-domain speaker, including four
controlled mixtures with clean references. Relative to original ONNX output,
selective FP16 (DPRNN + dense/grouped FC + 1×1 CNN) changes clean-reference
PESQ by +0.000031 to +0.000188 and STOI by -0.00000077 to +0.00001092.
Selective INT8 changes PESQ by -0.000393 to +0.005621 and STOI by
-0.000540 to +0.0000457. These small positive and negative perturbations are
not evidence of improved quality. FP16 is effectively unchanged on this
test; INT8 remains close, but neither this small corpus nor objective scores
can establish universal perceptual equivalence.

Recorded validation:

- Release and ASan/UBSan: all four contract suites pass, including expanded
  INT8 dimensions, precision masks, aliasing, reset and finite outputs.
- Scalar-only build: all three applicable suites pass; unsupported precision
  requests are rejected. A 300-frame recurrent comparison passes strict parity.
- Four independent simultaneous streams match serial results bit-for-bit for
  FP32, selective FP16/INT8 and all-convolution FP16/INT8 configurations.

Raw evidence: [quality](../results/extended_quality.json),
[release](../results/extended_contract.log),
[sanitizers](../results/extended_sanitizer.log),
[scalar tests](../results/extended_scalar.log),
[scalar parity](../results/extended_scalar_parity.json),
[independent streams](../results/extended_state.json).

## Reproduction

From the repository root in PowerShell, using the existing development image:

```powershell
& native_inference/.venv/Scripts/python.exe native_inference/native/generate_extended.py native_inference/models/dpdfnet8_48khz_hr.onnx native_inference/models/extended
$nativeMount = "type=bind,source=$((Resolve-Path native_inference).Path),target=/bench"
docker run --rm --network none --mount $nativeMount --entrypoint cmake dpdfnet-native-dev -S native -B build/extended '-DDPDF_GENERATED_MODEL=/bench/models/extended/generated_model.c' -DDPDF_EXTENDED_MODEL=ON
docker run --rm --network none --mount $nativeMount --entrypoint cmake dpdfnet-native-dev --build build/extended -j 4
docker run --rm --network none --mount $nativeMount --entrypoint ctest dpdfnet-native-dev --test-dir build/extended --output-on-failure
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-dev native/extended_probe.py
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-quality native/extended_quality.py
docker run --rm --network none --mount $nativeMount --entrypoint cc dpdfnet-native-dev -O2 native/memory_probe.c -ldl -o build/memory_probe
docker run --rm --network none --mount $nativeMount --entrypoint python dpdfnet-native-dev native/extended_final.py
& native_inference/.venv/Scripts/python.exe native_inference/native/summarize_extended.py
```

The quality script needs the public-domain voice source prepared by
[FULL_MODEL.md](FULL_MODEL.md). It screens every configuration on the café
fixture, and four finalists on all seven mixtures. Clean-reference alignment
is determined using original ONNX output and reused for every candidate.
The earlier `build/full/libdpdf_full.so` is a preserved historical binary in
this workspace, used for the final comparison; rebuilding that path changes
the baseline. Always use separate build directories while a benchmark runs.

Use `-DDPDF_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo` in a separate directory
for ASan/UBSan. `-DDPDF_ENABLE_AVX2=OFF` exercises the scalar fallback and rejects
unsupported reduced-precision requests. Hardware/OS and deployment limitations
from FULL_MODEL.md still apply; this is not an installed HushMic backend.

## C API

The extended generated library additionally exports:

```c
/* DPRNN FP16 + FP16 dense/grouped FC and 1x1 CNN; other CNN stays FP32. */
dpdf_model *m = dpdf_model_create_config(weights, count,
    DPDF_EXPERIMENTAL_FP16, 16, 7);
size_t owned_heap_bytes = dpdf_model_owned_bytes(m);
```

Check the constructor result, validate the input weight checksum, initialize
caller state with `dpdf_model_init_state`, and use the existing spectral
process function. Each context owns mutable operator scratch and must not
be called concurrently. Separate instances can run concurrently.
