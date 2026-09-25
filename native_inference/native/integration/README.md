# HushMic native integration (C ABI v1)

This is the integration entry point for both `dpdfnet2_48khz_hr` and
`dpdfnet8_48khz_hr`. The versioned API, prefixed model symbols and checked-in
[generated C, headers and weights](../../artifacts/v1/) address
[HushMic issue #18](https://github.com/Fovty/HushMic/issues/18#issuecomment-5825094931).
Consumers need a C11 compiler and CMake 3.18 or newer; Python, ONNX and ONNX
Runtime are not needed to configure, compile or run this target.

## Build both models together

From the repository root on Linux:

```sh
(cd native_inference/artifacts/v1 && sha256sum -c SHA256SUMS)
cmake -S native_inference/native/integration -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native -j 4
ctest --test-dir build/native --output-on-failure
```

The default output is `build/native/libdpdf_native.a`, containing both models
and one copy of the common kernels. Set `-DBUILD_SHARED_LIBS=ON` for
`libdpdf_native.so.1`. In a CMake host, `add_subdirectory` this integration
directory and link `dpdf_native`; its public include directories expose the
API and both generated headers. A Rust host can link the static library plus
`libm`, or load the shared library's two getters and bind the v1 function table.

CMake verifies all eight source/header/weight/manifest checksums at configure
time and rechecks when an artifact changes. For deployment, retain a trusted
copy of the expected checksum from a pinned repository commit and verify the
installed weight file before passing decoded floats to `create`. The C API
checks float count and finiteness, but does not hash caller memory.

The source distribution avoids imposing the development container's glibc
version on HushMic. Compile it in HushMic's release environment (its examined
Ubuntu 22.04 / glibc 2.35 floor). These changes were tested on Linux x86-64
under Docker/WSL2; other platforms and the actual PipeWire integration need
host-side validation.

## Version negotiation and named INT8 preset

Include [native_api.h](../native_api.h) and either or both generated headers:

```c
#include "dpdfnet2_48khz_hr/generated_model.h"
#include "dpdfnet8_48khz_hr/generated_model.h"

/* Select either model's getter; reject a NULL return (unsupported ABI). */
const dpdf_native_api_v1 *api = dpdfnet8_48khz_hr_get_api(DPDF_NATIVE_ABI_VERSION);
/* Check api != NULL before using it. */
if (!api || !api->preset_supported(DPDF_PRESET_INT8_SELECTIVE)) {
    /* Select HushMic's existing ONNX backend. */
    return;
}
/* weights: verified SHA-256, decoded little-endian float32, weight_count floats. */
dpdf_native_model *model = api->create(weights, weight_count, DPDF_PRESET_INT8_SELECTIVE);
if (!model) return; /* Handle invalid weights or allocation failure. */
/* Caller allocates state[api->state_size]; each spectrum holds 962 floats. */
api->init_state(state);
api->process(model, spectrum, state, enhanced_spectrum, state);
api->destroy(model);
```

ABI v1 uses a frozen `dpdf_native_api_v1` layout, a runtime `abi_version` and
`struct_size`, and an explicit version argument to the getter. Unsupported
versions return `NULL`; unknown presets report unsupported and cannot create
a model. `DPDF_PRESET_INT8_SELECTIVE` is exactly the measured `(4, 8, 7)`
experimental configuration: DPRNN, dense/grouped FC and 1x1 CNN use W8A8;
other CNN, normalization, gates and stream state remain FP32. It does not
require F16C. `DPDF_PRESET_FP32` uses the existing automatic FP32 dispatch.
The old experimental functions remain available for existing callers.

INT8 requires runtime AVX2/FMA support, including OS support for AVX state.
It is never silently changed to FP32. `-DDPDF_ENABLE_AVX2=OFF` keeps a working
scalar FP32 path and makes the INT8 preset unavailable, which lets HushMic
retain its faster ONNX fallback on older CPUs. The runtime does not select or
load ONNX itself. No global `-march=native` flag is used.

The table provides the model name, weight checksum/count, sample rate (48 kHz),
hop size (480), spectrum size (962) and model-specific state size (56,436 or
90,228 floats). `create` copies retained weights; the input blob can then be
released. Each concurrent call needs a separate mutable context, and each
stream needs independent caller-owned state. Initialize/reset state through
the same model's API; the learned normalization seeds are not all zero.
Processing allocates no heap memory. Preserve HushMic's existing STFT/iSTFT,
latency alignment, attenuation, buffering and adaptive model transitions.

## Regenerate or choose custom symbol prefixes (publishers only)

The existing generators keep the default `dpdf_model` prefix. Both accept
`--symbol-prefix`; this renames all model functions and the opaque model type
and emits a matching `generated_model.h` with its own include guard. Shared
kernel symbols retain their names. Compile each generated `.c` separately and
compile the common kernels once. For example:

```sh
python native_inference/native/generate_extended.py \
  native_inference/models/dpdfnet2_48khz_hr.onnx /tmp/dpdf-small --symbol-prefix hushmic_small
python native_inference/native/generate_extended.py \
  native_inference/models/dpdfnet8_48khz_hr.onnx /tmp/dpdf-large --symbol-prefix hushmic_large
```

Use `generate_extended.py` for the named presets and v1 table; the original
`generate_model.py` remains the earlier graph/DPRNN precision experiment.
Prefixes must start with an ASCII letter and use only letters, digits or
underscores. Choose distinct prefixes for models linked into the same binary.

To reproduce the distributed artifacts with the pinned exporter dependencies:

```sh
python native_inference/download_model.py all
python native_inference/native/package_native.py
```

Generation accepts only the two audited ONNX hashes. The publisher script
exports in temporary directories and copies only the C, header, weights and
manifest into `artifacts/v1`; no ONNX or oracle files enter the distribution.
Output is deterministic and independent of source path and wall clock.
`SHA256SUMS` covers all eight files; each manifest also records source ONNX,
generated source/header and weight checksums, symbol prefix and ABI version.
Git preserves these artifacts byte for byte on Windows and Linux.

The `.f32` blobs remain the existing packed FP32 export, 10,329,776 bytes for
DPDFNet-2 and 14,532,272 bytes for DPDFNet-8. The named preset constructs compact
INT8 operator weights at model creation; these are not new ONNX models or
prequantized disk files. Pin the source and artifacts to the same repository
commit. Distribution follows the repository's Apache-2.0 license.

## Validation

The integration CTest links both models into one executable, keeps both alive
while alternating hops, and tests ABI rejection, CPU/preset availability,
invalid weights, exact agreement with the old selective INT8/FP32 settings,
in-place spectrum/state, reset determinism and weight-buffer ownership.
Run it with AVX2 enabled, `-DDPDF_ENABLE_AVX2=OFF`, and
`-DDPDF_SANITIZE=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo`.
