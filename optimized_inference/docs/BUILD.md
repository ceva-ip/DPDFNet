# Building and installing

The validated toolchain is GCC 12.2 and CMake 3.25 on Linux x86-64. GCC,
CMake 3.18+, and an AVX2/FMA CPU are required. Build and run inside Linux or WSL2;
the hand-written assembly uses the Linux System V ABI.

## Recommended release build

```sh
python3 tools/build.py --jobs 2
```

The driver verifies both weight checksums, builds model 8 with profiling,
trains four independent 1,000-hop streams, rebuilds with profile use, and runs
all tests serially. Model 2 uses its accepted optimized code without PGO.
Profiles are generated locally in a fresh ignored directory on every run.
Missing profiles and coverage mismatches are build errors.

Keep the source and build paths unchanged between profile generation and use.
Re-run the driver after changing source, compiler or compile flags. Synthetic
training inputs exercise ordinary, quiet, strong and silent spectra; actual
latency must be measured on the application's inputs and target machine.

## Direct CMake build

A regular build requires no Python:

```sh
cmake -S . -B build-plain -DCMAKE_BUILD_TYPE=Release
cmake --build build-plain --parallel 2
ctest --test-dir build-plain --output-on-failure
```

This selects the same numerical implementation but omits the model-8 compiler
profile. To train and use the profile without the Python driver, start with a
fresh build/profile directory and use the same build directory for both stages:

```sh
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DDPDF_PGO_MODE=GENERATE -DDPDF_PROFILE_DIR="$PWD/build-profile/profiles"
cmake --build build-profile --parallel 2 --target dpdfnet_train
build-profile/dpdfnet_train models/dpdfnet8_48khz_hr/weights.f32
cmake -S . -B build-profile -DBUILD_TESTING=ON -DDPDF_PGO_MODE=USE
cmake --build build-profile --parallel 2
ctest --test-dir build-profile --output-on-failure
```

Do not enable fast-math, reassociation, automatic contraction, or global
`-march=native`. The build sets strict FP operations and enables AVX2/FMA only
in the relevant translation units. CPU support is checked before creation.

## Installation and consumers

```sh
cmake --install build --prefix "$PWD/build/install"
```

The install contains the library, public header, both weight files/manifests,
license and exported CMake package. A consuming CMake project can use:

```cmake
find_package(DPDFNet 1 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE DPDFNet::dpdfnet)
```

Set `CMAKE_PREFIX_PATH` to the installation prefix. Locate installed weights
under `share/dpdfnet/models`; the library accepts weights in memory and does
not assume an application-specific working directory.

For a static, non-PGO build:

```sh
cmake -S . -B build-static -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-static --parallel 2
ctest --test-dir build-static --output-on-failure
```

Direct static consumers must link `libm`. Exported CMake targets carry that
dependency automatically. The shared library exports only the seven public
functions in `dpdfnet.h`; graph and kernel symbols are private.

## Sanitizers and audio regression

```sh
python3 tools/build.py --build-dir build-asan --sanitize
```

This uses a separate non-PGO build and runs address/undefined-behavior checks.
Assembly is not instrumented: independent scalar oracles and read-only guard
pages test its exact operand lengths, canaries, aliases and FP control modes.
Sanitizer test executables use non-PIE linking to avoid ASan shadow-map startup
collisions on high-ASLR Linux/WSL hosts.

The optional WAV example and audio verifier were validated with Python 3.11,
NumPy 2.4.6 and SoundFile 0.13.1. Install their pinned dependencies with
`python3 -m pip install -r examples/requirements.txt`. The C runtime and build
driver do not need these packages. Other FFT/library versions can change PCM
rounding and reference hashes. With the original regression fixtures available:

```sh
python3 tools/verify_audio.py --data /path/to/fixtures --workers 4
```

`validation/reference_streams.json` records relative fixture paths and input,
spectrum, state and PCM checksums. Audio datasets and listening outputs are
not bundled. Run correctness jobs separately from latency benchmarks.

## Optional AVX-VNNI kernels

`DPDF_ENABLE_AVX_VNNI` defaults to `ON`. CMake checks that GCC supports both
VEX-encoded AVX-VNNI intrinsics and the CPU feature query. If that check fails,
the build retains the AVX2 kernels. To explicitly build only AVX2 kernels:

```sh
cmake -S . -B build/avx2 -DDPDF_ENABLE_AVX_VNNI=OFF -DDPDF_PGO_MODE=OFF
cmake --build build/avx2
ctest --test-dir build/avx2 --output-on-failure
```

The normal binary supports the same AVX2/FMA baseline as before. Each quantized
matrix selects its implementation during creation, using GCC's CPU feature
query. The audio callback performs no feature detection or allocation. Only
`int8_vnni.c` and the corresponding assembly require AVX-VNNI; no global
`-march=native`, AVX-512, or model format change is needed. Initialization adds
two private function pointers per quantized matrix (4,016 bytes for model 2 and
5,552 bytes for model 8 on x86-64). The public ABI is unchanged.

The generated VNNI assembly keeps the addressing, floating-point instruction
order, and ABI of the AVX2 assembly. Regenerate it after editing those sources:

```sh
python3 tools/generate_vnni.py
python3 tools/generate_vnni.py --check
```

The W7A8 activation range is 0–254 and the weight range is -63–63. Therefore,
the AVX2 intermediate signed 16-bit pair sums cannot saturate
(`2 * 254 * 63 = 32004`). Replacing its multiply/add reduction with the
non-saturating `VPDPBUSD` preserves integer sums, quantization, scaling, and
floating-point output. `int8_dispatch_contract` checks bitwise equality with
the AVX2 implementation for both models, including single and paired kernels,
row tiles and tails, extreme input values, and unaligned buffers.
