# Optimized DPDFNet inference

A small, single-threaded C runtime for **`dpdfnet2_48khz_hr`** and
**`dpdfnet8_48khz_hr`**, with bundled weights and optimized AVX2 assembly kernels.
Both models use the final W7A8 implementation: quantized matrix weights and
activations, FP32 recurrent state, and fitted GRU gate functions.

The supported target is **Linux x86-64 with AVX2/FMA and GCC**. Windows users can
build and run it through WSL2. Native Windows, ARM and macOS builds are not
provided by this package. Inference needs no ONNX Runtime, Python, BLAS or GPU.

## Build

From this folder, with GCC, CMake 3.18+ and Python 3.8+ installed:

```sh
python3 tools/build.py
```

This builds `build/libdpdfnet.so`, trains a fresh compiler profile for model 8,
and runs the contract tests. Python's standard library is sufficient for this
build driver; Python is not a runtime dependency. See [build instructions](docs/BUILD.md)
for direct CMake commands, static linking, installation and sanitizers.

## Use

Include [`dpdfnet.h`](include/dpdfnet.h) and link `dpdfnet`. Choose model ID `2`
or `8`, load its checksummed `models/<model>/weights.f32`, create a context,
and initialize the caller-owned stream state with `dpdfnet_init_state`.

Each `dpdfnet_process` call consumes **one 10 ms spectral hop**, represented as
962 floats: 481 complex bins with interleaved real/imaginary components. It
returns the enhanced spectrum and next state without allocating memory.
Use a separate context for every simultaneous call.

The host owns FFT/windowing and overlap-add synthesis. The model's **50 ms
algorithmic delay** is unchanged. See the [API and audio contract](docs/API.md)
before connecting it to an audio callback.

Examples:

```sh
# Verify bundled weight files before loading them.
sha256sum -c models/SHA256SUMS

# Raw float32 spectra: one 962-float frame per hop.
build/process_spectrum 8 models/dpdfnet8_48khz_hr/weights.f32 input.f32 output.f32

# Optional WAV example; requires NumPy and SoundFile.
python3 examples/enhance_wav.py noisy.wav enhanced.wav --model 2

# Measure the C API at a 10 ms cadence; synthetic input is a smoke test.
build/dpdfnet_benchmark 8 models/dpdfnet8_48khz_hr/weights.f32 1000 paced
```

## Results

Intel i7-8700, Linux/WSL2, one thread; mean ± SD over 32 fresh runs/model
(1,000 timed hops/run at 10 ms cadence).

| Final model | Mean ± SD / hop ↓ | Warmed incremental RSS ↓ | Context-owned heap ↓ |
| --- | ---: | ---: | ---: |
| `dpdfnet2_48khz_hr` | **0.774 ± 0.045 ms** | **5.201 MiB** | **3.800 MiB** |
| `dpdfnet8_48khz_hr` | **1.670 ± 0.058 ms** | **6.594 MiB** | **5.077 MiB** |

Quality means on the same six 48 kHz EARS/WHAM mixtures:

| Final model | PESQ-WB (16 kHz) ↑ | STOI ↑ | SI-SNR (48 kHz, dB) ↑ | SIGMOS SIG ↑ | SIGMOS NOISE ↑ | SIGMOS OVRL ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dpdfnet2_48khz_hr` | 2.44130 | 0.935874 | 16.0841 | 3.86312 | 4.61457 | 3.28124 |
| `dpdfnet8_48khz_hr` | 2.52099 | 0.940449 | 16.6446 | 3.84506 | 4.63650 | 3.35663 |

Spectrum, state and PCM regression passes on 65 files per model.
[Measurement methods and quality details](docs/PERFORMANCE.md).

## Project layout

| Folder | Purpose |
| --- | --- |
| `include/` | Public C API |
| `src/common/` | Shared matrix, DPRNN, graph and assembly helpers |
| `src/models/` | Final fixed graphs and model-specific assembly |
| `src/private/` | Internal declarations and symbol isolation |
| `models/` | Both weight files and SHA-256 manifests |
| `examples/` | C spectrum processing and optional Python/WAV adapter |
| `tests/` | API, numerical and assembly boundary contracts |
| `tools/` | Build, repeated benchmarking and audio verification |
| `benchmarks/` | Preallocated C latency/RSS measurement |
| `validation/` | Compact reference hashes and release evidence |
| `docs/` | Build, integration, architecture and measurement documentation |

[Architecture and maintenance](docs/ARCHITECTURE.md) explains the kernel layout.
The code and bundled weights are distributed under [Apache 2.0](LICENSE).
