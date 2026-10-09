# C API and audio integration

[`include/dpdfnet.h`](../include/dpdfnet.h) is the complete public API, ABI
version 1. The same library supports model IDs `DPDFNET_MODEL_2` and
`DPDFNET_MODEL_8`. There is one optimized precision configuration.

## Context lifecycle

1. Query `dpdfnet_get_model_info(id)` and check `dpdfnet_cpu_supported()`.
2. Load the corresponding little-endian IEEE-754 float32 weights. Check SHA-256
   against `weights_sha256` or the bundled manifest before using them.
3. Call `dpdfnet_create(id, weights, info->weight_floats)`. It copies/repacks the
   required weights; the source buffer may be freed after successful creation.
4. Allocate `info->state_floats` floats and call `dpdfnet_init_state(id, state)`.
   This initializes normalization seeds as well as zeroed recurrent entries.
5. Call `dpdfnet_process` once per hop, passing the state from the preceding hop.
6. Reset between unrelated streams. Free the state and destroy the context when
   finished. `dpdfnet_destroy(NULL)` is safe.

Unknown model IDs return NULL metadata. Creation returns NULL for invalid
arguments/counts, nonfinite weights, unsupported CPUs or allocation failure.
State initialization and processing return 0 on success and -1 for invalid
IDs/NULL pointers as described in the header. The API cannot infer buffer
lengths from pointers; callers must provide the documented sizes.

| Model | Weight floats | State floats | State bytes |
| --- | ---: | ---: | ---: |
| `dpdfnet2_48khz_hr` | 2,582,444 | 56,436 | 225,744 |
| `dpdfnet8_48khz_hr` | 3,633,068 | 90,228 | 360,912 |

## Processing contract

The input/output each contain **962 float32 values**, shape `[481,2]`, with
real/imaginary pairs. This is the positive-frequency half of an unnormalized
960-point real FFT. One hop advances 480 samples at 48,000 Hz.

Spectrum and state may each be updated in place. Other buffer overlaps are
invalid. All inputs/state/weights must be finite; processing does not scan the
entire state for NaNs. Four-byte float alignment is sufficient: 32-byte caller
alignment is not required. Keep the process floating-point control settings
stable; the validated release uses round-to-nearest and the host's default
denormal handling.

A context owns mutable scratch memory. Do not call the same context
simultaneously or destroy it during processing. Separate contexts can run
concurrently and can select different models. State belongs to the caller;
there is no hidden stream state or internal worker thread.

`dpdfnet_process` performs no allocation. Weight loading, context creation,
reset, and memory allocation should happen outside the audio callback. Some
kernel scratch uses the caller's stack; reserve at least 1 MiB of available
stack when creating an inference thread. Preallocate spectral/audio buffers
and avoid worker pool contention in the callback.

## FFT, synthesis and delay

Use a 960-sample analysis frame, a 480-sample hop and the float32 window:

```text
w[n] = sin((pi/2) * sin(pi*(n+0.5)/960)^2),  n = 0..959
```

Initialize analysis history with 480 zeros. Frame zero contains that history
and the first 480 input samples. Advance one hop, take a real FFT, and pass
interleaved real/imaginary bins to the model. Keep DC and Nyquist imaginary
components zero for a real input. Do not normalize the forward FFT.

For synthesis, use a 960-point inverse real FFT with `1/960` normalization,
multiply by the same window, overlap-add, and emit 480 samples per hop.
The complete pipeline has 2,400 samples (**50 ms**) of algorithmic delay.
That signal delay is distinct from CPU execution time.

For offline files, feed six extra zero hops to drain the pipeline, discard the
first 2,400 output samples and keep the original input length. The Python
[WAV example](../examples/enhance_wav.py) implements this exact convention and
rejects inputs other than 48 kHz mono. Live applications keep the causal delay
and drain only when ending a stream.

The Python [ctypes adapter](../examples/dpdfnet.py) verifies weight checksums,
owns a state vector and returns a borrowed output array overwritten by the
next call. Copy returned output when retaining it. Use its context manager or
call `close()` explicitly.

## Memory accounting

`dpdfnet_owned_bytes` reports context-owned heap allocations, including the
16-byte public wrapper on the supported 64-bit ABI. It excludes caller state,
spectra, thread stack, code, allocator overhead/retention, and temporary source
weights during creation. It is neither model file size nor process RSS.

Context creation temporarily needs the original float32 weights alongside
packed storage. Budget that startup peak separately from steady-state memory.
