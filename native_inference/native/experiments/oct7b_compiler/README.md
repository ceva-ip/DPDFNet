# Compiler candidates against the Oct7 exact profile

These experiments change code generation only. They retain every numerical C
and assembly operation, default ISA dispatch, explicit FMA, contraction-off
policy, FP32 state and one inference thread. Correctness must still be checked
after compilation: source-level equivalence alone does not prove binary output.

| Priority | Candidate | Rationale |
| --- | --- | --- |
| 1 | `compiler_ipo_nointerpose` | Cross-file IPO can propagate generated-graph dimensions and inline dispatch helpers when GCC is told that internal calls cannot be replaced by ELF symbol interposition. The public symbol set is retained. This is distinct from the earlier vanilla IPO and no-PLT experiments. |
| 2 | `compiler_pgo_ipo_nointerpose` | Train the changed Oct7 hot graph and let value profiles promote function-pointer calls. Earlier PGO on the pre-fitted-gate build barely helped and is not sufficient evidence for this combination. |
| 3 | `compiler_pgo` | A PGO-only control determines whether useful gains depend on IPO and interposition assumptions. |
| 4 | `compiler_hot_align` | Align only dispatched AVX2/integer functions and important loops. A cheap screen for instruction-fetch/code-layout limits, with possible code-size and cache regressions. |

`transforms.py` exposes `VARIANTS`, `PGO_VARIANTS`, `apply(name, source)` and
`build_options(name, source, target, phase='use')`. Apply exactly one compiler
transform to a fresh copy of `scratch/oct7/{8|2}/combo_asm_norm`. Its guarded
CMake hash is shared by both sizes. Numerical-source transforms can be composed
afterward; the compiler transform should be applied first if another candidate
also changes CMake. Non-PGO build options are empty.

Use `build_options(..., phase='plain')` for independent sanitizer/scalar builds
of a PGO candidate. This selects `DPDF_OCT7B_PGO_MODE=OFF`: no counter generation
or consumption and no existing profile requirement. IPO and the interposition
assumption, if included in the variant, still apply. These correctness builds
do not validate the exact PGO release binary by themselves; release parity is
also required.

For PGO, the generic driver must:

1. Configure using `build_options(..., phase='generate')` and build. Keep the
   final build directory: GCC counter identities include its object paths.
2. Run `train.py` as a separate process. Do not run instrumented CTest first;
   it would mix test-derived counters into the profile. Training rejects an
   existing `.gcda` set instead of silently accumulating it.
3. Reconfigure the same directory with `build_options(..., phase='use')`,
   rebuild, then run contracts and exact parity. No profile-correction or
   missing-profile warning suppression is enabled. The existing `-Werror`
   policy makes unexpected missing/mismatched profiles fail compilation.
4. Screen the resulting binary against fresh matched Oct7 reference calls.
   Instrumented training durations are not latency results.

Example worker invocation, from `native_inference/` in the development image:

```sh
python native/experiments/oct7b_compiler/train.py \
  --model models/dpdfnet8_48khz_hr.onnx --weights models/rework8/weights.f32 \
  --source scratch/oct7b/8/compiler_pgo_ipo_nointerpose \
  --build build/oct7b_8_compiler_pgo_ipo_nointerpose \
  --output results/oct7b_8_compiler_pgo_ipo_nointerpose_training.json
```

Training uses four independent deterministic streams, 1,000 hops each by
default: the existing synthetic FFT-grid probe, its quiet and higher-level
scaled forms, and separately generated complex noise with real DC/Nyquist bins.
It loads no scored EARS, robustness, clean-control or long-noise fixtures.
Inputs, model, weights, sources, compiler, instrumented library and flushed
counter files are hashed in the training report. All calls use caller-owned
preallocated buffers and in-place state. The parent waits for worker process
exit before collecting counters.

`-fno-semantic-interposition` intentionally assumes consumer software does not
replace native library symbols via ELF interposition. It does not use hidden
visibility, remove exported APIs, impose global `-march=native`, enable
fast-math, add a thread or change the assembly ABI. This profile remains Linux
GCC research until a consumer build and other platforms are measured.
