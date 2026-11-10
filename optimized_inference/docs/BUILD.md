# Building and installing

The historical package was validated with GCC 12.2 and CMake 3.25.
The i7-7700T per-model selection is additionally validated with GCC 14.2.0
on Linux x86-64; a fresh PGO build and all available release contracts also
passed with GCC 12.4.0. Performance figures remain specific to GCC 14.2.0.
GCC, CMake 3.18+, and an AVX2/FMA CPU are required. Build and run inside Linux or WSL2;
the hand-written assembly uses the Linux System V ABI.

## Recommended release build

For performance studies, use the repository-root [benchmark laboratory](BENCHMARK_LAB.md)
and `just bench-session MANIFEST OUTPUT` after the numerical build gates. Its
prepare/readiness/restore recipes are mandatory for new performance claims;
plain benchmark affinity alone does not establish a quiet environment.

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

### Automated PGO linkage checks

Native CI covers shared PGO, static PGO, plain static, AVX2-only and
ASan/UBSan builds. Both PGO jobs first build and run all contracts in
`DPDF_PGO_MODE=GENERATE`, including linking the optional kernel benchmarks.
This checks profiling-runtime linkage for direct backend-object tests and
static-library consumers. The final PGO release uses a separate build and
fresh training profiles, so test-generated profiles are not used for release.
CI retains CMake configuration and CTest logs for diagnosis.

### Per-model branch alignment

The accepted configuration aligns branches only for model 8. Model 2's
unconfirmed alignment path and the experimental alignment switches have
been removed. See the [result record](../validation/i7_7700t_final.json).

Model 8 uses GNU assembler's `-mbranches-within-32B-boundaries`; VNNI
sources use `-malign-branch-boundary=0` to preserve their accepted object code.
The assembler must support both options. This changes layout without adding
an instruction-set requirement. The selected benefit was measured on the
i7-7700T; it is not a claim of the same gain on every CPU.

For private microbenchmarks, additionally pass
`--cmake-arg=-DDPDF_BUILD_KERNEL_BENCHMARKS=ON`; these executables are not part
of the installed runtime.

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

The K64N64 channel-major path also follows the matrix's selected ISA.
`model2_cm_contract_vnni` and `model8_cm_contract_vnni` exercise the VNNI
path against row-major projection, including floating-point control modes,
guard pages, nonfinite/subnormal inputs and overlap rejection. They skip
with status 77 when AVX-VNNI is unavailable.

With Python available, `vnni_isa_contract` additionally disassembles both
backend object sets using the GCC toolchain's `objdump`. It rejects EVEX
instructions and verifies VEX `VPDPBUSD` encoding. This catches accidental
AVX-512 requirements that numerical tests on an AVX-512-capable CPU would
otherwise miss. Inline dot products must explicitly preserve this encoding.

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

## Ryzen factorial performance recheck

The [2026-11-10 recheck](../validation/ryzen_vnni_retest_20261110.json)
compares baseline, CM-only, quantize4-only and their combination with separate
fresh model-8 PGO profiles. The benchmark source, compiler flags and weights
are identical across the four variants; model 2 remains non-PGO.

CPU 3 was reserved through an isolated cpuset, its SMT sibling 15 was temporarily
offline, and movable IRQ/workqueue activity was redirected. Restoration was
verified after every phase. This excludes ordinary foreign userspace from the
worker, not all per-CPU kernel work, interrupts or shared-cache interference.

The unprofiled experiment retains 576 fresh-process records and all 633,600
individual call durations. Two randomized batches include balanced four-arm
blocks, two inputs, continuous and 10 ms paced execution, and A/A controls.
Report sample SD of process means separately from pooled per-call SD and SEM.
Use paired blocks, not individual calls, for effect intervals; preserve slow
observations and the preregistered multiplicity and A/A acceptance guards.

Hardware counting/sampling is a separate phase. FIFO control starts counters
after warmup; its timings must not be used as unprofiled latency evidence.
The record includes 144 PMU jobs with fully scheduled event groups and 512
exploratory microbench processes, including zero-row fallback and unchanged
row-path controls. Original raw artifacts and scripts are local under
`.local/retest-20261110/` at the repository root and are not bundled.

CM's continuous call-time reduction was confirmed on this high-frequency
Ryzen core: 3.637% for model 2 and 1.609% for model 8. Neither the sub-percent
quantize4 increment nor its addition to CM passed the predeclared A/A guard.
No gain was confirmed at a 10 ms cadence. These measurements do not replace
the earlier study, establish a WCET bound, or imply compact-core/other-CPU gains.
## Exact quantize4 refinement experiment (2026-11-10)

The range-reuse implementation is available for AVX2 and optional native VNNI.
Both experimental options default OFF: later studies confirm conditional gains,
but not global acceptance across models and workloads. See the [design and selection](QUANTIZE4_DESIGN.md)
and [aggregate evidence](../validation/quantize4_refinement_20261110.json).

From the repository root, configure an AVX2-only experimental build:

```sh
cmake -S optimized_inference -B build-q4-avx2 \
  -DCMAKE_BUILD_TYPE=Release -DDPDF_ENABLE_AVX_VNNI=OFF \
  -DDPDF_ENABLE_QUANTIZE4_REFINEMENT=ON
```

For the native experiment, use a separate directory and additionally enable
VNNI quantize4:

```sh
cmake -S optimized_inference -B build-q4-native \
  -DCMAKE_BUILD_TYPE=Release -DDPDF_ENABLE_QUANTIZE4_REFINEMENT=ON \
  -DDPDF_ENABLE_VNNI_QUANTIZE4=ON
```

Build and run CTest normally. Recreate model-8 PGO for each configuration;
do not reuse a profile from another option combination. The independent
`quantize4_refined_contract` and `quantize4_contract_vnni` check the candidate
even when the default library retains legacy quantization.

The short study used eight paired blocks per ISA/model/cadence, two A/A pairs,
160 fresh processes / 56,000 calls and 40 separate hardware jobs. It retained
all call durations, call/run-mean SD, tails and adjusted paired intervals.
Every graph interval included zero. Helper-only active/all-empty means improved
12.198%/22.431%, which is not a full-model speedup. CPU 3 isolation and parked
SMT sibling 15 were fully restored. A reduced 16-fixture/32-model-job stream
check passed; the full 65-fixture suite was not repeated in this short study.

The subsequent [isolated-L3 laboratory study](../validation/quantize4_lab_20261110.json)
used the common dlopen worker, Boost off, verified nominal frequency and worker
deep-idle restriction. It retained 200 unprofiled processes / 60,800 calls plus
40 separate hardware jobs. Only native model-8 batch passed the planned A/A and
effect gates: +0.832% time reduction, adjusted interval [0.580,1.084]%. This is
a controlled non-boost result; ordinary-power and paced acceptance remains
outstanding, and both experimental options still default OFF.

The [extended cadence/power matrix](../validation/quantize4_regimes_20261110.json)
then covers 48 conditions with family size48, eight A/A and eight comparative
pairs each: 1,536 unprofiled processes, 442,368 calls and 192 separate counter
jobs. It confirms model-8 VNNI steady batch16/batch64/paced10 and isolated-Boost
continuous, with time reductions 0.971%/0.769%/0.835%/0.739%. Boost-paced
acceptance remains unconfirmed; ordinary full-audio deployment is not measured.
All settings are restored after each profile. See BENCHMARK_LAB.md for custom
conditions, power profiles and recovery of a fully collected A/A journal.

## Final ready-binary qualification (2026-11-10)

The [final qualification](../validation/final_qualification_20261110.json)
covers both models on fast and compact Ryzen cores, steady and isolated Boost,
continuous spectral calls, paced10 spectral calls and a complete NumPy
FFT/inference/IFFT callback. One fixed plan retains 72 contrasts, 3,456 processes,
811,008 unprofiled calls, call/process SD, adjusted intervals, input/order splits
and 288 separate warmed PMU jobs. No slow observations were removed or sample
counts extended after inspecting results. Earlier layout/power cohorts were
not pooled.

CM remains selected for both models. On fast/steady continuous calls its time
reductions are 3.127% for model 2 and 1.429% for model 8; model 8 also passes
paced10 (1.472%) and complete audio (1.448%). Native quantize4 adds 0.810%
[0.683,0.937]% continuous and 0.863% [0.513,1.213]% paced10 for model 8 in that
environment. No quantize4 model-2, AVX2 or complete-audio condition passes every
superiority gate. Compact/Boost native model-8 continuous passes practical
noninferiority at the 0.5% loss margin, not useful superiority. Both options
remain OFF.

The complete regenerated 65-fixture numerical corpus passes 130 spectral/state
jobs and 130 PCM jobs on the PC; red passes 130 AVX2 portability jobs. All 130
cross-host diagnostic digests match. The unavailable historical reference
corpus is not covered. Red has no VNNI or separate housekeeping L3, so it is
not a strict isolated speed test; its eight service health checks remain 200.
Production device/audio-server/service-load latency is not measured.

Observed maximum call time is 2.787454 ms with zero 10 ms deadline misses and
zero major faults. Independent restoration verifies all CPUs online, original
power/idle/IRQ/workqueue/watchdog settings and irqbalance restored, no temporary
cgroup or workers, and global ASLR unchanged. Ready candidates additionally
pass 42 native and 32 AVX2 CTests; all 17 laboratory behavioral contracts pass.
Raw evidence/report/figures are in `.local/final-qualification-20261110/`.
No runtime default, commit, push or deployment is performed by this study.
