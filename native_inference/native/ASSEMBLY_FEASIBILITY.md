# Optional assembly kernels for Linux

**Latest:** [exact latency and memory optimization](LATENCY_REWORK.md) reaches
layout conversion, temporary storage, normalization, and batched INT8 kernels.
It uses portable C and the existing dispatched AVX2 tier; no assembly or newer
instruction set is required.

**Implemented follow-up:** [AVX2 register-lifetime optimization](AVX2_OPTIMIZATION.md)
compares revised intrinsics with a handwritten assembly prototype. The selected
runtime change uses intrinsics and preserves the existing CPU requirements.

Investigated 2026-09-20, after reading the native inference README and the
architecture/optimization report. This is a feasibility and compiler-output
audit, not an implemented kernel or a new latency result.

## Decision

**Yes.** Small, optional assembly kernels can coexist with a general C runtime.
Linux narrows the platform ABI; it does not imply any particular CPU instruction
set. Keep model execution, state, allocation and the public C/Rust boundary in
C. Select ISA-specific kernels once during context creation, with a baseline
implementation available. Specialize arithmetic shapes rather than model IDs.

The repository's pinned HushMic investigation identifies Linux x86-64 as the
initial deployment target. ARM64 would require its own implementation or the
portable path; x86 assembly cannot be reused directly there. This audit does
not revalidate the current upstream HushMic release matrix.

The existing implementation already supplies much of the structure:

- `CMakeLists.txt` isolates AVX2/FMA/F16C compiler flags to specialized files.
- `dpdf_dprnn.c` detects features and chooses scalar or AVX2 FP32 functions.
- `int8.c` implements packed, shape-parameterized affine operations using
  intrinsics; these already compile to SIMD machine instructions.

One important boundary: the existing scalar fallback is **FP32**. Explicit
INT8 and FP16 creation currently rejects unsupported CPUs. An optional AVX2
assembly implementation can fall back to the existing AVX2 intrinsics without
changing precision. Supporting the same INT8 mode on older x86/ARM CPUs would
need a portable integer implementation, or an explicit application policy to
select FP32. Do not silently describe FP32 as bit-identical INT8 fallback.

## First finding: a concrete AVX2 assembly candidate

The highest-priority candidate is `qaffine_row()` in `int8.c`, especially
`M=1, K=64, N=192` for sequential intra-GRU recurrence. The architecture report
counts 1,408 such recurrent projections per DPDFNet-8 hop. Its existing
64-output tile has eight independent vector accumulators.

On the available i7-8700, Linux Docker/WSL2, GCC
`12.2.0-14+deb12u1`, a fresh release-style compilation of the current source
confirmed two accumulators stored and reloaded in every inner-loop iteration.
The generated `.L43` loop includes:

```asm
vpaddd  ymm5, ymm1, YMMWORD PTR 224[rsp]
vpaddd  ymm2, ymm2, YMMWORD PTR 192[rsp]
vmovdqa YMMWORD PTR 224[rsp], ymm5
vmovdqa YMMWORD PTR 192[rsp], ymm2
```

It also stores one result to `160[rsp]` during that loop. These are actual
stack accesses inside the reduction, not merely function prologue saves.
The same pattern was present in the existing `opt_int8_tile` shared library.

A plausible assembly schedule uses eight YMM accumulators, one activation,
one absolute activation, one vector of word-ones and two temporary vectors:
13 of the 16 YMM registers. Process weight vectors in small groups instead
of keeping many weight/product temporaries live at once. Defer scale and
zero-point vectors until the epilogue. Start with the current packing, which
avoids changing model conversion or weight memory.

This establishes an optimization opportunity, **not a speedup**. More rigid
scheduling can lose instruction overlap; function calls and setup also cost
time. Compare against a rescheduled intrinsic implementation and the existing
kernel. A compiler change may remove the spills without assembly. Retain
handwritten code only if it improves full-model timing consistently.

## Ranked directions

| Priority | Direction | Applicability and constraint |
| --- | --- | --- |
| 1 | AVX2 one-row INT8 kernel with explicit register scheduling | Measurable on this host; target the observed spills; preserve quantization and FP32 epilogue exactly. |
| 2 | AVX-VNNI 256-bit INT8 dot products | Optional newer-CPU tier; this host does not expose VNNI. Intrinsics can access the instructions too. |
| 3 | Four-row and paired-direction INT8 tiles | Audit their generated loops and measure separately after the one-row path; different register/reuse tradeoffs. |
| 4 | ARM64 NEON/dot-product backend | Future architecture coverage; needs ARM hardware validation and feature detection. |

VNNI can replace the multiply-pair / word-reduction / accumulator-add sequence
with a non-saturating `VPDPBUSD` dot product. One first implementation can retain
the current absolute-activation/sign-transfer scheme and the existing
`(127-zp)*weight_sum` correction. Another can represent activation values as
unsigned `u=a+127` and use `dot(u,w)-zp*weight_sum`. The latter removes sign
transfer but requires deliberately changing the activation representation and
proving equivalent rounding and correction. Do not merely reinterpret the
current signed activation bytes as unsigned.

The AVX2 pair-sum bound is `2*127*127=32258`, below signed 16-bit saturation.
For the current supported `K<=512`, integer dot sums and correction also fit
INT32. Preserve those bounds and the exact conversion/scale/FMA epilogue.
AVX-VNNI and AVX-512 VNNI are separate dispatch capabilities; neither follows
from AVX2 support. No VNNI performance claim can be made on this machine.

FP32/FP16 affine assembly is lower priority for this first experiment because
the verified compiler deficiency is in INT8. LayerNorm/gate approximations,
new quantization, and threading are separate changes with different numerical
or CPU-resource implications. Avoid combining them with this exact-kernel test.

## Integration and acceptance

Use an optional Linux x86-64 `.S` translation unit behind a private C signature.
Enable CMake's ASM language only for that supported target. Preserve the SysV
ABI, callee-saved registers, position independence, stack alignment, unwind
metadata and a non-executable stack declaration. Keep ISA detection code
compiled for the baseline CPU, and avoid global `-march=native` in release
artifacts. Continue to use HushMic's documented glibc build floor.

Keep quantization/packing in C initially. A kernel should process a whole
affine row or sufficiently large tile so call overhead does not erase gains.
Handle unsupported shapes through the existing kernel. Select implementation
independently of precision and expose a test-only way to force each available
backend. Do not switch precision or backend midstream without an explicit
state/semantics policy.

Before adoption:

1. Test signed extremes, zeros, mixed values, supported dimensions, output
   tails and deliberately unaligned buffers against independent scalar integer
   arithmetic. Add explicit `M=1` coverage: the current `int8_contract.c` calls
   `M=7`, exercising the batch implementation rather than `qaffine_row()`.
2. Require identical output bytes and complete recurrent-state bytes against
   the preserved INT8 implementation on both supported models, including
   independent streams. Preserve arithmetic order in the FP32 epilogue.
3. Use guarded buffers/canaries and ABI-preservation checks for assembly.
   ASan/UBSan remain useful for C wrappers but do not instrument handwritten
   assembly memory accesses. Verify disabled-assembly and scalar-only builds.
4. Benchmark the kernel and then both whole models in alternating baseline /
   candidate order, continuously and at 10 ms cadence. Report p50/p95/p99/max
   and deadline exceedances. Validate on Intel and AMD before claiming broad
   performance benefit; run VNNI and ARM candidates on appropriate hardware.

## Reproduce the compiler audit

From the repository root in PowerShell, using the existing development image:

```powershell
$auditMount = "type=bind,source=$((Resolve-Path native_inference).Path),target=/bench"
docker run --rm --network none --mount $auditMount --entrypoint sh dpdfnet-native-dev -c 'mkdir -p /bench/scratch/assembly_audit && cc -O3 -DNDEBUG -fPIC -Wall -Wextra -Werror -ffp-contract=off -mavx2 -mfma -DDPDF_X86_DISPATCH -S -masm=intel /bench/native/int8.c -o /bench/scratch/assembly_audit/int8.gcc12.s'
```

The generated listing is ignored under `scratch/assembly_audit/`. The audited
`int8.c` SHA-256 was
`2e5f19a6d990a54dad64c3f6e71a2cd591d61e1ed61e302bd64c792e88010732`.
Compilation passed with warnings treated as errors. No runtime implementation
was changed and no new kernel benchmark or numerical-parity run was performed.

## References

- Local [architecture and prior measurements](DPDFNET8_ARCHITECTURE.md).
- Local [deployment and portability investigation](../FEASIBILITY.md).
- [GCC x86 feature detection](https://gcc.gnu.org/onlinedocs/gcc/x86-Built-in-Functions.html): separate baseline detection from ISA-specific compilation.
- [Intel optimization manual, section 8.2.1](https://cdrdv2-public.intel.com/821612/248966-Optimization-Reference-Manual-V1-050.pdf): VNNI dot products and the replaced instruction sequence.
- [Intel ISA extensions reference](https://cdrdv2-public.intel.com/790021/architecture-instruction-set-extensions-programming-reference.pdf): instruction semantics and feature distinctions.
- [Arm dot-product instructions](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/dot-neoverse-n1-accelerating-dsp-functions-with-the-dot-instructions): SDOT/UDOT as a separate ARM backend direction.
