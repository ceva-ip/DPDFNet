"""Aggressive exact quantized candidates against the frozen Oct7 baseline.

Only isolated source copies are edited. The source quantizer, W7A8 grid,
integer dot bounds, FP32 scale product, bias FMA and caller FP controls are
retained. No additional allocation, threads or CPU instruction set is added.
"""

import hashlib
from pathlib import Path

VARIANTS = ('quant_fused64', 'quant_fused192', 'quant_fused192_init',
            'quant_inline64', 'quant_fused192_init_inline')
BASELINE_INT8_SHA256 = 'd28028618d67f435dbfa6e069395b843434294e6b6fae0b8cd4be24bb29ce437'
BASELINE_ASM_SHA256 = '0d317938030abada28343467bd0b27f178ffd1cefe00f25ef4ab04498cd62ae6'


def once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f'Expected one occurrence; found {code.count(old)}: {old[:100]!r}')
    return code.replace(old, new, 1)


def inline64(code):
    start = code.index('static void quantize(')
    end = code.index('\n#ifndef DPDF_DISABLE_QAFFINE_ROW', start)
    clone = code[start:end]
    clone = once(clone,
                 'static void quantize(const float *x,int8_t *out,int k,float *scale,int *zp) {',
                 'static inline __attribute__((always_inline,unused)) void quantize64(\n'
                 '        const float *x,int8_t *out,float *scale,int *zp) {\n'
                 '    const int k=64;')
    code = code[:end] + '\n' + clone + code[end:]
    start = code.index('static void qaffine_row(')
    end = code.index('\n#endif', start)
    # The quantizer call is before nested dispatch preprocessor blocks.
    block = once(code[start:end], '    quantize(x,activation,k,&activation_scale,&zp);',
                 '    if (k==64) quantize64(x,activation,&activation_scale,&zp);\n'
                 '    else quantize(x,activation,k,&activation_scale,&zp);')
    return code[:start] + block + code[end:]


def assembly(initial_correction):
    """SysV six-pointer/one-float/two-integer leaf; tile count is1 or3.

    RDI activation, RSI packed, RDX weight sums, RCX weight scales,
    R8 bias, R9 output; activation scale XMM0; zero point [RSP+8],
    tile count [RSP+16]. YMM14/15 retain zero point/activation scale.
    YMM0..7 are sums,8 word-ones,9 current activation/epilogue scratch,
    and10..13 independent multiply chains. No stack changes or saves.
    """
    code = '''/* Research-only Linux x86-64 SysV, AVX2/FMA, K=64.
 * Tile count1 or3; every output retains the original FP32 scale and FMA. */
.intel_syntax noprefix
.text
.p2align 5
.globl dpdf_qaffine64_fused_avx2
.hidden dpdf_qaffine64_fused_avx2
.type dpdf_qaffine64_fused_avx2,@function
dpdf_qaffine64_fused_avx2:
    .cfi_startproc
    vbroadcastss ymm15, xmm0
    mov eax, DWORD PTR [rsp+8]
    neg eax
    vmovd xmm14, eax
    vpbroadcastd ymm14, xmm14
    mov r10d, DWORD PTR [rsp+16]
    mov eax, 0x00010001
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
.p2align 5
.Ltile:
'''
    for accumulator in range(8):
        if initial_correction:
            code += f'    vpmulld ymm{accumulator}, ymm14, YMMWORD PTR [rdx+{accumulator*32}]\n'
        else:
            code += f'    vpxor ymm{accumulator}, ymm{accumulator}, ymm{accumulator}\n'
    for j in range(0, 64, 4):
        code += f'    vpbroadcastd ymm9, DWORD PTR [rdi+{j}]\n'
        for group in range(0, 8, 4):
            for t in range(4):
                offset = j*8+(group+t)*512
                code += f'    vpmaddubsw ymm{10+t}, ymm9, YMMWORD PTR [rsi+{offset}]\n'
            for t in range(4):
                code += f'    vpmaddwd ymm{10+t}, ymm{10+t}, ymm8\n'
                code += f'    vpaddd ymm{group+t}, ymm{group+t}, ymm{10+t}\n'
    for accumulator in range(8):
        offset = accumulator*32
        if not initial_correction:
            code += f'    vpmulld ymm9, ymm14, YMMWORD PTR [rdx+{offset}]\n'
            code += f'    vpaddd ymm{accumulator}, ymm{accumulator}, ymm9\n'
        code += f'    vcvtdq2ps ymm{accumulator}, ymm{accumulator}\n'
        code += f'    vmulps ymm9, ymm15, YMMWORD PTR [rcx+{offset}]\n'
        code += f'    vfmadd213ps ymm{accumulator}, ymm9, YMMWORD PTR [r8+{offset}]\n'
        code += f'    vmovups YMMWORD PTR [r9+{offset}], ymm{accumulator}\n'
    code += '''    add rsi, 4096
    add rdx, 256
    add rcx, 256
    add r8, 256
    add r9, 256
    dec r10d
    jnz .Ltile
    vzeroupper
    ret
    .cfi_endproc
.size dpdf_qaffine64_fused_avx2,.-dpdf_qaffine64_fused_avx2
.section .note.GNU-stack,"",@progbits
'''
    return code


def fused(code, source, entire192, initial_correction):
    (source/'qaffine64_fused_avx2.S').write_text(assembly(initial_correction))
    code = once(code,
                 'void dpdf_qdot64_fixed_avx2(const int8_t *,const int8_t *,int32_t *);',
                 'void dpdf_qdot64_fixed_avx2(const int8_t *,const int8_t *,int32_t *);\n'
                 '#ifdef DPDF_QFUSED64\n'
                 'void dpdf_qaffine64_fused_avx2(const int8_t *,const int8_t *,const int32_t *,\n'
                 '        const float *,const float *,float *,float,int,int);\n'
                 '#endif')
    start = code.index('static void qaffine_row(')
    # Insert only into this row path, preserving batch/paired kernels.
    end = code.index('\n/* Four rows and two independent', start)
    block = code[start:end]
    if entire192:
        block = once(block, '    const __m256i ones=_mm256_set1_epi16(1);',
                     '''#ifdef DPDF_QFUSED64
    if (k==64 && n==192) {
        dpdf_qaffine64_fused_avx2(activation,q->packed,q->sum,q->scale,bias,y,
                                activation_scale,zp,3);
        return;
    }
#endif
    const __m256i ones=_mm256_set1_epi16(1);''')
    block = once(block, '    for (;c+63<n;c+=64) {',
                 '''    for (;c+63<n;c+=64) {
#ifdef DPDF_QFUSED64
        if (k==64) {
            dpdf_qaffine64_fused_avx2(activation,q->packed+c*k,q->sum+c,q->scale+c,
                                    bias+c,y+c,activation_scale,zp,1);
            continue;
        }
#endif''')
    code = code[:start] + block + code[end:]
    contract = Path(__file__).with_name('fused_contract.c')
    (source/contract.name).write_text(contract.read_text())
    cmake_path = source/'CMakeLists.txt'
    cmake_path.write_text(cmake_path.read_text() + '''
# Oct7b exact fused row-dot; same Linux SysV gate as the retained assembler.
if(DPDF_ENABLE_AVX2 AND NOT WIN32 AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"
   AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  enable_language(ASM)
  target_sources(dpdf_kernels PRIVATE qaffine64_fused_avx2.S)
  target_compile_definitions(dpdf_kernels PRIVATE DPDF_QFUSED64)
  if(BUILD_TESTING)
    add_executable(dpdf_fused_contract fused_contract.c qaffine64_fused_avx2.S)
    target_compile_options(dpdf_fused_contract PRIVATE -Wall -Wextra -Werror
      -ffp-contract=off -frounding-math -fno-tree-vectorize)
    target_link_libraries(dpdf_fused_contract PRIVATE m)
    if(DPDF_SANITIZE)
      target_compile_options(dpdf_fused_contract PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
    add_test(NAME fused_contract COMMAND dpdf_fused_contract)
    set_tests_properties(fused_contract PROPERTIES TIMEOUT 30
      ENVIRONMENT "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:handle_segv=0;UBSAN_OPTIONS=halt_on_error=1")
  endif()
endif()
''')
    return code


def apply(name, source: Path):
    if name not in VARIANTS:
        raise ValueError(name)
    path = source/'int8.c'
    code = path.read_text()
    if hashlib.sha256(code.encode()).hexdigest() != BASELINE_INT8_SHA256:
        raise ValueError('Expected untouched frozen Oct7 combo_asm_norm int8.c')
    if hashlib.sha256((source/'qdot64_fixed_avx2.S').read_text().encode()).hexdigest() != BASELINE_ASM_SHA256:
        raise ValueError('Expected unchanged frozen Oct7 K64 assembly baseline')
    if name=='quant_inline64':
        path.write_text(inline64(code))
        return
    if name.endswith('_inline'):
        code = inline64(code)
    path.write_text(fused(code, source, entire192=name!='quant_fused64',
                          initial_correction='_init' in name))
