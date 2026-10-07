"""One exact K256/N768 fused assembly candidate for external GRU projections.

Apply to a frozen Oct7 copy, optionally after an oct7b_quant fused64/inline64
transform. All other dimensions and batch paths retain the accepted fallback.
The quantizer, packed layout, stored weights, integer correction, separate FP32
scale product and bias FMA are unchanged. This is Linux x86-64 SysV only.
"""
import hashlib
from pathlib import Path

from oct7b_quant import transforms as q64

VARIANTS = ('quant_fused256',)

INLINE_SIGNATURE = 'static inline __attribute__((always_inline,unused)) void quantize64(\n'
INLINE_HEADER = (INLINE_SIGNATURE + '        const float *x,int8_t *out,float *scale,int *zp) {\n'
                 '    const int k=64;')
ORIGINAL_QUANTIZER = 'static void quantize(const float *x,int8_t *out,int k,float *scale,int *zp) {'
INLINE_CALL = ('    if (k==64) quantize64(x,activation,&activation_scale,&zp);\n'
               '    else quantize(x,activation,k,&activation_scale,&zp);')
ORIGINAL_CALL = '    quantize(x,activation,k,&activation_scale,&zp);'
FUSED64_DECLARATION = ('\n#ifdef DPDF_QFUSED64\n'
    'void dpdf_qaffine64_fused_avx2(const int8_t *,const int8_t *,const int32_t *,\n'
    '        const float *,const float *,float *,float,int,int);\n#endif')
FUSED192_DISPATCH = '''#ifdef DPDF_QFUSED64
    if (k==64 && n==192) {
        dpdf_qaffine64_fused_avx2(activation,q->packed,q->sum,q->scale,bias,y,
                                activation_scale,zp,3);
        return;
    }
#endif
'''
FUSED64_TILE_DISPATCH = '''
#ifdef DPDF_QFUSED64
        if (k==64) {
            dpdf_qaffine64_fused_avx2(activation,q->packed+c*k,q->sum+c,q->scale+c,
                                    bias+c,y+c,activation_scale,zp,1);
            continue;
        }
#endif'''


def guarded_original(code, source):
    """Remove only known exact preceding transforms and hash the remainder."""
    original = code
    if INLINE_SIGNATURE in original:
        start = original.index(ORIGINAL_QUANTIZER)
        end = original.index('\n}\n', start) + len('\n}\n')
        clone = q64.once(original[start:end], ORIGINAL_QUANTIZER, INLINE_HEADER)
        original = q64.once(original, '\n' + clone, '')
        original = q64.once(original, INLINE_CALL, ORIGINAL_CALL)
    if FUSED64_DECLARATION in original:
        original = q64.once(original, FUSED64_DECLARATION, '')
        original = q64.once(original, FUSED64_TILE_DISPATCH, '')
        if FUSED192_DISPATCH in original:
            original = q64.once(original, FUSED192_DISPATCH, '')
        asm = (source / 'qaffine64_fused_avx2.S').read_text()
        if asm not in (q64.assembly(False), q64.assembly(True)):
            raise ValueError('Unexpected preceding fused64 assembly')
    if hashlib.sha256(original.encode()).hexdigest() != q64.BASELINE_INT8_SHA256:
        raise ValueError('Expected frozen Oct7 int8.c or only known oct7b_quant changes')
    if hashlib.sha256((source / 'qdot64_fixed_avx2.S').read_text().encode()).hexdigest() != q64.BASELINE_ASM_SHA256:
        raise ValueError('Expected unchanged frozen Oct7 K64 assembly')


def assembly():
    """Six pointers + scalar float + stack zp/tile count, same fused64 ABI.

    Callers pass positive tile counts; native dispatch passes exactly12.
    RDI activation, RSI packed, RDX sums, RCX scales, R8 bias, R9 output;
    XMM0 activation scale, [RSP+8] zero point, [RSP+16] tile count.
    Only caller-saved RAX/R10/R11 and YMM0..15 are modified. No stack stores.
    """
    code = '''/* Research-only Linux x86-64 SysV, AVX2/FMA, fixed K=256.
 * Eight independent output vectors per tile; native N768 dispatch uses12 tiles.
 * Weight grid [-63,63], unsigned activation grid [0,254]. */
.intel_syntax noprefix
.text
.p2align 5
.globl dpdf_qaffine256_fused_avx2
.hidden dpdf_qaffine256_fused_avx2
.type dpdf_qaffine256_fused_avx2,@function
dpdf_qaffine256_fused_avx2:
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
.Lwide_tile:
'''
    for accumulator in range(8):
        code += f'    vpxor ymm{accumulator}, ymm{accumulator}, ymm{accumulator}\n'
    code += '''    xor r11d, r11d
.p2align 5
.Lwide_k:
    vpbroadcastd ymm9, DWORD PTR [rdi+r11]
'''
    for group in range(0, 8, 4):
        for t in range(4):
            code += f'    vpmaddubsw ymm{10+t}, ymm9, YMMWORD PTR [rsi+r11*8+{(group+t)*2048}]\n'
        for t in range(4):
            code += f'    vpmaddwd ymm{10+t}, ymm{10+t}, ymm8\n'
            code += f'    vpaddd ymm{group+t}, ymm{group+t}, ymm{10+t}\n'
    code += '''    add r11d, 4
    cmp r11d, 256
    jb .Lwide_k
'''
    for accumulator in range(8):
        offset = accumulator * 32
        code += f'    vpmulld ymm9, ymm14, YMMWORD PTR [rdx+{offset}]\n'
        code += f'    vpaddd ymm{accumulator}, ymm{accumulator}, ymm9\n'
        code += f'    vcvtdq2ps ymm{accumulator}, ymm{accumulator}\n'
        # Multiplication must round separately before the original bias FMA.
        code += f'    vmulps ymm9, ymm15, YMMWORD PTR [rcx+{offset}]\n'
        code += f'    vfmadd213ps ymm{accumulator}, ymm9, YMMWORD PTR [r8+{offset}]\n'
        code += f'    vmovups YMMWORD PTR [r9+{offset}], ymm{accumulator}\n'
    code += '''    add rsi, 16384
    add rdx, 256
    add rcx, 256
    add r8, 256
    add r9, 256
    dec r10d
    jnz .Lwide_tile
    vzeroupper
    ret
    .cfi_endproc
.size dpdf_qaffine256_fused_avx2,.-dpdf_qaffine256_fused_avx2
.section .note.GNU-stack,"",@progbits
'''
    return code


def apply(name, source: Path):
    if name not in VARIANTS:
        raise ValueError(name)
    path = source / 'int8.c'
    code = path.read_text()
    guarded_original(code, source)
    code = q64.once(code,
        'void dpdf_qdot64_fixed_avx2(const int8_t *,const int8_t *,int32_t *);',
        'void dpdf_qdot64_fixed_avx2(const int8_t *,const int8_t *,int32_t *);\n'
        '#ifdef DPDF_QFUSED256\n'
        'void dpdf_qaffine256_fused_avx2(const int8_t *,const int8_t *,const int32_t *,\n'
        '        const float *,const float *,float *,float,int,int);\n#endif')
    start = code.index('static void qaffine_row(')
    end = code.index('\n/* Four rows and two independent', start)
    row = code[start:end]
    position = row.index('    const __m256i ones=_mm256_set1_epi16(1);')
    existing_dispatch = row.find('#ifdef DPDF_QFUSED64')
    if 0 <= existing_dispatch < position:
        position = existing_dispatch
    row = row[:position] + '''#ifdef DPDF_QFUSED256
    if (k==256 && n==768) {
        dpdf_qaffine256_fused_avx2(activation,q->packed,q->sum,q->scale,bias,y,
                                 activation_scale,zp,12);
        return;
    }
#endif
''' + row[position:]
    code = code[:start] + row + code[end:]
    path.write_text(code)
    (source / 'qaffine256_fused_avx2.S').write_text(assembly())
    contract = Path(__file__).with_name('wide_contract.c')
    (source / contract.name).write_text(contract.read_text())
    cmake = source / 'CMakeLists.txt'
    cmake.write_text(cmake.read_text() + '''
# Oct7b exact external-GRU row projection; accepted intrinsic fallbacks remain.
if(DPDF_ENABLE_AVX2 AND NOT WIN32 AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"
   AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  enable_language(ASM)
  target_sources(dpdf_kernels PRIVATE qaffine256_fused_avx2.S)
  target_compile_definitions(dpdf_kernels PRIVATE DPDF_QFUSED256)
  if(BUILD_TESTING)
    add_executable(dpdf_wide_contract wide_contract.c qaffine256_fused_avx2.S)
    target_compile_definitions(dpdf_wide_contract PRIVATE DPDF_FUSED_K=256
      DPDF_FUSED_MAX_N=768 DPDF_FUSED_MAX_TILES=12
      DPDF_FUSED_FUNCTION=dpdf_qaffine256_fused_avx2)
    target_compile_options(dpdf_wide_contract PRIVATE -Wall -Wextra -Werror
      -ffp-contract=off -frounding-math -fno-tree-vectorize)
    target_link_libraries(dpdf_wide_contract PRIVATE m)
    if(DPDF_SANITIZE)
      target_compile_options(dpdf_wide_contract PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
    add_test(NAME wide_contract COMMAND dpdf_wide_contract)
    set_tests_properties(wide_contract PROPERTIES TIMEOUT 90
      ENVIRONMENT "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:handle_segv=0;UBSAN_OPTIONS=halt_on_error=1")
  endif()
endif()
''')
