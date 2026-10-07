"""Fixed-width, integer-exact four-row/two-tile AVX2 assembly experiments."""
import hashlib
from pathlib import Path


VARIANTS=('quant_pair64','quant_pair64_96')
PAIR_BLOCK_SHA256='5e1c9f9b49931b4aa16261e864bd4121c07b5f4116f1eab6fb069e991564b0fb'


def assembly(widths):
    code='''/* Research-only Linux x86-64 SysV, AVX2, unsigned A8 / signed W7.
 * RDI four activation rows; RSI/RDX two packed eight-column weight tiles;
 * RCX output: tile0 rows0..3, then tile1 rows0..3. No stack or callee saves. */
.intel_syntax noprefix
.text
'''
    for k in widths:
        symbol=f'dpdf_qdot4pair{k}_avx2'
        code+=f'''.p2align 5
.globl {symbol}
.hidden {symbol}
.type {symbol},@function
{symbol}:
    .cfi_startproc
    mov eax, 0x00010001
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
'''
        for accumulator in range(8):
            code+=f'    vpxor ymm{accumulator}, ymm{accumulator}, ymm{accumulator}\n'
        for j in range(0,k,4):
            code+=f'    vmovdqu ymm9, YMMWORD PTR [rsi+{j*8}]\n'
            code+=f'    vmovdqu ymm10, YMMWORD PTR [rdx+{j*8}]\n'
            for row in range(4):
                code+=f'    vpbroadcastd ymm{12+row}, DWORD PTR [rdi+{row*k+j}]\n'
            for tile in range(2):
                for row in range(4):
                    code+=f'    vpmaddubsw ymm11, ymm{12+row}, ymm{9+tile}\n'
                    code+='    vpmaddwd ymm11, ymm11, ymm8\n'
                    accumulator=tile*4+row
                    code+=f'    vpaddd ymm{accumulator}, ymm{accumulator}, ymm11\n'
        for accumulator in range(8):
            code+=f'    vmovdqu YMMWORD PTR [rcx+{accumulator*32}], ymm{accumulator}\n'
        code+=f'''    vzeroupper
    ret
    .cfi_endproc
.size {symbol},.-{symbol}
'''
    return code+'.section .note.GNU-stack,"",@progbits\n'


def apply(name,source):
    if name not in VARIANTS:
        raise ValueError(name)
    source=Path(source)
    path=source/'int8.c'
    code=path.read_text()
    start=code.index('static __attribute__((noinline)) void qdot4pair(')
    end=code.index('static void qaffine_batch_tiled(',start)
    block=code[start:end]
    if hashlib.sha256(block.encode()).hexdigest()!=PAIR_BLOCK_SHA256:
        raise RuntimeError('Fixed-pair experiment requires unchanged Oct7 qdot4pair block')
    widths=(64,96) if name=='quant_pair64_96' else (64,)
    generic=block.replace('void qdot4pair(', 'void qdot4pair_generic(',1)
    wrapper='#ifdef DPDF_QPAIR_FIXED\n'
    for k in widths:
        wrapper+=f'void dpdf_qdot4pair{k}_avx2(const int8_t *,const int8_t *,const int8_t *,__m256i *);\n'
    wrapper+='#endif\n'
    wrapper+='''static __attribute__((noinline)) void qdot4pair(const int8_t *activation,
        const int8_t *packed0,const int8_t *packed1,int k,__m256i *out) {
#ifdef DPDF_QPAIR_FIXED
'''
    for k in widths:
        wrapper+=f'    if (k=={k}) {{ dpdf_qdot4pair{k}_avx2(activation,packed0,packed1,out); return; }}\n'
    wrapper+='''#endif
    qdot4pair_generic(activation,packed0,packed1,k,out);
}

'''
    path.write_text(code[:start]+generic+wrapper+code[end:])
    (source/'qdot4pair_fixed_avx2.S').write_text(assembly(widths))
    contract=Path(__file__).with_name('pair_contract.c')
    (source/contract.name).write_text(contract.read_text())
    cmake=source/'CMakeLists.txt'
    fragment='''
# Oct7b exact fixed-K four-row/two-tile assembly; other platforms retain C.
if(DPDF_ENABLE_AVX2 AND NOT WIN32 AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"
   AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  enable_language(ASM)
  target_sources(dpdf_kernels PRIVATE qdot4pair_fixed_avx2.S)
  target_compile_definitions(dpdf_kernels PRIVATE DPDF_QPAIR_FIXED)
  if(BUILD_TESTING)
    add_executable(dpdf_pair_contract pair_contract.c qdot4pair_fixed_avx2.S)
    target_compile_options(dpdf_pair_contract PRIVATE -Wall -Wextra -Werror -fno-tree-vectorize -ffp-contract=off)
'''
    if len(widths)==2:
        fragment+='    target_compile_definitions(dpdf_pair_contract PRIVATE DPDF_PAIR_TEST96)\n'
    fragment+='''    if(DPDF_SANITIZE)
      target_compile_options(dpdf_pair_contract PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    endif()
    add_test(NAME pair_contract COMMAND dpdf_pair_contract)
    set_tests_properties(pair_contract PROPERTIES TIMEOUT 30
      ENVIRONMENT "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:handle_segv=0;UBSAN_OPTIONS=halt_on_error=1")
  endif()
endif()
'''
    cmake.write_text(cmake.read_text()+fragment)
