"""Isolated exact W7A8 AVX2 scheduling/packing candidates for Oct 7.

Apply to a source snapshot of the accepted Oct 3 best_single profile. This
module does not build, benchmark, or modify any checked-in runtime source.
All candidates retain the quantizer, integer values, correction, FP32 scale
products, bias FMA, and fitted GRU gates. Packed alternate weights consume
additional memory that dpdf_qbytes reports explicitly.
"""

import hashlib
from pathlib import Path

VARIANTS = ('quant_2x32', 'quant_broadcast', 'quant_blocked_row', 'quant_asm64',
            'quant_recurrent_layout')
BASELINE_INT8_SHA256 = 'd9f24ebf71107fa68ac6665cf5244e9c8987abe12c3d286f9c0f4d7f2c6e1f30'


def once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f'Expected one occurrence of {old[:90]!r}; found {code.count(old)}')
    return code.replace(old, new, 1)


def replace_function(code, start_text, following, replacement):
    start = code.index(start_text)
    end = code.index(following, start)
    return code[:start] + replacement + '\n\n' + code[end:]


def two_by_four(code):
    """Two rows x four output vectors: eight integer accumulators.

    The previous four-row x two-vector tile repeats four activation
    broadcasts per K step. This tile repeats two, with four weight loads;
    total memory instructions stay six. Saturating pair products retain the
    W7A8 bound 2*254*63=32004; complete sums remain safely in signed i32.
    Dimensions outside the new tile retain the original implementation.
    """
    kernel = '''/* Oct7: two rows and four independent weight vectors. */
static __attribute__((noinline)) void qdot2quad(const int8_t *activation,
        const int8_t *p0,const int8_t *p1,const int8_t *p2,const int8_t *p3,
        int k,__m256i *out) {
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
    __m256i b0=a0,b1=a0,b2=a0,b3=a0;
    for (int j=0;j<k;j+=4) {
        int32_t v0,v1;
        memcpy(&v0,activation+j,4); memcpy(&v1,activation+k+j,4);
        __m256i x0=_mm256_set1_epi32(v0),x1=_mm256_set1_epi32(v1);
#define DPDF_TWO_ROWS(tile,sa,sb) do { \\
        __m256i w=_mm256_loadu_si256((const __m256i *)(p##tile+j*8)); \\
        sa=_mm256_add_epi32(sa,_mm256_madd_epi16(_mm256_maddubs_epi16(x0,w),ones)); \\
        sb=_mm256_add_epi32(sb,_mm256_madd_epi16(_mm256_maddubs_epi16(x1,w),ones)); \\
    } while (0)
        DPDF_TWO_ROWS(0,a0,b0); DPDF_TWO_ROWS(1,a1,b1);
        DPDF_TWO_ROWS(2,a2,b2); DPDF_TWO_ROWS(3,a3,b3);
#undef DPDF_TWO_ROWS
    }
    out[0]=a0;out[1]=b0;out[2]=a1;out[3]=b1;
    out[4]=a2;out[5]=b2;out[6]=a3;out[7]=b3;
}

'''
    at = code.index('static void qaffine_batch_tiled(')
    code = code[:at] + kernel + code[at:]
    batch = '''static void qaffine_batch_2x32(const dpdf_qmatrix *q,const float *x,const float *bias,float *y,int m) {
    int8_t activation[48*512]; float scales[48]; int zp[48];
    const int k=q->k,n=q->n;
    for (int r=0;r<m;++r) quantize(x+r*k,activation+r*k,k,scales+r,zp+r);
    for (int r=0;r<m;r+=2) for (int c=0;c<n;c+=32) {
        __m256i sums[8];
        qdot2quad(activation+r*k,q->packed+c*k,q->packed+(c+8)*k,
                 q->packed+(c+16)*k,q->packed+(c+24)*k,k,sums);
        for (int tile=0;tile<4;++tile) for (int t=0;t<2;++t) {
            int col=c+tile*8;
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+col)),_mm256_set1_epi32(-zp[r+t]));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r+t]),_mm256_loadu_ps(q->scale+col));
            __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t+tile*2],correction)),scale,_mm256_loadu_ps(bias+col));
            _mm256_storeu_ps(y+(r+t)*n+col,result);
        }
    }
}
'''
    at = code.index('static void qaffine_batch(')
    code = code[:at] + batch + '\n' + code[at:]
    code = once(code, '    if (m%4==0 && q->n%16==0) { qaffine_batch_tiled(q,x,bias,y,m); return; }',
                '    if (m%2==0 && q->n%32==0) { qaffine_batch_2x32(q,x,bias,y,m); return; }\n'
                '    if (m%4==0 && q->n%16==0) { qaffine_batch_tiled(q,x,bias,y,m); return; }')
    pair = '''    if (q0->n%16==0) {
        for (int r=0;r<m;r+=2) for (int c=0;c<n;c+=16) {
            __m256i sums[8];
            qdot2quad(activation+r*k,q0->packed+c*k,q0->packed+(c+8)*k,
                     q1->packed+c*k,q1->packed+(c+8)*k,k,sums);
            for (int matrix=0;matrix<2;++matrix) for (int tile=0;tile<2;++tile) {
                const dpdf_qmatrix *q=matrix?q1:q0;
                const float *bias=matrix?bias1:bias0; float *y=matrix?y1:y0;
                int col=c+tile*8;
                __m256i weight_sum=_mm256_loadu_si256((const __m256i *)(q->sum+col));
                __m256 weight_scale=_mm256_loadu_ps(q->scale+col);
                for (int t=0;t<2;++t) {
                    __m256i correction=_mm256_mullo_epi32(weight_sum,_mm256_set1_epi32(-zp[r+t]));
                    __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r+t]),weight_scale);
                    __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t+tile*2+matrix*4],correction)),scale,_mm256_loadu_ps(bias+col));
                    _mm256_storeu_ps(y+(r+t)*n+col,result);
                }
            }
        }
        return;
    }
'''
    start = code.index('void dpdf_qaffine_pair(')
    pair_code = code[start:]
    at = pair_code.index('    for (int r=0;r<m;r+=4) for (int c=0;c<n;c+=8) {')
    return code[:start] + pair_code[:at] + pair + pair_code[at:]


def broadcast_blocks(code):
    """Prepare four-row activation broadcasts once per output-column sweep.

    Only the four currently processed rows are expanded, bounded by 16 KiB
    at K=512 and 2 KiB for common K=64. All other dimensions/tails retain the
    original implementation. No long-lived model allocation is added.
    """
    kernel = '''/* Oct7: reuse already expanded activations across column tiles. */
static void broadcast_four(const int8_t *activation,int k,__m256i *expanded) {
    for (int j=0;j<k;j+=4) for (int row=0;row<4;++row) {
        int32_t bytes; memcpy(&bytes,activation+row*k+j,4);
        expanded[j+row]=_mm256_set1_epi32(bytes);
    }
}
static __attribute__((noinline)) void qdot4pair_expanded(const __m256i *activation,
        const int8_t *packed0,const int8_t *packed1,int k,__m256i *out) {
    const __m256i ones=_mm256_set1_epi16(1);
    __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;
    __m256i b0=a0,b1=a0,b2=a0,b3=a0;
    for (int j=0;j<k;j+=4) {
        __m256i w0=_mm256_loadu_si256((const __m256i *)(packed0+j*8));
        __m256i w1=_mm256_loadu_si256((const __m256i *)(packed1+j*8));
#define DPDF_EXPANDED_ROW(row,sa,sb) do { \\
        __m256i v=activation[j+row]; \\
        sa=_mm256_add_epi32(sa,_mm256_madd_epi16(_mm256_maddubs_epi16(v,w0),ones)); \\
        sb=_mm256_add_epi32(sb,_mm256_madd_epi16(_mm256_maddubs_epi16(v,w1),ones)); \\
    } while (0)
        DPDF_EXPANDED_ROW(0,a0,b0); DPDF_EXPANDED_ROW(1,a1,b1);
        DPDF_EXPANDED_ROW(2,a2,b2); DPDF_EXPANDED_ROW(3,a3,b3);
#undef DPDF_EXPANDED_ROW
    }
    out[0]=a0;out[1]=a1;out[2]=a2;out[3]=a3;
    out[4]=b0;out[5]=b1;out[6]=b2;out[7]=b3;
}

'''
    at = code.index('static void qaffine_batch_tiled(')
    code = code[:at] + kernel + code[at:]
    for start_text, end_text, old_call, new_call, columns in (
            ('static void qaffine_batch_tiled(', 'static void qaffine_batch(',
             'qdot4pair(activation+r*k,q->packed+c*k,q->packed+(c+8)*k,k,sums);',
             'qdot4pair_expanded(expanded,q->packed+c*k,q->packed+(c+8)*k,k,sums);', 16),
            ('void dpdf_qaffine_pair(', None,
             'qdot4pair(activation+r*k,q0->packed+c*k,q1->packed+c*k,k,sums);',
             'qdot4pair_expanded(expanded,q0->packed+c*k,q1->packed+c*k,k,sums);', 8)):
        start = code.index(start_text)
        end = code.index(end_text, start) if end_text else len(code)
        block = code[start:end]
        block = once(block, f'    for (int r=0;r<m;r+=4) for (int c=0;c<n;c+={columns}) {{',
                     '    __m256i expanded[512];\n'
                     '    for (int r=0;r<m;r+=4) {\n'
                     '      broadcast_four(activation+r*k,k,expanded);\n'
                     f'      for (int c=0;c<n;c+={columns}) {{')
        block = once(block, old_call, new_call)
        # The added row block encloses the original column body.
        end_brace = block.rfind('}')
        block = block[:end_brace] + '    }\n' + block[end_brace:]
        code = code[:start] + block + code[end:]
    # All original qdot4pair callers above were replaced; retain that function
    # for fallback dimensions by deliberately marking it possibly unused.
    code = once(code, 'static __attribute__((noinline)) void qdot4pair(',
                'static __attribute__((noinline,unused)) void qdot4pair(')
    return code


def blocked_recurrent(code):
    """Add K-major 64-column packing only to common K=64,N=192 matrices.

    Original packing remains available for batch and odd dimensions. A row
    dot traverses each 64-column tile as one sequential 256-byte K step,
    rather than eight streams separated by 512 bytes. Every duplicate packed
    byte comes directly from the original rounded integer; scales are shared.
    The extra allocation and padding are included in dpdf_qbytes and freed.
    """
    code = once(code, '    void *packed_allocation;\n',
                '    void *packed_allocation;\n    int8_t *row_packed;\n    void *row_packed_allocation;\n')
    code = once(code, 'free(q->packed_allocation);',
                'free(q->packed_allocation); free(q->row_packed_allocation);')
    code = once(code, '    for (int c=0;c<n;++c) {',
                '''    if (k==64 && n==192) {
        q->row_packed_allocation=malloc((size_t)k*n+63);
        if (!q->row_packed_allocation) { dpdf_qdestroy(q); return NULL; }
        q->row_packed=(int8_t *)(((uintptr_t)q->row_packed_allocation+63)&~(uintptr_t)63);
    }
    for (int c=0;c<n;++c) {''')
    code = once(code,
                '            q->packed[(c/8)*k*8+(j/4)*32+(c%8)*4+j%4]=(int8_t)v;',
                '            q->packed[(c/8)*k*8+(j/4)*32+(c%8)*4+j%4]=(int8_t)v;\n'
                '            if (q->row_packed) q->row_packed[(c/64)*k*64+(j/4)*256+((c%64)/8)*32+(c%8)*4+j%4]=(int8_t)v;')
    code = once(code, 'sizeof(*q)+(size_t)q->k*q->n+63+(size_t)q->n*8',
                'sizeof(*q)+(size_t)q->k*q->n+63+(size_t)q->n*8+(q->row_packed?(size_t)q->k*q->n+63:0)')
    start = code.index('static __attribute__((noinline)) void qdot64(')
    end = code.index('\nstatic void qaffine_row', start)
    clone = code[start:end].replace('void qdot64(', 'void qdot64_blocked(')
    clone = once(clone, 'const int8_t *base=packed+j*8;', 'const int8_t *base=packed+j*64;')
    clone = once(clone, '(base+k*8)', '(base+32)')
    for tile in range(2, 8):
        clone = once(clone, f'(base+{tile}*k*8)', f'(base+{tile*32})')
    code = code[:end] + '\n' + clone + code[end:]
    return once(code, '        qdot64(activation,q->packed+c*k,k,integer_sums);',
                '        if (q->row_packed) qdot64_blocked(activation,q->row_packed+c*k,k,integer_sums);\n'
                '        else qdot64(activation,q->packed+c*k,k,integer_sums);')


def assembly_fixed64(code, source):
    """Fully unrolled private SysV K=64 row-dot with four-product scheduling.

    This experiment removes dynamic K/weight-stride addresses and branches,
    while leaving all correction/dequantization in the original C caller.
    Only caller-saved registers are used; every weight load is memory-folded
    into maddubs, and four product temporaries hold independent chains.
    The original intrinsic path remains for other shapes/platforms.
    """
    assembly = '''/* Research-only Linux x86-64 SysV; K=64, output=64 i32.
 * W7A8 direct unsigned activation and signed seven-bit weights. */
.intel_syntax noprefix
.text
.p2align 5
.globl dpdf_qdot64_fixed_avx2
.hidden dpdf_qdot64_fixed_avx2
.type dpdf_qdot64_fixed_avx2,@function
dpdf_qdot64_fixed_avx2:
    .cfi_startproc
'''
    for accumulator in range(8):
        assembly += f'    vpxor ymm{accumulator}, ymm{accumulator}, ymm{accumulator}\n'
    assembly += '''    mov eax, 0x00010001
    vmovd xmm8, eax
    vpbroadcastd ymm8, xmm8
'''
    for j in range(0, 64, 4):
        assembly += f'    vpbroadcastd ymm9, DWORD PTR [rdi+{j}]\n'
        for group in range(0, 8, 4):
            for t in range(4):
                offset = j*8 + (group+t)*512
                assembly += f'    vpmaddubsw ymm{10+t}, ymm9, YMMWORD PTR [rsi+{offset}]\n'
            for t in range(4):
                assembly += f'    vpmaddwd ymm{10+t}, ymm{10+t}, ymm8\n'
                assembly += f'    vpaddd ymm{group+t}, ymm{group+t}, ymm{10+t}\n'
    for accumulator in range(8):
        assembly += f'    vmovdqu YMMWORD PTR [rdx+{accumulator*32}], ymm{accumulator}\n'
    assembly += '''    vzeroupper
    ret
    .cfi_endproc
.size dpdf_qdot64_fixed_avx2,.-dpdf_qdot64_fixed_avx2
.section .note.GNU-stack,"",@progbits
'''
    (source / 'qdot64_fixed_avx2.S').write_text(assembly)
    cmake_path = source / 'CMakeLists.txt'
    cmake_path.write_text(cmake_path.read_text() + '''
# Oct7 optional private assembly experiment, exact intrinsic fallback.
if(DPDF_ENABLE_AVX2 AND NOT WIN32 AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64"
   AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  enable_language(ASM)
  target_sources(dpdf_kernels PRIVATE qdot64_fixed_avx2.S)
  target_compile_definitions(dpdf_kernels PRIVATE DPDF_QASM64)
endif()
''')
    code = once(code, '#ifndef DPDF_DISABLE_QAFFINE_ROW\n/* Keep integer reduction',
                '#ifndef DPDF_DISABLE_QAFFINE_ROW\n#ifdef DPDF_QASM64\n'
                'void dpdf_qdot64_fixed_avx2(const int8_t *,const int8_t *,int32_t *);\n'
                '#endif\n/* Keep integer reduction')
    return once(code, '        qdot64(activation,q->packed+c*k,k,integer_sums);',
                '#ifdef DPDF_QASM64\n'
                '        if (k==64) dpdf_qdot64_fixed_avx2(activation,q->packed+c*k,integer_sums);\n'
                '        else\n#endif\n'
                '            qdot64(activation,q->packed+c*k,k,integer_sums);')


def recurrent_layout(code, source):
    """Repack only the two sequential intra-GRU matrices at creation.

    The original packed allocation is retained. A 12,288-byte creation-only
    scratch allocation reorders its already quantized bytes and is freed
    immediately. No inference allocation or duplicate weights are retained.
    All private affine entry points accept a prepared matrix by executing
    individual rows, including paired/multi-row defensive fallbacks.
    """
    code = once(code, '    int k,n;\n', '    int k,n;\n    int row_layout;\n')
    prepare = '''/* Oct7: creation-only repacking for sequential intra-GRU recurrence. */
int dpdf_qprepare_row_layout(dpdf_qmatrix *q) {
    if (!q || q->k!=64 || q->n!=192) return -1;
    if (q->row_layout) return 0;
    int8_t *temporary=malloc((size_t)q->k*q->n);
    if (!temporary) return -1;
    for (int c=0;c<q->n;++c) for (int j=0;j<q->k;++j)
        temporary[(c/64)*q->k*64+(j/4)*256+((c%64)/8)*32+(c%8)*4+j%4]=
            q->packed[(c/8)*q->k*8+(j/4)*32+(c%8)*4+j%4];
    memcpy(q->packed,temporary,(size_t)q->k*q->n);
    free(temporary); q->row_layout=1;
    return 0;
}

'''
    at = code.index('static void quantize(')
    code = code[:at] + prepare + code[at:]
    start = code.index('static __attribute__((noinline)) void qdot64(')
    end = code.index('\nstatic void qaffine_row', start)
    clone = code[start:end].replace('void qdot64(', 'void qdot64_recurrent_layout(')
    clone = once(clone, 'const int8_t *base=packed+j*8;', 'const int8_t *base=packed+j*64;')
    clone = once(clone, '(base+k*8)', '(base+32)')
    for tile in range(2, 8):
        clone = once(clone, f'(base+{tile}*k*8)', f'(base+{tile*32})')
    row = '''static void qaffine_recurrent_layout_row(const dpdf_qmatrix *q,
        const float *x,const float *bias,float *y) {
    int8_t activation[64]; float activation_scale; int zp;
    quantize(x,activation,64,&activation_scale,&zp);
    for (int c=0;c<192;c+=64) {
        int32_t integer_sums[64];
        qdot64_recurrent_layout(activation,q->packed+c*64,64,integer_sums);
        for (int t=0;t<8;++t) {
            int column=c+t*8;
            __m256i sum=_mm256_loadu_si256((const __m256i *)(integer_sums+t*8));
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+column)),_mm256_set1_epi32(-zp));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(activation_scale),_mm256_loadu_ps(q->scale+column));
            _mm256_storeu_ps(y+column,_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum,correction)),scale,_mm256_loadu_ps(bias+column)));
        }
    }
}

'''
    # Keep this path available even when the legacy row kernel is disabled.
    at = code.index('#ifndef DPDF_DISABLE_QAFFINE_ROW\n')
    code = code[:at] + clone + '\n\n' + row + code[at:]
    code = once(code,
                'void dpdf_qaffine(const dpdf_qmatrix *q,const float *x,const float *bias,float *y,int m) {',
                '''void dpdf_qaffine(const dpdf_qmatrix *q,const float *x,const float *bias,float *y,int m) {
    if (q->row_layout) {
        for (int r=0;r<m;++r) qaffine_recurrent_layout_row(q,x+r*q->k,bias,y+r*q->n);
        return;
    }''')
    code = once(code, '    if (q0->k!=q1->k || q0->n!=q1->n || m%4) {',
                '    if (q0->row_layout || q1->row_layout || q0->k!=q1->k || q0->n!=q1->n || m%4) {')
    header_path = source / 'internal.h'
    header_path.write_text(once(header_path.read_text(),
            'dpdf_qmatrix *dpdf_qcreate(const float *, int, int);',
            'dpdf_qmatrix *dpdf_qcreate(const float *, int, int);\n'
            'int dpdf_qprepare_row_layout(dpdf_qmatrix *);'))
    block_path = source / 'dpdf_dprnn.c'
    block_path.write_text(once(block_path.read_text(),
            '        if (!b->q[i]) { dpdf_destroy(b); return NULL; }',
            '        if (!b->q[i]) { dpdf_destroy(b); return NULL; }\n'
            '        if ((i==2 || i==3) && dpdf_qprepare_row_layout(b->q[i])!=0) { dpdf_destroy(b); return NULL; }'))
    contract_source = Path(__file__).with_name('recurrent_layout_contract.c')
    (source / contract_source.name).write_text(contract_source.read_text())
    cmake_path = source / 'CMakeLists.txt'
    cmake_path.write_text(cmake_path.read_text() + '''
# Private prepared-layout API regression, in addition to the standard oracle.
if(BUILD_TESTING AND DPDF_ENABLE_AVX2 AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  add_executable(dpdf_recurrent_layout_contract recurrent_layout_contract.c)
  target_compile_definitions(dpdf_recurrent_layout_contract PRIVATE DPDF_X86_DISPATCH)
  target_compile_options(dpdf_recurrent_layout_contract PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
  target_link_libraries(dpdf_recurrent_layout_contract PRIVATE dpdf_dprnn)
  if(NOT WIN32)
    target_link_libraries(dpdf_recurrent_layout_contract PRIVATE m)
  endif()
  if(DPDF_SANITIZE)
    target_compile_options(dpdf_recurrent_layout_contract PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  endif()
  add_test(NAME recurrent_layout_contract COMMAND dpdf_recurrent_layout_contract)
  set_tests_properties(recurrent_layout_contract PROPERTIES TIMEOUT 30
    ENVIRONMENT "ASAN_OPTIONS=halt_on_error=1:abort_on_error=1:handle_segv=0;UBSAN_OPTIONS=halt_on_error=1")
endif()
''')
    return code


def apply(name, source: Path):
    if name not in VARIANTS:
        raise ValueError(name)
    path = source / 'int8.c'
    code = path.read_text()
    if hashlib.sha256(code.encode()).hexdigest() != BASELINE_INT8_SHA256:
        raise ValueError('Expected untouched Oct3 best_single int8.c snapshot')
    if name == 'quant_asm64':
        path.write_text(assembly_fixed64(code, source))
        return
    if name == 'quant_recurrent_layout':
        path.write_text(recurrent_layout(code, source))
        return
    transform = {'quant_2x32': two_by_four, 'quant_broadcast': broadcast_blocks,
                 'quant_blocked_row': blocked_recurrent}[name]
    path.write_text(transform(code))
