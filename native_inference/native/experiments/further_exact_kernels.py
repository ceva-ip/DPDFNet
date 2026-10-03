"""Guarded transforms for the accepted single-thread W7A8 source profile.

The further_optimization.py driver applies all five transforms to isolated
copies of the checked-in fitted W7A8 kernels. Reference sources are untouched.
"""


def once(code, old, new):
    if code.count(old) != 1:
        raise ValueError(f"Expected one occurrence, found {code.count(old)}: {old[:100]!r}")
    return code.replace(old, new, 1)


def byte_clamp(code):
    """Delay integer clamp to bytes; retains all float arithmetic and rounding.

    PACKSSDW followed by PACKUSWB implements min(max(q,0),255) for every
    signed i32 q, including INT_MIN returned by CVTPS2DQ on invalid input.
    An unsigned byte min with254 therefore exactly implements the old clamp.
    """
    code = once(code,
        '            q[t]=_mm256_min_epi32(_mm256_set1_epi32(254),_mm256_max_epi32(_mm256_setzero_si256(),q[t]));\n',
        '')
    code = once(code,
        '        __m256i p=_mm256_packus_epi16(_mm256_packs_epi32(q[0],q[1]),_mm256_packs_epi32(q[2],q[3]));',
        '        __m256i p=_mm256_packus_epi16(_mm256_packs_epi32(q[0],q[1]),_mm256_packs_epi32(q[2],q[3]));\n'
        '        /* The two packs saturate to [0,255]; one byte min gives [0,254]. */\n'
        '        p=_mm256_min_epu8(p,_mm256_set1_epi8((char)254));')
    code = once(code,
        '        q=_mm256_min_epi32(_mm256_set1_epi32(254),_mm256_max_epi32(_mm256_setzero_si256(),q));\n',
        '')
    code = once(code,
        '        p=_mm_packus_epi16(p,p); _mm_storel_epi64((__m128i *)(out+j),p);',
        '        p=_mm_packus_epi16(p,p);\n'
        '        p=_mm_min_epu8(p,_mm_set1_epi8((char)254));\n'
        '        _mm_storel_epi64((__m128i *)(out+j),p);')
    return code


def norm_transpose(code):
    """Load adjacent channels and transpose without altering reduction order."""
    old_mean = '''        for (int c=0;c<64;++c) {
            __m128 values=_mm_set_ps(x[192+c],x[128+c],x[64+c],x[c]);
            mean=_mm256_add_pd(mean,_mm256_cvtps_pd(values));
        }'''
    new_mean = '''        for (int c=0;c<64;c+=4) {
            __m128 v0=_mm_loadu_ps(x+c),v1=_mm_loadu_ps(x+64+c);
            __m128 v2=_mm_loadu_ps(x+128+c),v3=_mm_loadu_ps(x+192+c);
            _MM_TRANSPOSE4_PS(v0,v1,v2,v3);
            /* Add channels in their original increasing order in each lane. */
            mean=_mm256_add_pd(mean,_mm256_cvtps_pd(v0));
            mean=_mm256_add_pd(mean,_mm256_cvtps_pd(v1));
            mean=_mm256_add_pd(mean,_mm256_cvtps_pd(v2));
            mean=_mm256_add_pd(mean,_mm256_cvtps_pd(v3));
        }'''
    old_var = '''        for (int c=0;c<64;++c) {
            __m128 values=_mm_set_ps(x[192+c],x[128+c],x[64+c],x[c]);
            __m256d d=_mm256_sub_pd(_mm256_cvtps_pd(values),mean);
            var=_mm256_add_pd(var,_mm256_mul_pd(d,d));
        }'''
    new_var = '''        for (int c=0;c<64;c+=4) {
            __m128 v0=_mm_loadu_ps(x+c),v1=_mm_loadu_ps(x+64+c);
            __m128 v2=_mm_loadu_ps(x+128+c),v3=_mm_loadu_ps(x+192+c);
            _MM_TRANSPOSE4_PS(v0,v1,v2,v3);
            __m256d d=_mm256_sub_pd(_mm256_cvtps_pd(v0),mean);
            var=_mm256_add_pd(var,_mm256_mul_pd(d,d));
            d=_mm256_sub_pd(_mm256_cvtps_pd(v1),mean);
            var=_mm256_add_pd(var,_mm256_mul_pd(d,d));
            d=_mm256_sub_pd(_mm256_cvtps_pd(v2),mean);
            var=_mm256_add_pd(var,_mm256_mul_pd(d,d));
            d=_mm256_sub_pd(_mm256_cvtps_pd(v3),mean);
            var=_mm256_add_pd(var,_mm256_mul_pd(d,d));
        }'''
    return once(once(code, old_mean, new_mean), old_var, new_var)


def zero_initial_recurrent(code):
    """Skip the integer projection of the known initial all-zero GRU state.

    Each intra-frequency direction initializes h with memset(+0). On its first
    iteration, INT8 quantization yields activation0, scale1, zero-point0. Thus
    every corrected dot is integer0 and dequantization computes fma(+0,
    positive-finite-weight-scale,bias), exactly (+0+bias). Keep that addition
    instead of copying bias, including its signed-zero behavior. The FP32 and
    FP16 paths retain their original operations.
    """
    return once(code,
        '            affine(b, h, 24576+direction*64*192, bias+direction*384+192, step, 1, 64, 192);',
        '''#ifdef DPDF_X86_DISPATCH
            if (t==0 && b->q[0]) {
                const float *initial_bias=bias+direction*384+192;
                for (int c=0;c<192;++c) step[c]=0.0f+initial_bias[c];
            } else
#endif
                affine(b, h, 24576+direction*64*192, bias+direction*384+192, step, 1, 64, 192);''')


def aligned_weights(code):
    """Ensure packed matrix vectors never straddle a cache-line boundary.

    Packed columns start at multiples of32 bytes, and K is a multiple of8.
    A64-byte base alignment therefore aligns every32-byte weight load. Keep
    the malloc owner separately for normal destruction and account for all63
    padding bytes in the model-owned allocation estimate. No weights change.
    """
    code = once(code, '#include <stdlib.h>\n', '#include <stdlib.h>\n#include <stdint.h>\n')
    code = once(code, '    int8_t *packed;\n', '    int8_t *packed;\n    void *packed_allocation;\n')
    code = once(code, 'free(q->packed);', 'free(q->packed_allocation);')
    code = once(code,
        '    q->packed=malloc((size_t)k*n); q->scale=malloc(n*sizeof(float)); q->sum=calloc(n,sizeof(int32_t));',
        '    q->packed_allocation=malloc((size_t)k*n+63);\n'
        '    if (q->packed_allocation) q->packed=(int8_t *)(((uintptr_t)q->packed_allocation+63)&~(uintptr_t)63);\n'
        '    q->scale=malloc(n*sizeof(float)); q->sum=calloc(n,sizeof(int32_t));')
    return once(code,
        'sizeof(*q)+(size_t)q->k*q->n+(size_t)q->n*8',
        'sizeof(*q)+(size_t)q->k*q->n+63+(size_t)q->n*8')


def exp_nonpositive(code):
    """Remove the redundant upper exponent clamp only in the tanh path.

    tanh calls exp8 with -2*abs(x), which is nonpositive for every finite or
    infinite binary32 x (including signed-zero). The old MINPS(87,argument)
    returns argument unchanged on this range. If argument is a NaN, MINPS
    returns its second operand, also unchanged; the preceding multiplication
    already quiets a signaling NaN. The lower clamp, range reduction, fitted
    polynomial, reconstruction and final division all retain their arithmetic.
    Keep the generic exp8 unchanged for sigmoid, whose exponent may be positive.
    """
    start = code.index('static inline __m256 exp8(__m256 x) {')
    end = code.index('static inline __m256 sigmoid8(', start)
    original = code[start:end]
    specialized = once(original, 'exp8(__m256 x)', 'exp8_nonpositive(__m256 x)')
    specialized = once(specialized,
        '    x=_mm256_max_ps(_mm256_set1_ps(-87.0f),_mm256_min_ps(_mm256_set1_ps(87.0f),x));',
        '    x=_mm256_max_ps(_mm256_set1_ps(-87.0f),x);')
    code = code[:end]+specialized+code[end:]
    return once(code,
        '    __m256 e=exp8(_mm256_mul_ps(_mm256_set1_ps(-2),absolute));',
        '    __m256 e=exp8_nonpositive(_mm256_mul_ps(_mm256_set1_ps(-2),absolute));')


def optimize(source):
    """Apply the measured best_single combination in its original order."""
    path = source / 'int8.c'
    path.write_text(aligned_weights(byte_clamp(path.read_text())))
    path = source / 'avx2.c'
    path.write_text(exp_nonpositive(norm_transpose(path.read_text())))
    path = source / 'dpdf_dprnn.c'
    path.write_text(zero_initial_recurrent(path.read_text()))
