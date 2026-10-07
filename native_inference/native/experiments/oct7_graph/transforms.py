"""Guarded exact DPRNN buffer transforms for the October 7 investigation.

Apply to isolated copies of the accepted October 3 sources. These transforms
only move where existing arithmetic reads or writes; they introduce no new
rounding, activation approximation, precision mode, state or worker thread.
"""


def once(code, old, new):
    count = code.count(old)
    if count != 1:
        raise ValueError(f"Expected one source occurrence, found {count}: {old[:100]!r}")
    return code.replace(old, new, 1)


def block_io(code):
    """Borrow frequency-major input and normalize directly to matching output.

    The original first LayerNorm is the final consumer of the input residual.
    Consequently the final norm can safely overwrite that same input buffer.
    Channel-major inputs still transpose into x and channel-major outputs still
    transpose from x. Input/output and state/state_out in-place cases remain
    supported. The operation order and all norm/affine/gate calls are unchanged.
    """
    code = once(code,
        "    if (layout_flags & DPDF_INPUT_FREQ_MAJOR)\n"
        "        memcpy(x,input,(size_t)f*64*sizeof(float));\n"
        "    else\n",
        "    const float *features=(layout_flags & DPDF_INPUT_FREQ_MAJOR)?input:x;\n"
        "    if (!(layout_flags & DPDF_INPUT_FREQ_MAJOR))\n")
    code = once(code,
        "dpdf_qaffine_pair(b->q[0],b->q[1],x,bias,bias+384,a,hproj,f)",
        "dpdf_qaffine_pair(b->q[0],b->q[1],features,bias,bias+384,a,hproj,f)")
    code = once(code,
        "affine(b, x, direction*64*192, bias+direction*384, intra[direction], f, 64, 192)",
        "affine(b, features, direction*64*192, bias+direction*384, intra[direction], f, 64, 192)")
    code = once(code,
        "b->norm(mid, p, x, nis, nib, b->eps[0], f)",
        "b->norm(mid, p, features, nis, nib, b->eps[0], f)")
    code = once(code,
        "    b->norm(x, p, mid, nos, nob, b->eps[1], f);\n"
        "    if (layout_flags & DPDF_OUTPUT_FREQ_MAJOR)\n"
        "        memcpy(output,x,(size_t)f*64*sizeof(float));\n"
        "    else\n",
        "    float *final=(layout_flags & DPDF_OUTPUT_FREQ_MAJOR)?output:x;\n"
        "    b->norm(final, p, mid, nos, nob, b->eps[1], f);\n"
        "    if (!(layout_flags & DPDF_OUTPUT_FREQ_MAJOR))\n")
    return code


def intra_store(code):
    """Write intra-GRU outputs to bi directly, then use that row as old state.

    Each row and direction occupies a disjoint 64-float bi slice. During the
    ordered recurrence the previous row is already initialized and the next
    row has not been written. Positive-zero initial h is retained, including
    the accepted W7A8 first-step +0+bias projection behavior.
    """
    code = once(code,
        "    float h[64], step[192];",
        "    float initial_h[64], step[192];")
    code = once(code,
        "        memset(h, 0, sizeof(h));",
        "        memset(initial_h, 0, sizeof(initial_h));\n"
        "        const float *h=initial_h;")
    code = once(code,
        "            b->gates(intra[direction]+r*192, step, h, h, 1);\n"
        "            memcpy(bi+r*128+direction*64, h, sizeof(h));",
        "            float *next_h=bi+r*128+direction*64;\n"
        "            b->gates(intra[direction]+r*192, step, h, next_h, 1);\n"
        "            h=next_h;")
    return code


def block_copies(code):
    """Compose the independent input/output and intra-state copy transforms."""
    return intra_store(block_io(code))


DEPTHWISE_STRIDE = r'''
/* Exact specialization for the model's depthwise 1x3 stride-two/three
 * downsamplers. The original sw!=1 path uses separate binary32 multiply/add,
 * so these vectors deliberately do not fuse. Padding and odd-width tails
 * use the original ordered scalar terms; vector loads stay inside input. */
int dpdf_depthwise_stride_avx2(const float *x,const float *w,const float *bias,float *y,
        int ci,int hi,int wi,int co,int ho,int wo,int kh,int kw,
        int sh,int sw,int ph,int pw,int group) {
    if (ci<=0 || group<=0 || hi!=1 || ho!=1 || kh!=1 || kw!=3 || sh!=1 || ph!=0 || pw!=1 ||
        group!=ci || co%group || (sw!=2 && sw!=3) || wi<8 ||
        wo!=(wi+2*pw-kw)/sw+1) return 0;
    const __m256i offsets=_mm256_setr_epi32(0,sw,2*sw,3*sw,4*sw,5*sw,6*sw,7*sw);
    int last=(wi+pw-kw)/sw+1;
    if (last>wo) last=wo;
    for (int oc=0;oc<co;++oc) {
        const float *in=x+(oc/(co/group))*wi;
        const float *weight=w+oc*3;
        float *out=y+oc*wo;
        float initial=bias?bias[oc]:0;
        int ow=0;
        while (ow<wo) {
            if (ow>=1 && ow+7<last) {
                __m256 sum=_mm256_set1_ps(initial);
                for (int kx=0;kx<3;++kx) {
                    const float *p=in+ow*sw+kx-pw;
                    __m256 values=LOAD_STRIDED_VALUES;
                    sum=_mm256_add_ps(sum,_mm256_mul_ps(_mm256_set1_ps(weight[kx]),values));
                }
                _mm256_storeu_ps(out+ow,sum);
                ow+=8;
            } else {
                float sum=initial;
                for (int kx=0;kx<3;++kx) {
                    int ix=ow*sw+kx-pw;
                    if (ix>=0 && ix<wi) sum+=weight[kx]*in[ix];
                }
                out[ow++]=sum;
            }
        }
    }
    return 1;
}
'''


def depthwise_stride(source, deinterleave=False):
    """Separate-multiply/add AVX2 stride kernel with scalar padded fringes.

    The gather variant handles both strides directly. The deinterleave variant
    replaces stride-two gathers with four bounded contiguous SSE loads; it
    retains gathers for stride three. At stride two the last needed element
    is p[14], but a full contiguous load accesses p[15], so that alternative
    checks one additional input element before issuing those loads.
    """
    kernel = DEPTHWISE_STRIDE.replace("LOAD_STRIDED_VALUES", "_mm256_i32gather_ps(p,offsets,4)")
    if deinterleave:
        kernel = once(kernel,
            "                    __m256 values=_mm256_i32gather_ps(p,offsets,4);",
            "                    __m256 values;\n"
            "                    if (sw==2 && ow*sw+kx-pw+15<wi) {\n"
            "                        __m128 lo=_mm_shuffle_ps(_mm_loadu_ps(p),_mm_loadu_ps(p+4),\n"
            "                                                  _MM_SHUFFLE(2,0,2,0));\n"
            "                        __m128 hi=_mm_shuffle_ps(_mm_loadu_ps(p+8),_mm_loadu_ps(p+12),\n"
            "                                                  _MM_SHUFFLE(2,0,2,0));\n"
            "                        values=_mm256_insertf128_ps(_mm256_castps128_ps256(lo),hi,1);\n"
            "                    } else values=_mm256_i32gather_ps(p,offsets,4);")
    path = source / "avx2.c"
    code = path.read_text()
    code = once(code, "/* Input-major, output-contiguous weights. Four rows share each loaded tile.",
                kernel + "\n/* Input-major, output-contiguous weights. Four rows share each loaded tile.")
    path.write_text(code)
    path = source / "full_ops.h"
    code = once(path.read_text(), "int dpdf_conv_row_avx2(",
                "int dpdf_depthwise_stride_avx2(const float *, const float *, const float *, float *,\n"
                "                      int, int, int, int, int, int, int, int, int, int, int, int, int);\n"
                "int dpdf_conv_row_avx2(")
    path.write_text(code)
    path = source / "full_ops.c"
    code = once(path.read_text(), "#ifdef DPDF_X86_DISPATCH\n    if (axpy==dpdf_axpy_avx2",
                "#ifdef DPDF_X86_DISPATCH\n"
                "    if (axpy==dpdf_axpy_avx2 && sw>1 &&\n"
                "        dpdf_depthwise_stride_avx2(x,w,bias,y,ci,hi,wi,co,ho,wo,\n"
                "                                   kh,kw,sh,sw,ph,pw,group)) return;\n"
                "    if (axpy==dpdf_axpy_avx2")
    path.write_text(code)


def norm8(source):
    """Use two independent ordered four-row FP64 normalization chains.

    Each row still adds channels 0..63 in precisely their original order.
    Mean, centered-square variance, square root/division and output arithmetic
    retain their original precision and operation order. The second chain can
    overlap the first chain's dependent additions. Retain the original four-row
    implementation for contract widths not divisible by eight.
    """
    path = source / "avx2.c"
    code = path.read_text()
    start = code.index("void dpdf_norm_residual_avx2(")
    end = code.index("/* Exact 8x8 transpose:", start)
    original = code[start:end]
    helper = once(original, "void dpdf_norm_residual_avx2(",
                  "static void dpdf_norm_residual_four_avx2(")
    new = '''void dpdf_norm_residual_avx2(float *out,const float *p,const float *skip,
        const float *scale,const float *bias,float eps,int rows) {
    int r=0;
    for (;r+7<rows;r+=8) {
        __m256d mean0=_mm256_setzero_pd(),mean1=mean0;
        __m256d var0=mean0,var1=mean0;
        const float *x=p+r*64;
        for (int c=0;c<64;c+=4) {
            __m128 v0=_mm_loadu_ps(x+c),v1=_mm_loadu_ps(x+64+c);
            __m128 v2=_mm_loadu_ps(x+128+c),v3=_mm_loadu_ps(x+192+c);
            __m128 w0=_mm_loadu_ps(x+256+c),w1=_mm_loadu_ps(x+320+c);
            __m128 w2=_mm_loadu_ps(x+384+c),w3=_mm_loadu_ps(x+448+c);
            _MM_TRANSPOSE4_PS(v0,v1,v2,v3);
            _MM_TRANSPOSE4_PS(w0,w1,w2,w3);
'''
    for c in range(4):
        new += (f"            mean0=_mm256_add_pd(mean0,_mm256_cvtps_pd(v{c}));\n"
                f"            mean1=_mm256_add_pd(mean1,_mm256_cvtps_pd(w{c}));\n")
    new += '''        }
        mean0=_mm256_mul_pd(mean0,_mm256_set1_pd(1.0/64));
        mean1=_mm256_mul_pd(mean1,_mm256_set1_pd(1.0/64));
        for (int c=0;c<64;c+=4) {
            __m128 v0=_mm_loadu_ps(x+c),v1=_mm_loadu_ps(x+64+c);
            __m128 v2=_mm_loadu_ps(x+128+c),v3=_mm_loadu_ps(x+192+c);
            __m128 w0=_mm_loadu_ps(x+256+c),w1=_mm_loadu_ps(x+320+c);
            __m128 w2=_mm_loadu_ps(x+384+c),w3=_mm_loadu_ps(x+448+c);
            _MM_TRANSPOSE4_PS(v0,v1,v2,v3);
            _MM_TRANSPOSE4_PS(w0,w1,w2,w3);
'''
    for c in range(4):
        new += (f"            __m256d d{c}=_mm256_sub_pd(_mm256_cvtps_pd(v{c}),mean0);\n"
                f"            __m256d e{c}=_mm256_sub_pd(_mm256_cvtps_pd(w{c}),mean1);\n"
                f"            var0=_mm256_add_pd(var0,_mm256_mul_pd(d{c},d{c}));\n"
                f"            var1=_mm256_add_pd(var1,_mm256_mul_pd(e{c},e{c}));\n")
    new += '''        }
        __m256d inv0=_mm256_div_pd(_mm256_set1_pd(1),_mm256_sqrt_pd(
            _mm256_add_pd(_mm256_mul_pd(var0,_mm256_set1_pd(1.0/64)),_mm256_set1_pd(eps))));
        __m256d inv1=_mm256_div_pd(_mm256_set1_pd(1),_mm256_sqrt_pd(
            _mm256_add_pd(_mm256_mul_pd(var1,_mm256_set1_pd(1.0/64)),_mm256_set1_pd(eps))));
        float means[8],invs[8];
        _mm_storeu_ps(means,_mm256_cvtpd_ps(mean0));
        _mm_storeu_ps(means+4,_mm256_cvtpd_ps(mean1));
        _mm_storeu_ps(invs,_mm256_cvtpd_ps(inv0));
        _mm_storeu_ps(invs+4,_mm256_cvtpd_ps(inv1));
        for (int t=0;t<8;++t) for (int c=0;c<64;c+=8) {
            int offset=(r+t)*64+c;
            __m256 value=_mm256_sub_ps(_mm256_loadu_ps(p+offset),_mm256_set1_ps(means[t]));
            value=_mm256_mul_ps(value,_mm256_set1_ps(invs[t]));
            value=_mm256_mul_ps(value,_mm256_loadu_ps(scale+c));
            value=_mm256_add_ps(value,_mm256_loadu_ps(bias+c));
            value=_mm256_add_ps(value,_mm256_loadu_ps(skip+offset));
            _mm256_storeu_ps(out+offset,value);
        }
    }
    if (r<rows) dpdf_norm_residual_four_avx2(out+r*64,p+r*64,skip+r*64,
                                          scale,bias,eps,rows-r);
}

'''
    path.write_text(code[:start]+helper+new+code[end:])


TRANSFORMS = {"block_io": block_io, "intra_store": intra_store,
              "block_copies": block_copies}
VARIANTS = (*TRANSFORMS, "depthwise_stride", "depthwise_deinterleave", "norm8")


def apply(name, source):
    if name == "norm8":
        norm8(source)
        return
    if name in ("depthwise_stride", "depthwise_deinterleave"):
        depthwise_stride(source, name == "depthwise_deinterleave")
        return
    path = source / "dpdf_dprnn.c"
    path.write_text(TRANSFORMS[name](path.read_text()))
