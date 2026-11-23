/* Experimental exact four-row refinement; gated until full-graph confirmation. */
/* Keep range reduction and packing identical for scalar and four-row paths.
 * Constant K wrappers let GCC specialize these private inline helpers. */
static inline void quantize_range(const float *x,int k,float *low,float *high) {
    __m256 lo=_mm256_setzero_ps(),hi=lo;
    for (int j=0;j<k;j+=8) { __m256 v=_mm256_loadu_ps(x+j); lo=_mm256_min_ps(lo,v); hi=_mm256_max_ps(hi,v); }
    /* Reduce in registers. The scalar reference starts at zero and ignores
     * unordered comparisons; a zero second operand preserves that behavior. */
    lo=_mm256_min_ps(lo,_mm256_setzero_ps());
    hi=_mm256_max_ps(hi,_mm256_setzero_ps());
    __m128 low4=_mm_min_ps(_mm256_castps256_ps128(lo),_mm256_extractf128_ps(lo,1));
    __m128 high4=_mm_max_ps(_mm256_castps256_ps128(hi),_mm256_extractf128_ps(hi,1));
    low4=_mm_min_ps(low4,_mm_movehl_ps(low4,low4));
    high4=_mm_max_ps(high4,_mm_movehl_ps(high4,high4));
    *low=_mm_cvtss_f32(_mm_min_ss(low4,_mm_shuffle_ps(low4,low4,0x55)));
    *high=_mm_cvtss_f32(_mm_max_ss(high4,_mm_shuffle_ps(high4,high4,0x55)));
}

static inline void quantize_pack(const float *x,int8_t *out,int k,float inverse,int zp) {
    __m256 inv=_mm256_set1_ps(inverse),z=_mm256_set1_ps((float)zp);
    int j=0;
    for (;j+31<k;j+=32) {
        __m256i q[4];
        for (int t=0;t<4;++t) {
            __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j+t*8),inv),z);
            q[t]=_mm256_cvtps_epi32(v);
        }
        __m256i p=_mm256_packus_epi16(_mm256_packs_epi32(q[0],q[1]),_mm256_packs_epi32(q[2],q[3]));
        /* The two packs saturate to [0,255]; one byte min gives [0,254]. */
        p=_mm256_min_epu8(p,_mm256_set1_epi8((char)254));
        /* AVX2 packs within 128-bit lanes: reorder four-byte groups. */
        p=_mm256_permutevar8x32_epi32(p,_mm256_setr_epi32(0,4,1,5,2,6,3,7));
        _mm256_storeu_si256((__m256i *)(out+j),p);
    }
    for (;j<k;j+=8) {
        __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j),inv),z);
        __m256i q=_mm256_cvtps_epi32(v);
        __m128i p=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
        p=_mm_packus_epi16(p,p);
        p=_mm_min_epu8(p,_mm_set1_epi8((char)254));
        _mm_storel_epi64((__m128i *)(out+j),p);
    }
}

static inline void quantize_from_range(const float *x,int8_t *out,int k,
        float low,float high,float *scale,int *zp) {
    if (low==high) { memset(out,0,k); *scale=1; *zp=0; return; }
    *scale=(high-low)/254.0f;
    *zp=(int)nearbyintf(-low / *scale);
    if (*zp<0) *zp=0;
    if (*zp>254) *zp=254;
    quantize_pack(x,out,k,1.0f / *scale,*zp);
}

static void quantize(const float *x,int8_t *out,int k,float *scale,int *zp) {
    float low,high;
    quantize_range(x,k,&low,&high);
    quantize_from_range(x,out,k,low,high,scale,zp);
}

static inline __attribute__((always_inline,unused)) void quantize64(
        const float *x,int8_t *out,float *scale,int *zp) {
    float low,high;
    quantize_range(x,64,&low,&high);
    quantize_from_range(x,out,64,low,high,scale,zp);
}

#if !defined(DPDF_INT8_VNNI) || defined(DPDF_VNNI_QUANTIZE4)
/* Each range is computed once. Empty means low==high with zero included,
 * not a nonzero constant row, nor an active row whose scale underflows. */
static void quantize4_64(const float *input,int8_t *output,float *scales,int *zps) {
    float lows[4],highs[4],inverses[4];
    for (int row=0;row<4;++row)
        quantize_range(input+row*64,64,lows+row,highs+row);
    __m128 l=_mm_loadu_ps(lows),h=_mm_loadu_ps(highs);
    __m128 empty=_mm_cmpeq_ps(l,h);
    int mask=_mm_movemask_ps(empty);
    if (mask==15) {
        memset(output,0,256);
        _mm_storeu_ps(scales,_mm_set1_ps(1.0f));
        _mm_storeu_si128((__m128i *)zps,_mm_setzero_si128());
        return;
    }
    __m128 span=_mm_sub_ps(h,l);
    if (mask) {
        /* Neutralize before division: empty lanes use 254/254, -0/1, 1/1.
         * Blending results after a 0/0 would still raise invalid. */
        l=_mm_andnot_ps(empty,l);
        span=_mm_blendv_ps(span,_mm_set1_ps(254.0f),empty);
    }
    __m128 s=_mm_div_ps(span,_mm_set1_ps(254.0f));
    _mm_storeu_ps(scales,s);
    __m128 negative_low=_mm_xor_ps(l,_mm_set1_ps(-0.0f));
    __m128i points=_mm_cvtps_epi32(_mm_div_ps(negative_low,s));
    points=_mm_min_epi32(_mm_set1_epi32(254),_mm_max_epi32(_mm_setzero_si128(),points));
    _mm_storeu_si128((__m128i *)zps,points);
    _mm_storeu_ps(inverses,_mm_div_ps(_mm_set1_ps(1.0f),s));
    if (!mask) {
        for (int row=0;row<4;++row)
            quantize_pack(input+row*64,output+row*64,64,inverses[row],zps[row]);
    } else {
        for (int row=0;row<4;++row) {
            if (mask & (1<<row)) memset(output+row*64,0,64);
            else quantize_pack(input+row*64,output+row*64,64,inverses[row],zps[row]);
        }
    }
}
#endif
