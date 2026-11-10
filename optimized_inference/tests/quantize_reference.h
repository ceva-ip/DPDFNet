/* Frozen pre-refinement SIMD oracle from 8ad669a; intentionally independent. */
static void reference_quantize(const float *x,int8_t *out,int k,float *scale,int *zp) {
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
    float low=_mm_cvtss_f32(_mm_min_ss(low4,_mm_shuffle_ps(low4,low4,0x55)));
    float high=_mm_cvtss_f32(_mm_max_ss(high4,_mm_shuffle_ps(high4,high4,0x55)));
    if (low==high) { memset(out,0,k); *scale=1; *zp=0; return; }
    *scale=(high-low)/254.0f;
    *zp=(int)nearbyintf(-low / *scale);
    if (*zp<0) *zp=0;
    if (*zp>254) *zp=254;
    __m256 inv=_mm256_set1_ps(1.0f / *scale),z=_mm256_set1_ps((float)*zp);
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
