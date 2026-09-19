/* Original AVX2 W8A8 experiment. Weights: per-output symmetric [-127,127].
 * Activations: per-row asymmetric 254-step range, centered at -127.
 * No static calibration, saturating pair sums, or quantized recurrent state.
 */
#include "internal.h"
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct dpdf_qmatrix {
    int k,n;
    int8_t *packed;
    float *scale;
    int32_t *sum;
};
void dpdf_qdestroy(dpdf_qmatrix *q) {
    if (q) { free(q->packed); free(q->scale); free(q->sum); free(q); }
}
dpdf_qmatrix *dpdf_qcreate(const float *w,int k,int n) {
    if (!w || k<=0 || k>512 || k%8 || n<=0 || n%8) return NULL;
    dpdf_qmatrix *q=calloc(1,sizeof(*q)); if (!q) return NULL;
    q->k=k; q->n=n;
    q->packed=malloc((size_t)k*n); q->scale=malloc(n*sizeof(float)); q->sum=calloc(n,sizeof(int32_t));
    if (!q->packed || !q->scale || !q->sum) { dpdf_qdestroy(q); return NULL; }
    for (int c=0;c<n;++c) {
        float maximum=0;
        for (int j=0;j<k;++j) { if (!isfinite(w[j*n+c])) { dpdf_qdestroy(q); return NULL; } maximum=fmaxf(maximum,fabsf(w[j*n+c])); }
        float scale=maximum>0 ? maximum/127.0f : 1.0f; q->scale[c]=scale;
        for (int j=0;j<k;++j) {
            int v=(int)nearbyintf(w[j*n+c]/scale);
            if (v>127) v=127;
            if (v< -127) v= -127;
            /* Each vector contains four K entries for each of eight outputs. */
            q->packed[(c/8)*k*8+(j/4)*32+(c%8)*4+j%4]=(int8_t)v;
            q->sum[c]+=v;
        }
    }
    return q;
}
size_t dpdf_qbytes(const dpdf_qmatrix *q) {
    return q ? sizeof(*q)+(size_t)q->k*q->n+(size_t)q->n*8 : 0;
}

static void quantize(const float *x,int8_t *out,int k,float *scale,int *zp) {
    __m256 lo=_mm256_setzero_ps(),hi=lo;
    for (int j=0;j<k;j+=8) { __m256 v=_mm256_loadu_ps(x+j); lo=_mm256_min_ps(lo,v); hi=_mm256_max_ps(hi,v); }
    float a[8],b[8]; _mm256_storeu_ps(a,lo); _mm256_storeu_ps(b,hi);
    float low=0,high=0;
    for (int j=0;j<8;++j) { if (a[j]<low) low=a[j]; if (b[j]>high) high=b[j]; }
    if (low==high) { memset(out,0,k); *scale=1; *zp=127; return; }
    *scale=(high-low)/254.0f;
    *zp=(int)nearbyintf(-low / *scale);
    if (*zp<0) *zp=0;
    if (*zp>254) *zp=254;
    __m256 inv=_mm256_set1_ps(1.0f / *scale),z=_mm256_set1_ps((float)*zp);
    for (int j=0;j<k;j+=8) {
        __m256 v=_mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(x+j),inv),z);
        __m256i q=_mm256_cvtps_epi32(v);
        q=_mm256_min_epi32(_mm256_set1_epi32(254),_mm256_max_epi32(_mm256_setzero_si256(),q));
        q=_mm256_sub_epi32(q,_mm256_set1_epi32(127));
        __m128i p=_mm_packs_epi32(_mm256_castsi256_si128(q),_mm256_extracti128_si256(q,1));
        p=_mm_packs_epi16(p,p); _mm_storel_epi64((__m128i *)(out+j),p);
    }
}

void dpdf_qaffine(const dpdf_qmatrix *q,const float *x,const float *bias,float *y,int m) {
    /* Callers split larger batches into at most 48 rows. */
    int8_t activation[48*512]; float scales[48]; int zp[48];
    const int k=q->k,n=q->n;
    for (int r=0;r<m;++r) quantize(x+r*k,activation+r*k,k,scales+r,zp+r);
    const __m256i ones=_mm256_set1_epi16(1);
    int r=0;
    for (;r+3<m;r+=4) for (int c=0;c<n;c+=8) {
        __m256i s0=_mm256_setzero_si256(),s1=s0,s2=s0,s3=s0;
        for (int j=0;j<k;j+=4) {
            int32_t v0,v1,v2,v3;
            memcpy(&v0,activation+r*k+j,4); memcpy(&v1,activation+(r+1)*k+j,4);
            memcpy(&v2,activation+(r+2)*k+j,4); memcpy(&v3,activation+(r+3)*k+j,4);
            __m256i a0=_mm256_set1_epi32(v0),a1=_mm256_set1_epi32(v1);
            __m256i a2=_mm256_set1_epi32(v2),a3=_mm256_set1_epi32(v3);
            __m256i w=_mm256_loadu_si256((const __m256i *)(q->packed+c*k+j*8));
            s0=_mm256_add_epi32(s0,_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(a0),_mm256_sign_epi8(w,a0)),ones));
            s1=_mm256_add_epi32(s1,_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(a1),_mm256_sign_epi8(w,a1)),ones));
            s2=_mm256_add_epi32(s2,_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(a2),_mm256_sign_epi8(w,a2)),ones));
            s3=_mm256_add_epi32(s3,_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(a3),_mm256_sign_epi8(w,a3)),ones));
        }
        __m256i sums[4]={s0,s1,s2,s3};
        for (int t=0;t<4;++t) {
            __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+c)),_mm256_set1_epi32(127-zp[r+t]));
            __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r+t]),_mm256_loadu_ps(q->scale+c));
            __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sums[t],correction)),scale,_mm256_loadu_ps(bias+c));
            _mm256_storeu_ps(y+(r+t)*n+c,result);
        }
    }
    for (;r<m;++r) for (int c=0;c<n;c+=8) {
        __m256i sum=_mm256_setzero_si256();
        for (int j=0;j<k;j+=4) {
            int32_t bytes; memcpy(&bytes,activation+r*k+j,4);
            __m256i a=_mm256_set1_epi32(bytes);
            __m256i w=_mm256_loadu_si256((const __m256i *)(q->packed+c*k+j*8));
            /* |a|<=127, |w|<=127 => pair sums <=32258, no i16 saturation.
             * Neither operand contains -128, so sign transfer is exact. */
            __m256i pairs=_mm256_maddubs_epi16(_mm256_abs_epi8(a),_mm256_sign_epi8(w,a));
            sum=_mm256_add_epi32(sum,_mm256_madd_epi16(pairs,ones));
        }
        __m256i correction=_mm256_mullo_epi32(_mm256_loadu_si256((const __m256i *)(q->sum+c)),_mm256_set1_epi32(127-zp[r]));
        sum=_mm256_add_epi32(sum,correction);
        __m256 scale=_mm256_mul_ps(_mm256_set1_ps(scales[r]),_mm256_loadu_ps(q->scale+c));
        __m256 result=_mm256_fmadd_ps(_mm256_cvtepi32_ps(sum),scale,_mm256_loadu_ps(bias+c));
        _mm256_storeu_ps(y+r*n+c,result);
    }
}
