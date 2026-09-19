#include "internal.h"
#include <immintrin.h>
#include <math.h>

void dpdf_axpy_avx2(float *y, const float *x, float a, int n) {
    __m256 v=_mm256_set1_ps(a);
    int i=0;
    for (; i+7<n; i+=8)
        _mm256_storeu_ps(y+i,_mm256_fmadd_ps(v,_mm256_loadu_ps(x+i),_mm256_loadu_ps(y+i)));
    for (; i<n; ++i) y[i]+=a*x[i];
}

/* Input-major, output-contiguous weights. Four rows share each loaded tile.
 * Handles arbitrary widths, including the grouped projections. */
void dpdf_affine_avx2(const float *x, const float *w, const float *bias,
                      float *y, int m, int k, int n) {
    int r = 0;
    for (; n%16==0 && r+3 < m; r += 4) {
        for (int c = 0; c < n; c += 16) {
            __m256 a0 = _mm256_loadu_ps(bias+c), a1 = _mm256_loadu_ps(bias+c+8);
            __m256 b0=a0, b1=a1, c0=a0, c1=a1, d0=a0, d1=a1;
            for (int j = 0; j < k; ++j) {
                __m256 w0 = _mm256_loadu_ps(w+j*n+c), w1 = _mm256_loadu_ps(w+j*n+c+8);
                __m256 v = _mm256_set1_ps(x[r*k+j]);
                a0=_mm256_fmadd_ps(v,w0,a0); a1=_mm256_fmadd_ps(v,w1,a1);
                v=_mm256_set1_ps(x[(r+1)*k+j]);
                b0=_mm256_fmadd_ps(v,w0,b0); b1=_mm256_fmadd_ps(v,w1,b1);
                v=_mm256_set1_ps(x[(r+2)*k+j]);
                c0=_mm256_fmadd_ps(v,w0,c0); c1=_mm256_fmadd_ps(v,w1,c1);
                v=_mm256_set1_ps(x[(r+3)*k+j]);
                d0=_mm256_fmadd_ps(v,w0,d0); d1=_mm256_fmadd_ps(v,w1,d1);
            }
            _mm256_storeu_ps(y+r*n+c,a0); _mm256_storeu_ps(y+r*n+c+8,a1);
            _mm256_storeu_ps(y+(r+1)*n+c,b0); _mm256_storeu_ps(y+(r+1)*n+c+8,b1);
            _mm256_storeu_ps(y+(r+2)*n+c,c0); _mm256_storeu_ps(y+(r+2)*n+c+8,c1);
            _mm256_storeu_ps(y+(r+3)*n+c,d0); _mm256_storeu_ps(y+(r+3)*n+c+8,d1);
        }
    }
    for (; r < m; ++r) {
        int c=0;
        for (; c+63<n; c+=64) {
            __m256 a0=_mm256_loadu_ps(bias+c),a1=_mm256_loadu_ps(bias+c+8);
            __m256 a2=_mm256_loadu_ps(bias+c+16),a3=_mm256_loadu_ps(bias+c+24);
            __m256 a4=_mm256_loadu_ps(bias+c+32),a5=_mm256_loadu_ps(bias+c+40);
            __m256 a6=_mm256_loadu_ps(bias+c+48),a7=_mm256_loadu_ps(bias+c+56);
            for (int j=0;j<k;++j) {
                __m256 v=_mm256_set1_ps(x[r*k+j]);
                const float *p=w+j*n+c;
                a0=_mm256_fmadd_ps(v,_mm256_loadu_ps(p),a0);
                a1=_mm256_fmadd_ps(v,_mm256_loadu_ps(p+8),a1);
                a2=_mm256_fmadd_ps(v,_mm256_loadu_ps(p+16),a2);
                a3=_mm256_fmadd_ps(v,_mm256_loadu_ps(p+24),a3);
                a4=_mm256_fmadd_ps(v,_mm256_loadu_ps(p+32),a4);
                a5=_mm256_fmadd_ps(v,_mm256_loadu_ps(p+40),a5);
                a6=_mm256_fmadd_ps(v,_mm256_loadu_ps(p+48),a6);
                a7=_mm256_fmadd_ps(v,_mm256_loadu_ps(p+56),a7);
            }
            _mm256_storeu_ps(y+r*n+c,a0); _mm256_storeu_ps(y+r*n+c+8,a1);
            _mm256_storeu_ps(y+r*n+c+16,a2); _mm256_storeu_ps(y+r*n+c+24,a3);
            _mm256_storeu_ps(y+r*n+c+32,a4); _mm256_storeu_ps(y+r*n+c+40,a5);
            _mm256_storeu_ps(y+r*n+c+48,a6); _mm256_storeu_ps(y+r*n+c+56,a7);
        }
        for (; c+31 < n; c += 32) {
            __m256 a=_mm256_loadu_ps(bias+c), b=_mm256_loadu_ps(bias+c+8);
            __m256 d=_mm256_loadu_ps(bias+c+16), e=_mm256_loadu_ps(bias+c+24);
            for (int j = 0; j < k; ++j) {
                __m256 v=_mm256_set1_ps(x[r*k+j]);
                a=_mm256_fmadd_ps(v,_mm256_loadu_ps(w+j*n+c),a);
                b=_mm256_fmadd_ps(v,_mm256_loadu_ps(w+j*n+c+8),b);
                d=_mm256_fmadd_ps(v,_mm256_loadu_ps(w+j*n+c+16),d);
                e=_mm256_fmadd_ps(v,_mm256_loadu_ps(w+j*n+c+24),e);
            }
            _mm256_storeu_ps(y+r*n+c,a); _mm256_storeu_ps(y+r*n+c+8,b);
            _mm256_storeu_ps(y+r*n+c+16,d); _mm256_storeu_ps(y+r*n+c+24,e);
        }
        for (; c+7<n; c+=8) {
            __m256 a=_mm256_loadu_ps(bias+c);
            for (int j=0; j<k; ++j)
                a=_mm256_fmadd_ps(_mm256_set1_ps(x[r*k+j]),_mm256_loadu_ps(w+j*n+c),a);
            _mm256_storeu_ps(y+r*n+c,a);
        }
        for (; c<n; ++c) {
            float a=bias[c];
            for (int j=0; j<k; ++j) a+=x[r*k+j]*w[j*n+c];
            y[r*n+c]=a;
        }
    }
}

/* exp(x): range reduction x=n*ln(2)+r, |r|<=ln(2)/2, degree-7 Taylor
 * evaluated by FMA. No external approximation code. Clamp affects only
 * saturated sigmoid/tanh tails. Activations are FP32 approximations, not
 * bit-identical libm functions; their error is explicitly tested. */
static inline __m256 exp8(__m256 x) {
    x=_mm256_max_ps(_mm256_set1_ps(-87.0f),_mm256_min_ps(_mm256_set1_ps(87.0f),x));
    __m256 nf=_mm256_round_ps(_mm256_mul_ps(x,_mm256_set1_ps(1.4426950408889634f)),
                             _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256 r=_mm256_fnmadd_ps(nf,_mm256_set1_ps(0.693145751953125f),x);
    r=_mm256_fnmadd_ps(nf,_mm256_set1_ps(1.428606765330187e-6f),r);
    __m256 p=_mm256_set1_ps(1.0f/5040.0f);
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/720.0f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/120.0f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/24.0f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f/6.0f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(0.5f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.0f));
    __m256i e=_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(nf),_mm256_set1_epi32(127)),23);
    return _mm256_mul_ps(p,_mm256_castsi256_ps(e));
}
static inline __m256 sigmoid8(__m256 x) {
    return _mm256_div_ps(_mm256_set1_ps(1),_mm256_add_ps(_mm256_set1_ps(1),exp8(_mm256_sub_ps(_mm256_setzero_ps(),x))));
}
static inline __m256 tanh8(__m256 x) {
    __m256 sign=_mm256_and_ps(x,_mm256_set1_ps(-0.0f));
    __m256 absolute=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),x);
    __m256 e=exp8(_mm256_mul_ps(_mm256_set1_ps(-2),absolute));
    return _mm256_xor_ps(sign,_mm256_div_ps(_mm256_sub_ps(_mm256_set1_ps(1),e),_mm256_add_ps(_mm256_set1_ps(1),e)));
}
void dpdf_gates_avx2(const float *a, const float *b, const float *old, float *out, int rows) {
    for (int r=0; r<rows; ++r) for (int c=0; c<64; c+=8) {
        __m256 reset=sigmoid8(_mm256_add_ps(_mm256_loadu_ps(a+r*192+c),_mm256_loadu_ps(b+r*192+c)));
        __m256 update=sigmoid8(_mm256_add_ps(_mm256_loadu_ps(a+r*192+64+c),_mm256_loadu_ps(b+r*192+64+c)));
        __m256 candidate=tanh8(_mm256_fmadd_ps(reset,_mm256_loadu_ps(b+r*192+128+c),_mm256_loadu_ps(a+r*192+128+c)));
        __m256 result=_mm256_fmadd_ps(update,_mm256_sub_ps(_mm256_loadu_ps(old+r*64+c),candidate),candidate);
        _mm256_storeu_ps(out+r*64+c,result);
    }
}
void dpdf_activations_avx2(const float *x, float *s, float *t, size_t count) {
    size_t i=0;
    for (; i+7<count; i+=8) {
        __m256 v=_mm256_loadu_ps(x+i);
        _mm256_storeu_ps(s+i,sigmoid8(v)); _mm256_storeu_ps(t+i,tanh8(v));
    }
    for (; i<count; ++i) { s[i]=1.0f/(1.0f+expf(-x[i])); t[i]=tanhf(x[i]); }
}
