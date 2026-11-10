#define _GNU_SOURCE
#include "int8_matrix.h"
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>
/* Independent scalar integer arithmetic checks the SIMD dot path, including
 * extreme signed values that would expose saturated pair sums. */
#include "internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"INT8 test line %d\n",__LINE__); return 1; } } while (0)

typedef struct {void *mapping;size_t size;int8_t *old;} packed_guard;
static packed_guard protect_packed(dpdf_qmatrix *q) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE),bytes=(size_t)q->k*q->n,span=(bytes+page-1)/page*page;
    void *mapping=mmap(NULL,span+page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if(mapping==MAP_FAILED || mprotect((char *)mapping+span,page,PROT_NONE)) abort();
    int8_t *packed=(int8_t *)mapping+span-bytes;memcpy(packed,q->packed,bytes);
    if(mprotect(mapping,span,PROT_READ)) abort();
    packed_guard result={mapping,span+page,q->packed};q->packed=packed;return result;
}
static void release_packed(dpdf_qmatrix *q,packed_guard g) {q->packed=g.old;munmap(g.mapping,g.size);}
static uint32_t rng=0x194812;
static float sample(void) {
    rng=rng*1664525u+1013904223u;
    return ((int)(rng>>16)-32768)/8192.0f;
}

/* Independent scalar quantization/dot oracle, including M=1, short widths,
 * 64-output tile tails and buffers without 32-byte alignment. */
static int row_contract(void) {
    const int widths[]={8,56,64,72,128,192,768};
    const int depths[]={8,24,64,128,256,512};
    for (size_t ki=0;ki<sizeof(depths)/sizeof(depths[0]);++ki)
      for (size_t ni=0;ni<sizeof(widths)/sizeof(widths[0]);++ni) {
        int k=depths[ki],n=widths[ni];
        float *w=malloc((size_t)k*n*sizeof(float));
        float *xa=malloc((k+1)*sizeof(float)),*ya=malloc((n+2)*sizeof(float));
        float *ba=malloc((n+1)*sizeof(float));
        CHECK(w && xa && ya && ba);
        float *x=xa+1,*y=ya+1,*bias=ba+1;
        for (int j=0;j<k;++j) for (int c=0;c<n;++c)
            w[j*n+c]=c%5==0 ? 0 : c%5==1 ? (j%2 ? -1.0f : 1.0f) : sample();
        for (int c=0;c<n;++c) bias[c]=sample();
        dpdf_qmatrix *q=dpdf_qcreate(w,k,n); CHECK(q); packed_guard g=protect_packed(q);
        for (int pattern=0;pattern<8;++pattern) {
            for (int j=0;j<k;++j)
                x[j]=pattern==0 ? 0 : pattern==1 ? .5f : pattern==2 ? -.5f :
                     pattern==3 ? (j%2 ? -1.0f : 1.0f) : sample();
            float low=0,high=0;
            for (int j=0;j<k;++j) { low=fminf(low,x[j]); high=fmaxf(high,x[j]); }
            float scale=low==high ? 1 : (high-low)/254.0f;
            int zp=low==high ? 127 : (int)nearbyintf(-low/scale);
            if (zp<0) zp=0;
            if (zp>254) zp=254;
            int aq[512];
            for (int j=0;j<k;++j) {
                int v=(int)nearbyintf(x[j]*(1.0f/scale)+(float)zp);
                if (v<0) v=0;
                if (v>254) v=254;
                aq[j]=v-zp;
            }
            ya[0]=12345; ya[n+1]=-12345;
            dpdf_qaffine(q,x,bias,y,1);
            CHECK(ya[0]==12345 && ya[n+1]==-12345);
            for (int c=0;c<n;++c) {
                float maximum=0;
                for (int j=0;j<k;++j) maximum=fmaxf(maximum,fabsf(w[j*n+c]));
                float ws=maximum>0 ? maximum/63.0f : 1;
                int32_t sum=0;
                for (int j=0;j<k;++j) {
                    int v=(int)nearbyintf(w[j*n+c]/ws);
                    if (v>63) v=63;
                    if (v< -63) v= -63;
                    sum+=aq[j]*v;
                }
                float expected=fmaf((float)sum,scale*ws,bias[c]);
                CHECK(memcmp(y+c,&expected,sizeof(float))==0);
            }
        }
        release_packed(q,g);dpdf_qdestroy(q); free(w); free(xa); free(ya); free(ba);
      }
    return 0;
}

static int batch_contract(void) {
    const int depths[]={8,64,128,512},widths[]={8,64,72,192},rows[]={4,7,40,48};
    for (int ki=0;ki<4;++ki) for (int ni=0;ni<4;++ni) {
        int k=depths[ki],n=widths[ni];
        float *w=malloc((size_t)k*n*4),*x=malloc((size_t)48*k*4),*bias=malloc(n*4);
        float *y=malloc((size_t)(48*n+2)*4),*z=malloc((size_t)(48*n+2)*4),*ref=malloc(n*4);
        CHECK(w && x && bias && y && z && ref);
        for (int i=0;i<k*n;++i) w[i]=sample();
        dpdf_qmatrix *q0=dpdf_qcreate(w,k,n); CHECK(q0); packed_guard g0=protect_packed(q0);
        for (int i=0;i<k*n;++i) w[i]=sample();
        dpdf_qmatrix *q1=dpdf_qcreate(w,k,n); CHECK(q1); packed_guard g1=protect_packed(q1);
        for (int i=0;i<48*k;++i) x[i]=i/k==0 ? 0 : sample();
        for (int c=0;c<n;++c) bias[c]=sample();
        for (int mi=0;mi<4;++mi) {
            int m=rows[mi];
            y[0]=z[0]=12345;y[m*n+1]=z[m*n+1]=-12345;
            dpdf_qaffine(q0,x,bias,y+1,m);
            for (int r=0;r<m;++r) {
                dpdf_qaffine(q0,x+r*k,bias,ref,1);
                CHECK(!memcmp(y+1+r*n,ref,n*4));
            }
            dpdf_qaffine_pair(q0,q1,x,bias,bias,y+1,z+1,m);
            for (int r=0;r<m;++r) {
                dpdf_qaffine(q0,x+r*k,bias,ref,1); CHECK(!memcmp(y+1+r*n,ref,n*4));
                dpdf_qaffine(q1,x+r*k,bias,ref,1); CHECK(!memcmp(z+1+r*n,ref,n*4));
            }
            CHECK(y[0]==12345 && z[0]==12345 && y[m*n+1]==-12345 && z[m*n+1]==-12345);
        }
        release_packed(q0,g0);release_packed(q1,g1);dpdf_qdestroy(q0);dpdf_qdestroy(q1);
        free(w);free(x);free(bias);free(y);free(z);free(ref);
    }
    return 0;
}

int main(void) {
    if(!dpdf_has_avx2()) return 77;
    unsigned saved=_mm_getcsr();unsigned rounds[]={_MM_ROUND_NEAREST,_MM_ROUND_DOWN,_MM_ROUND_UP,_MM_ROUND_TOWARD_ZERO};
    for(int mode=0;mode<4;++mode) for(int denorm=0;denorm<4;++denorm) {
        _mm_setcsr((saved&~(0x6000|0x8040))|rounds[mode]|((denorm&1)?0x8000:0)|((denorm&2)?0x40:0));
        if(row_contract() || batch_contract()) {_mm_setcsr(saved);return 1;}
    }
    _mm_setcsr(saved);puts("Scalar oracle and guarded read-only packed weights: 16 rounding/FTZ/DAZ combinations passed");return 0;
}
