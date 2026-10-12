/* Compare runtime selection with the unchanged AVX2 implementation in the
 * same executable. Cover every row/tile/tail/paired shape and unaligned I/O. */
#include "int8_matrix.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"Dispatch test line %d\n",__LINE__); return 1; } } while (0)

static uint32_t seed=0x194812;
static float sample(void) {
    seed=seed*1664525u+1013904223u;
    return ((int)(seed>>16)-32768)/8192.0f;
}

int main(void) {
    const int depths[]={8,64,256,512},widths[]={8,64,192,768},rows[]={1,4,7,48};
    int cases=0;
    for (int ki=0;ki<4;++ki) for (int ni=0;ni<4;++ni) for (int mi=0;mi<4;++mi) {
        int k=depths[ki],n=widths[ni],m=rows[mi];
        float *w=malloc((size_t)k*n*sizeof(float));
        float *xa=malloc(((size_t)m*k+1)*sizeof(float));
        float *ba=malloc((n+1)*sizeof(float));
        float *a[4];
        CHECK(w && xa && ba);
        for (int i=0;i<4;++i) { a[i]=malloc(((size_t)m*n+2)*sizeof(float)); CHECK(a[i]); }
        float *x=xa+1,*bias=ba+1;
        for (int i=0;i<k*n;++i) w[i]=sample();
        dpdf_qmatrix *q0=dpdf_qcreate(w,k,n); CHECK(q0);
        for (int i=0;i<k*n;++i) w[i]=sample();
        dpdf_qmatrix *q1=dpdf_qcreate(w,k,n); CHECK(q1);
        CHECK(q0->affine==q1->affine && q0->pair==q1->pair);
#ifdef DPDF_HAS_AVX_VNNI
        if (dpdf_has_avx_vnni()) {
            CHECK(q0->affine==dpdf_qaffine_vnni);
            CHECK(q0->pair==dpdf_qaffine_pair_vnni);
        } else
#endif
        {
            CHECK(q0->affine==dpdf_qaffine_avx2);
            CHECK(q0->pair==dpdf_qaffine_pair_avx2);
        }
        for (int i=0;i<n;++i) bias[i]=sample();
        for (int pattern=0;pattern<4;++pattern) {
            for (int i=0;i<m*k;++i)
                x[i]=pattern==0 ? 0 : pattern==1 ? .001f :
                     pattern==2 ? (i%2 ? -1000.0f : 1000.0f) : sample();
            for (int i=0;i<4;++i) { a[i][0]=12345; a[i][m*n+1]=-12345; }
            dpdf_qaffine(q0,x,bias,a[0]+1,m);
            dpdf_qaffine_avx2(q0,x,bias,a[1]+1,m);
            CHECK(!memcmp(a[0]+1,a[1]+1,(size_t)m*n*sizeof(float)));
            dpdf_qaffine_pair(q0,q1,x,bias,bias,a[0]+1,a[1]+1,m);
            dpdf_qaffine_pair_avx2(q0,q1,x,bias,bias,a[2]+1,a[3]+1,m);
            CHECK(!memcmp(a[0]+1,a[2]+1,(size_t)m*n*sizeof(float)));
            CHECK(!memcmp(a[1]+1,a[3]+1,(size_t)m*n*sizeof(float)));
            for (int i=0;i<4;++i) CHECK(a[i][0]==12345 && a[i][m*n+1]==-12345);
            ++cases;
        }
        dpdf_qdestroy(q0); dpdf_qdestroy(q1); free(w); free(xa); free(ba);
        for (int i=0;i<4;++i) free(a[i]);
    }
    printf("%d exact dispatch cases; selected %s\n",cases,dpdf_has_avx_vnni() ? "AVX-VNNI" : "AVX2");
    return 0;
}
