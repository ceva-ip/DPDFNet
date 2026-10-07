/* Exact prepared-layout versus ordinary-layout contract. Independent ordinary
 * qmatrices retain the existing scalar-oracle-validated integer implementation. */
#include "internal.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(value) do { if (!(value)) { fprintf(stderr,"Recurrent layout contract line %d\n",__LINE__); return 1; } } while (0)

static uint32_t random_state=0x19a8734u;
static float sample(void) {
    random_state=random_state*1664525u+1013904223u;
    return ((int)(random_state>>16)-32768)/8192.0f;
}

static int unsupported_dimensions(const float *weights) {
    const int dims[][2]={{64,64},{64,72},{8,192},{128,192}};
    float xa[129],bias[193],before[194],after[194];
    float *x=xa+1,*b=bias+1;
    CHECK(dpdf_qprepare_row_layout(NULL)==-1);
    for (int i=0;i<128;++i) x[i]=sample();
    for (int i=0;i<192;++i) b[i]=sample();
    for (int d=0;d<4;++d) {
        int k=dims[d][0],n=dims[d][1];
        dpdf_qmatrix *q=dpdf_qcreate(weights,k,n); CHECK(q);
        size_t owned=dpdf_qbytes(q);
        dpdf_qaffine(q,x,b,before+1,1);
        CHECK(dpdf_qprepare_row_layout(q)==-1);
        CHECK(dpdf_qbytes(q)==owned);
        dpdf_qaffine(q,x,b,after+1,1);
        CHECK(!memcmp(before+1,after+1,(size_t)n*sizeof(float)));
        dpdf_qdestroy(q);
    }
    return 0;
}

int main(void) {
    if (!dpdf_has_avx2()) { puts("Recurrent layout SIMD unavailable; skipped"); return 0; }
    float *w0=malloc((size_t)128*192*sizeof(float));
    float *w1=malloc((size_t)128*192*sizeof(float));
    float *xa=malloc(((size_t)48*64+1)*sizeof(float));
    float *y0=malloc(((size_t)48*192+2)*sizeof(float));
    float *y1=malloc(((size_t)48*192+2)*sizeof(float));
    float *reference0=malloc((size_t)48*192*sizeof(float));
    float *reference1=malloc((size_t)48*192*sizeof(float));
    float bias0[193],bias1[193];
    CHECK(w0 && w1 && xa && y0 && y1 && reference0 && reference1);
    for (int j=0;j<128;++j) for (int c=0;c<192;++c) {
        w0[j*192+c]=c%7==0?0.0f:sample();
        w1[j*192+c]=c%5==0?0.0f:sample();
    }
    CHECK(unsupported_dimensions(w0)==0);
    for (int c=0;c<192;++c) {
        bias0[c+1]=c%7==0?(c%2?-0.0f:+0.0f):sample();
        bias1[c+1]=c%5==0?(c%2?-0.0f:+0.0f):sample();
    }
    dpdf_qmatrix *q0=dpdf_qcreate(w0,64,192),*q1=dpdf_qcreate(w1,64,192);
    dpdf_qmatrix *ordinary0=dpdf_qcreate(w0,64,192),*ordinary1=dpdf_qcreate(w1,64,192);
    CHECK(q0 && q1 && ordinary0 && ordinary1);
    size_t bytes0=dpdf_qbytes(q0),bytes1=dpdf_qbytes(q1);
    CHECK(dpdf_qprepare_row_layout(q0)==0);
    CHECK(dpdf_qprepare_row_layout(q0)==0);
    CHECK(dpdf_qbytes(q0)==bytes0);
    const int rows[]={1,2,4,5,48};
    float *x=xa+1;
    for (int prepared_pair=0;prepared_pair<2;++prepared_pair) {
        if (prepared_pair) {
            CHECK(dpdf_qprepare_row_layout(q1)==0);
            CHECK(dpdf_qprepare_row_layout(q1)==0);
            CHECK(dpdf_qbytes(q1)==bytes1);
        }
        for (int pattern=0;pattern<8;++pattern) {
            for (int i=0;i<48*64;++i)
                x[i]=pattern==0?+0.0f:pattern==1?-0.0f:
                     pattern==2?(i%2?-0.0f:+0.0f):
                     pattern==3?(i%2?-10000.0f:10000.0f):
                     pattern==4?(i%2?-1.0e-20f:1.0e-20f):sample();
            for (int mi=0;mi<5;++mi) {
                int m=rows[mi];
                dpdf_qaffine(ordinary0,x,bias0+1,reference0,m);
                dpdf_qaffine(ordinary1,x,bias1+1,reference1,m);
                y0[0]=y1[0]=12345.0f;
                y0[m*192+1]=y1[m*192+1]=-12345.0f;
                dpdf_qaffine(q0,x,bias0+1,y0+1,m);
                dpdf_qaffine(q1,x,bias1+1,y1+1,m);
                CHECK(!memcmp(y0+1,reference0,(size_t)m*192*sizeof(float)));
                CHECK(!memcmp(y1+1,reference1,(size_t)m*192*sizeof(float)));
                CHECK(y0[0]==12345.0f && y1[0]==12345.0f);
                CHECK(y0[m*192+1]==-12345.0f && y1[m*192+1]==-12345.0f);
                dpdf_qaffine_pair(q0,q1,x,bias0+1,bias1+1,y0+1,y1+1,m);
                CHECK(!memcmp(y0+1,reference0,(size_t)m*192*sizeof(float)));
                CHECK(!memcmp(y1+1,reference1,(size_t)m*192*sizeof(float)));
                /* Also exercise a prepared matrix in the second position. */
                dpdf_qaffine_pair(ordinary0,q0,x,bias0+1,bias0+1,y0+1,y1+1,m);
                CHECK(!memcmp(y0+1,reference0,(size_t)m*192*sizeof(float)));
                CHECK(!memcmp(y1+1,reference0,(size_t)m*192*sizeof(float)));
                CHECK(y0[0]==12345.0f && y1[0]==12345.0f);
                CHECK(y0[m*192+1]==-12345.0f && y1[m*192+1]==-12345.0f);
                CHECK(dpdf_qbytes(q0)==bytes0 && dpdf_qbytes(q1)==bytes1);
            }
        }
    }
    dpdf_qdestroy(q0);dpdf_qdestroy(q1);dpdf_qdestroy(ordinary0);dpdf_qdestroy(ordinary1);
    free(w0);free(w1);free(xa);free(y0);free(y1);free(reference0);free(reference1);
    puts("Recurrent prepared-layout row/batch/mixed-pair, idempotence, unaligned, signed-zero and owned-byte checks passed");
    return 0;
}
