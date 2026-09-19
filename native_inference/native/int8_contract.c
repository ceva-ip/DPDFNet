/* Independent scalar integer arithmetic checks the SIMD dot path, including
 * extreme signed values that would expose saturated pair sums. */
#include "internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"INT8 test line %d\n",__LINE__); return 1; } } while (0)

int main(void) {
    if (!dpdf_has_avx2()) { puts("INT8 SIMD unavailable; skipped"); return 0; }
    for (int k=64;k<=512;k*=2) for (int n=64;n<=192;n+=128) {
        float *w=malloc(k*n*sizeof(float)),*x=malloc(7*k*sizeof(float));
        float *y=malloc(7*n*sizeof(float)),*bias=malloc(n*sizeof(float));
        CHECK(w && x && y && bias);
        for (int j=0;j<k;++j) for (int c=0;c<n;++c)
            w[j*n+c]=((j+c)%3==0 ? -1.0f : 1.0f)*(1+(c%7))*.0625f;
        for (int c=0;c<n;++c) bias[c]=(c%5)*.001f;
        for (int r=0;r<7;++r) for (int j=0;j<k;++j)
            x[r*k+j]=r==0 ? 0 : r==1 ? .5f : r==2 ? -.5f : ((j%2) ? -1.0f : 1.0f);
        dpdf_qmatrix *q=dpdf_qcreate(w,k,n); CHECK(q);
        dpdf_qaffine(q,x,bias,y,7);
        for (int r=0;r<7;++r) for (int c=0;c<n;++c) {
            /* These inputs/weights lie exactly on the chosen quantization
             * grid. Independent double dot must agree with SIMD dequant. */
            double expected=bias[c];
            for (int j=0;j<k;++j) expected+=(double)x[r*k+j]*w[j*n+c];
            CHECK(fabs(y[r*n+c]-expected)<2e-5);
        }
        dpdf_qdestroy(q); free(w); free(x); free(y); free(bias);
    }
    puts("INT8 saturated-dot, zero, positive/negative activation and batch-tail checks passed");
    return 0;
}
