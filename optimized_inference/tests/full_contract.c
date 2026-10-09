#include "full_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"Failed line %d: %s\n",__LINE__,#x); return 1; } } while (0)

int main(int argc,char **argv) {
    CHECK(argc==2);
    size_t n=dpdf_model_weight_count();
    size_t state_size=dpdf_model_state_size(); CHECK(state_size>0);
    float *weights=malloc(n*sizeof(float));
    float *a=malloc(state_size*sizeof(float)),*b=malloc(state_size*sizeof(float));
    float *next=malloc(state_size*sizeof(float));
    CHECK(weights && a && b && next);
    FILE *f=fopen(argv[1],"rb"); CHECK(f);
    CHECK(fread(weights,sizeof(float),n,f)==n); CHECK(fgetc(f)==EOF); fclose(f);
    CHECK(!dpdf_model_create(NULL,n,0)); CHECK(!dpdf_model_create(weights,n-1,0));
    CHECK(!dpdf_model_create(weights,n,5)); CHECK(dpdf_model_init_state(NULL)==-1);
    if (!dpdf_has_avx2()) {
        CHECK(!dpdf_model_create(weights,n,DPDF_AVX2));
        CHECK(!dpdf_model_create(weights,n,DPDF_EXPERIMENTAL_INT8));
    }
    if (!dpdf_has_fp16()) CHECK(!dpdf_model_create(weights,n,DPDF_EXPERIMENTAL_FP16));
    float saved=weights[0]; weights[0]=NAN; CHECK(!dpdf_model_create(weights,n,0)); weights[0]=saved;
    dpdf_model_destroy(NULL);
    int tiers[]={DPDF_SCALAR,DPDF_AVX2,DPDF_EXPERIMENTAL_FP16,DPDF_EXPERIMENTAL_INT8};
    for (int t=0;t<4;++t) {
        if (t==1 && !dpdf_has_avx2()) continue;
        if (t==2 && !dpdf_has_fp16()) continue;
        if (t==3 && !dpdf_has_avx2()) continue;
        dpdf_model *m=dpdf_model_create(weights,n,tiers[t]); CHECK(m);
        CHECK(dpdf_model_init_state(a)==0); CHECK(dpdf_model_init_state(b)==0);
        float x[962],y[962],z[962];
        CHECK(dpdf_model_process(NULL,x,a,y,next)==-1);
        CHECK(dpdf_model_process(m,NULL,a,y,next)==-1);
        CHECK(dpdf_model_process(m,x,NULL,y,next)==-1);
        CHECK(dpdf_model_process(m,x,a,NULL,next)==-1);
        CHECK(dpdf_model_process(m,x,a,y,NULL)==-1);
        for (int frame=0;frame<32;++frame) {
            for (int i=0;i<962;++i) x[i]=frame>16 ? 0.0f : .2f*sinf((float)(i+frame)*.17f);
            CHECK(dpdf_model_process(m,x,a,y,next)==0);
            memcpy(z,x,sizeof(x));
            CHECK(dpdf_model_process(m,z,b,z,b)==0);
            CHECK(!memcmp(y,z,sizeof(y)) && !memcmp(next,b,state_size*sizeof(float)));
            for (int i=0;i<962;++i) CHECK(isfinite(y[i]));
            for (size_t i=0;i<state_size;++i) CHECK(isfinite(next[i]));
            memcpy(a,next,state_size*sizeof(float));
        }
        /* Reusing an arena after reset must match a fresh context. */
        dpdf_model *fresh=dpdf_model_create(weights,n,tiers[t]); CHECK(fresh);
        CHECK(dpdf_model_init_state(a)==0); CHECK(dpdf_model_init_state(b)==0);
        CHECK(dpdf_model_process(m,x,a,y,a)==0);
        CHECK(dpdf_model_process(fresh,x,b,z,b)==0);
        CHECK(!memcmp(y,z,sizeof(y)) && !memcmp(a,b,state_size*sizeof(float)));
        dpdf_model_destroy(fresh); dpdf_model_destroy(m);
    }
    free(weights); free(a); free(b); free(next);
    puts("Full model contract, in-place buffers, reset and finite-state checks passed");
    return 0;
}
