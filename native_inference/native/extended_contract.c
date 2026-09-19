#include "full_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define CHECK(x) do { if (!(x)) {fprintf(stderr,"Extended contract failed line %d\n",__LINE__);return 1;} } while (0)
int main(int argc,char **argv) {
    CHECK(argc==2);size_t n=dpdf_model_weight_count(),state_size=dpdf_model_state_size();CHECK(state_size>0);
    float *w=malloc(n*4),*a=malloc(state_size*4),*b=malloc(state_size*4);CHECK(w&&a&&b);
    FILE *f=fopen(argv[1],"rb");CHECK(f);CHECK(fread(w,4,n,f)==n);fclose(f);
    CHECK(!dpdf_model_create_config(w,n,0,7,0));CHECK(!dpdf_model_create_config(w,n,0,0,16));
    int precisions[]={0,16,8},masks[]={0,1,2,4,8,7,15};
    for (int p=0;p<3;++p) {
        int precision=precisions[p];
        if (precision==16 && !dpdf_has_fp16()) {CHECK(!dpdf_model_create_config(w,n,0,16,15));continue;}
        if (precision==8 && !dpdf_has_avx2()) {CHECK(!dpdf_model_create_config(w,n,0,8,15));continue;}
        for (int c=0;c<7;++c) {
            int tier=precision==16?3:precision==8?4:0;
            dpdf_model *m=dpdf_model_create_config(w,n,tier,precision,masks[c]);CHECK(m);
            CHECK(dpdf_model_owned_bytes(m)>0);CHECK(!dpdf_model_init_state(a));CHECK(!dpdf_model_init_state(b));
            float x[962],y[962],z[962];
            for (int frame=0;frame<4;++frame) {
                for (int i=0;i<962;++i) x[i]=frame==2?0:.1f*sinf(i*.31f+frame);
                memcpy(z,x,sizeof(x));
                CHECK(!dpdf_model_process(m,x,a,y,a));CHECK(!dpdf_model_process(m,z,b,z,b));
                CHECK(!memcmp(y,z,sizeof(y)) && !memcmp(a,b,state_size*4));
                for (int i=0;i<962;++i) CHECK(isfinite(y[i]));
                for (size_t i=0;i<state_size;++i) CHECK(isfinite(a[i]));
            }
            dpdf_model_destroy(m);
        }
    }
    free(w);free(a);free(b);puts("All extended precision/family masks, aliasing and finite-state contracts passed");return 0;
}
