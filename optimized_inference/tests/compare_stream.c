/* Differential regression across two independent shared libraries. Check every
 * spectrum and every state element; alternate all four supported in-place
 * combinations. This executable links no DPDFNet implementation itself. */
#define _POSIX_C_SOURCE 200809L
#include "dpdfnet.h"
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *library;
    const dpdfnet_model_info *(*info)(dpdfnet_model_id);
    dpdfnet_model *(*create)(dpdfnet_model_id,const float *,size_t);
    int (*reset)(dpdfnet_model_id,float *);
    int (*process)(dpdfnet_model *,const float *,const float *,float *,float *);
    void (*destroy)(dpdfnet_model *);
} api;
#define CHECK(x) do {if(!(x)){fprintf(stderr,"Stream comparison line %d: %s\n",__LINE__,#x);return 1;}}while(0)
static int load(api *a,const char *path) {
    a->library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if(!a->library){fprintf(stderr,"%s\n",dlerror());return -1;}
#define SYMBOL(field,name) do {*(void **)(&a->field)=dlsym(a->library,name);if(!a->field)return -1;}while(0)
    SYMBOL(info,"dpdfnet_get_model_info"); SYMBOL(create,"dpdfnet_create");
    SYMBOL(reset,"dpdfnet_init_state"); SYMBOL(process,"dpdfnet_process"); SYMBOL(destroy,"dpdfnet_destroy");
#undef SYMBOL
    return 0;
}
static uint64_t digest(uint64_t hash,const float *x,size_t n) {
    for(size_t i=0;i<n;++i){uint32_t bits;memcpy(&bits,x+i,4);hash=(hash^bits)*UINT64_C(1099511628211);}
    return hash;
}
int main(int argc,char **argv) {
    CHECK(argc>=6 && argc<=20);
    /* Preserve the two-library CLI; append additional candidate paths. */
    const int libraries=argc-4;
    api a[libraries];
    for(int j=0;j<libraries;++j)CHECK(!load(a+j,argv[j<2?j+1:j+4]));
    dpdfnet_model_id id=(dpdfnet_model_id)atoi(argv[3]);
    const dpdfnet_model_info *info=a[0].info(id);CHECK(info);
    for(int j=1;j<libraries;++j){
        const dpdfnet_model_info *other=a[j].info(id);
        CHECK(other&&info->state_floats==other->state_floats&&info->weight_floats==other->weight_floats);
    }
    size_t n=info->state_floats,wn=info->weight_floats;
    float *weights=malloc(wn*sizeof(float));CHECK(weights);
    FILE *wf=fopen(argv[4],"rb");CHECK(wf);
    CHECK(fread(weights,sizeof(float),wn,wf)==wn && fgetc(wf)==EOF);fclose(wf);
    dpdfnet_model *m[libraries];
    for(int j=0;j<libraries;++j){m[j]=a[j].create(id,weights,wn);CHECK(m[j]);}
    free(weights);
    float *storage[libraries][2],*s[libraries],*t[libraries];
    for(int j=0;j<libraries;++j){
        for(int k=0;k<2;++k){storage[j][k]=malloc((n+2)*sizeof(float));CHECK(storage[j][k]);
            storage[j][k][0]=12345;storage[j][k][n+1]=-12345;}
        s[j]=storage[j][0]+1;t[j]=storage[j][1]+1;CHECK(!a[j].reset(id,s[j]));
    }
    FILE *input=fopen(argv[5],"rb");CHECK(input);int status=0;
    float frame[962],x[libraries][962],y[libraries][962],first[8][962];
    uint64_t prefix[libraries],replay[libraries],whole[libraries];size_t hops=0;
    memset(prefix,0,sizeof(prefix));memset(replay,0,sizeof(replay));memset(whole,0,sizeof(whole));
    for(;;){
        size_t got=fread(frame,sizeof(float),962,input);if(!got)break;CHECK(got==962);
        if(hops<8)memcpy(first[hops],frame,sizeof(frame));
        float *outs[libraries],*states[libraries];
        for(int j=0;j<libraries;++j){
            memcpy(x[j],frame,sizeof(frame));outs[j]=(hops&1)?x[j]:y[j];
            states[j]=(hops&2)?s[j]:t[j];
            CHECK(!a[j].process(m[j],x[j],s[j],outs[j],states[j]));
            for(size_t k=0;k<962;++k)CHECK(isfinite(outs[j][k]));
            for(size_t k=0;k<n;++k)CHECK(isfinite(states[j][k]));
            if(hops<8){prefix[j]=digest(prefix[j],outs[j],962);prefix[j]=digest(prefix[j],states[j],n);}
            whole[j]=digest(whole[j],outs[j],962);whole[j]=digest(whole[j],states[j],n);
        }
        for(int j=1;j<libraries;++j){
            if(memcmp(outs[0],outs[j],sizeof(frame)) || memcmp(states[0],states[j],n*sizeof(float))){
                fprintf(stderr,"Model %d: candidate %d mismatch at hop %zu, input %s\n",id,j,hops,argv[5]);status=1;goto release;
            }
        }
        for(int j=0;j<libraries;++j){
            if(states[j]==t[j]){float *tmp=s[j];s[j]=t[j];t[j]=tmp;}
            for(int k=0;k<2;++k)CHECK(storage[j][k][0]==12345 && storage[j][k][n+1]==-12345);
        }
        ++hops;
    }
    CHECK(!ferror(input) && hops>=8);fclose(input);input=NULL;
    for(int j=1;j<libraries;++j)CHECK(whole[j]==whole[0]);
    for(int j=0;j<libraries;++j){
        CHECK(!a[j].reset(id,s[j]));
        for(int h=0;h<8;++h){
            CHECK(!a[j].process(m[j],first[h],s[j],y[j],s[j]));
            replay[j]=digest(replay[j],y[j],962);replay[j]=digest(replay[j],s[j],n);
        }
        CHECK(replay[j]==prefix[j]);
    }
release:
    if(input)fclose(input);
    for(int j=0;j<libraries;++j){
        a[j].destroy(m[j]);
        for(int k=0;k<2;++k)free(storage[j][k]);
        dlclose(a[j].library);
    }
    if(status)return status;
    printf("{\"model\":%d,\"hops\":%zu,\"candidates\":%d,\"every_spectrum_and_state_bitwise_equal\":true,\"reset_replay\":true,\"four_alias_modes\":true,\"finite\":true,\"canaries\":true,\"diagnostic_digest\":\"%016llx\"}\n",id,hops,libraries-1,(unsigned long long)whole[0]);
    return 0;
}
