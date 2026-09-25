#include "dpdfnet2_48khz_hr/generated_model.h"
#include "dpdfnet8_48khz_hr/generated_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while (0)

/* Independently exercise the existing experimental interface to guard the
 * named preset's exact precision scope, including after recurrent updates. */
#define LEGACY_ADAPTER(prefix) \
static void *prefix##_legacy_create(const float *w,size_t n,int preset) { \
    return prefix##_create_config(w,n,preset?4:0,preset?8:0,preset?7:0); \
} \
static int prefix##_legacy_process(void *m,const float *x,const float *s,float *y,float *t) { \
    return prefix##_process((prefix *)m,x,s,y,t); \
} \
static void prefix##_legacy_destroy(void *m) { prefix##_destroy((prefix *)m); }
LEGACY_ADAPTER(dpdfnet2_48khz_hr)
LEGACY_ADAPTER(dpdfnet8_48khz_hr)

int main(int argc,char **argv) {
    CHECK(argc==3);
    /* Disk format requires little-endian IEEE754 float32 on this test host. */
    unsigned endian=1; CHECK(*(unsigned char *)&endian==1 && sizeof(float)==4);
    const dpdf_native_api_v1 *apis[]={dpdfnet2_48khz_hr_get_api(DPDF_NATIVE_ABI_VERSION),
                                   dpdfnet8_48khz_hr_get_api(DPDF_NATIVE_ABI_VERSION)};
    CHECK(!dpdfnet2_48khz_hr_get_api(0) && !dpdfnet8_48khz_hr_get_api(2));
    void *(*legacy_create[])(const float *,size_t,int)={
        dpdfnet2_48khz_hr_legacy_create,dpdfnet8_48khz_hr_legacy_create};
    int (*legacy_process[])(void *,const float *,const float *,float *,float *)={
        dpdfnet2_48khz_hr_legacy_process,dpdfnet8_48khz_hr_legacy_process};
    void (*legacy_destroy[])(void *)={dpdfnet2_48khz_hr_legacy_destroy,dpdfnet8_48khz_hr_legacy_destroy};
    const size_t expected_states[]={56436,90228};
    const char *names[]={"dpdfnet2_48khz_hr","dpdfnet8_48khz_hr"};
    for (int preset=DPDF_PRESET_FP32;preset<=DPDF_PRESET_INT8_SELECTIVE;++preset) {
        dpdf_native_model *models[2]={NULL,NULL}; void *legacy[2]={NULL,NULL};
        float *states[2]={NULL,NULL},*reference[2]={NULL,NULL},*initial[2]={NULL,NULL};
        float first[2][962];
        for (int i=0;i<2;++i) {
            const dpdf_native_api_v1 *a=apis[i]; CHECK(a);
            CHECK(a->abi_version==1 && a->struct_size==sizeof(*a));
            CHECK(!strcmp(a->model_name,names[i])); CHECK(strlen(a->weights_sha256)==64);
            CHECK(a->sample_rate==48000 && a->hop_size==480 && a->spectrum_size==962);
            CHECK(a->state_size==expected_states[i] && a->weight_count>0);
            CHECK(a->preset_supported(DPDF_PRESET_FP32)); CHECK(!a->preset_supported(999));
            CHECK(a->preset_supported(DPDF_PRESET_INT8_SELECTIVE)==dpdf_has_avx2());
            CHECK(a->init_state(NULL)==-1); a->destroy(NULL); CHECK(!a->owned_bytes(NULL));
            size_t n=a->weight_count;
            float *w=malloc(n*sizeof(float)); CHECK(w);
            FILE *f=fopen(argv[i+1],"rb"); CHECK(f);
            CHECK(fread(w,sizeof(float),n,f)==n && fgetc(f)==EOF); fclose(f);
            CHECK(!a->create(w,n,999) && !a->create(w,n-1,preset) && !a->create(NULL,n,preset));
            if (!a->preset_supported(preset)) {
                CHECK(!a->create(w,n,preset)); free(w); continue;
            }
            float saved=w[0]; w[0]=NAN; CHECK(!a->create(w,n,preset)); w[0]=saved;
            models[i]=a->create(w,n,preset); legacy[i]=legacy_create[i](w,n,preset);
            CHECK(models[i] && legacy[i] && a->owned_bytes(models[i])>0);
            /* Weights must not be borrowed after successful construction. */
            memset(w,0,n*sizeof(float)); free(w);
            states[i]=malloc(a->state_size*sizeof(float));
            reference[i]=malloc(a->state_size*sizeof(float));
            initial[i]=malloc(a->state_size*sizeof(float));
            CHECK(states[i] && reference[i] && initial[i]);
            CHECK(!a->init_state(states[i]) && !a->init_state(reference[i]));
            memcpy(initial[i],states[i],a->state_size*sizeof(float));
        }
        /* Both models are alive in one binary and process alternating hops.
         * Test aliasing against independent out-of-place recurrent output. */
        for (int frame=0;frame<16;++frame) for (int i=0;i<2;++i) {
            if (!models[i]) continue;
            const dpdf_native_api_v1 *a=apis[i];
            float x[962],y[962],z[962];
            for (int j=0;j<962;++j) x[j]=frame==3?0:0.1f*sinf(j*.31f+frame+i);
            memcpy(y,x,sizeof(x));
            CHECK(a->process(NULL,x,states[i],y,states[i])==-1);
            CHECK(a->process(models[i],NULL,states[i],y,states[i])==-1);
            CHECK(a->process(models[i],x,NULL,y,states[i])==-1);
            CHECK(a->process(models[i],x,states[i],NULL,states[i])==-1);
            CHECK(a->process(models[i],x,states[i],y,NULL)==-1);
            CHECK(!a->process(models[i],y,states[i],y,states[i]));
            CHECK(!legacy_process[i](legacy[i],x,reference[i],z,reference[i]));
            CHECK(!memcmp(y,z,sizeof(y)));
            CHECK(!memcmp(states[i],reference[i],a->state_size*sizeof(float)));
            for (int j=0;j<962;++j) CHECK(isfinite(y[j]));
            for (size_t j=0;j<a->state_size;++j) CHECK(isfinite(states[i][j]));
            if (!frame) memcpy(first[i],y,sizeof(y));
        }
        for (int i=0;i<2;++i) {
            if (!models[i]) continue;
            const dpdf_native_api_v1 *a=apis[i]; float x[962],y[962];
            CHECK(!a->init_state(states[i]));
            CHECK(!memcmp(states[i],initial[i],a->state_size*sizeof(float)));
            for (int j=0;j<962;++j) x[j]=0.1f*sinf(j*.31f+i);
            CHECK(!a->process(models[i],x,states[i],y,states[i]));
            CHECK(!memcmp(first[i],y,sizeof(y)));
            a->destroy(models[i]); legacy_destroy[i](legacy[i]);
            free(states[i]); free(reference[i]); free(initial[i]);
        }
    }
    puts("Both models: ABI negotiation, named presets, legacy parity, state reset and ownership passed");
    return 0;
}
