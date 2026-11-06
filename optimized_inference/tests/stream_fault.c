/* Test-only proxy: corrupt one intermediate spectrum, leaving later state and
 * spectra untouched. A final-frame-only comparator would miss this fault. */
#include "dpdfnet.h"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {dpdfnet_model *context;unsigned calls;} proxy;
static void *reference_library;
static const dpdfnet_model_info *(*reference_info)(dpdfnet_model_id);
static dpdfnet_model *(*reference_create)(dpdfnet_model_id,const float *,size_t);
static int (*reference_reset)(dpdfnet_model_id,float *);
static int (*reference_process)(dpdfnet_model *,const float *,const float *,float *,float *);
static void (*reference_destroy)(dpdfnet_model *);
__attribute__((constructor)) static void initialize(void) {
    const char *path=getenv("DPDF_FAULT_REFERENCE");
    void *library=path?dlopen(path,RTLD_NOW|RTLD_LOCAL):NULL;reference_library=library;
    if(!library){fprintf(stderr,"Fault proxy needs DPDF_FAULT_REFERENCE\n");abort();}
#define LOAD(field,name) do {*(void **)(&field)=dlsym(library,name);if(!field)abort();}while(0)
    LOAD(reference_info,"dpdfnet_get_model_info");LOAD(reference_create,"dpdfnet_create");
    LOAD(reference_reset,"dpdfnet_init_state");LOAD(reference_process,"dpdfnet_process");LOAD(reference_destroy,"dpdfnet_destroy");
#undef LOAD
}
__attribute__((destructor)) static void finish(void){if(reference_library)dlclose(reference_library);}
const dpdfnet_model_info *dpdfnet_get_model_info(dpdfnet_model_id id){return reference_info(id);}
int dpdfnet_init_state(dpdfnet_model_id id,float *state){return reference_reset(id,state);}
dpdfnet_model *dpdfnet_create(dpdfnet_model_id id,const float *weights,size_t n){
    proxy *p=calloc(1,sizeof(*p));if(!p)return NULL;
    p->context=reference_create(id,weights,n);if(!p->context){free(p);return NULL;}
    return (dpdfnet_model *)p;
}
int dpdfnet_process(dpdfnet_model *model,const float *x,const float *state,float *out,float *next){
    proxy *p=(proxy *)model;int result=reference_process(p->context,x,state,out,next);
    if(!result && ++p->calls==129){uint32_t bits;memcpy(&bits,out,4);bits^=1;memcpy(out,&bits,4);}
    return result;
}
void dpdfnet_destroy(dpdfnet_model *model){if(model){proxy *p=(proxy *)model;reference_destroy(p->context);free(p);}}
