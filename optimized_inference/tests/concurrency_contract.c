#define _POSIX_C_SOURCE 200809L
#include "../tools/support.h"
#include <pthread.h>
#include <string.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"Concurrency contract line %d\n",__LINE__);exit(1);}}while(0)
enum {WORKERS=4,HOPS=64};
typedef struct {
    dpdfnet_model *serial,*parallel;
    float *serial_state,*parallel_state;
    float input[962],serial_output[962],parallel_output[962];
    const dpdfnet_model_info *info;
    dpdfnet_model_id id;
    pthread_barrier_t *start,*done;
} worker;
static void *process(void *opaque) {
    worker *w=opaque;
    for(int hop=0;hop<HOPS;++hop){
        pthread_barrier_wait(w->start);
        CHECK(!dpdfnet_process(w->parallel,w->input,w->parallel_state,w->parallel_output,w->parallel_state));
        pthread_barrier_wait(w->done);
    }
    return NULL;
}
int main(int argc,char **argv) {
    CHECK(argc==3);
    if(!dpdfnet_cpu_supported())return 77;
    pthread_barrier_t start,done;
    CHECK(!pthread_barrier_init(&start,NULL,WORKERS+1));CHECK(!pthread_barrier_init(&done,NULL,WORKERS+1));
    worker w[WORKERS]={0};pthread_t threads[WORKERS];
    for(int i=0;i<WORKERS;++i){
        w[i].id=i%2?DPDFNET_MODEL_8:DPDFNET_MODEL_2;
        w[i].info=dpdfnet_get_model_info(w[i].id);w[i].start=&start;w[i].done=&done;
        float *weights=read_weights(argv[i%2+1],w[i].info);CHECK(weights);
        w[i].serial=dpdfnet_create(w[i].id,weights,w[i].info->weight_floats);
        w[i].parallel=dpdfnet_create(w[i].id,weights,w[i].info->weight_floats);free(weights);
        CHECK(w[i].serial&&w[i].parallel);
        w[i].serial_state=malloc(w[i].info->state_floats*sizeof(float));
        w[i].parallel_state=malloc(w[i].info->state_floats*sizeof(float));
        CHECK(w[i].serial_state&&w[i].parallel_state);
        CHECK(!dpdfnet_init_state(w[i].id,w[i].serial_state));CHECK(!dpdfnet_init_state(w[i].id,w[i].parallel_state));
        CHECK(!pthread_create(threads+i,NULL,process,w+i));
    }
    for(int hop=0;hop<HOPS;++hop){
        for(int i=0;i<WORKERS;++i){
            if(hop==HOPS/2){CHECK(!dpdfnet_init_state(w[i].id,w[i].serial_state));CHECK(!dpdfnet_init_state(w[i].id,w[i].parallel_state));}
            synthetic_spectrum(w[i].input,hop,(hop+i)%8==0?0:((i+1)*.25f));
        }
        pthread_barrier_wait(&start);
        for(int i=0;i<WORKERS;++i)CHECK(!dpdfnet_process(w[i].serial,w[i].input,w[i].serial_state,w[i].serial_output,w[i].serial_state));
        pthread_barrier_wait(&done);
        for(int i=0;i<WORKERS;++i){
            CHECK(!memcmp(w[i].serial_output,w[i].parallel_output,sizeof(w[i].serial_output)));
            CHECK(!memcmp(w[i].serial_state,w[i].parallel_state,w[i].info->state_floats*sizeof(float)));
        }
    }
    for(int i=0;i<WORKERS;++i){CHECK(!pthread_join(threads[i],NULL));dpdfnet_destroy(w[i].serial);dpdfnet_destroy(w[i].parallel);free(w[i].serial_state);free(w[i].parallel_state);}
    CHECK(!pthread_barrier_destroy(&start));CHECK(!pthread_barrier_destroy(&done));
    puts("Both models: four concurrent contexts match every serial spectrum/state, including reset.");return 0;
}
