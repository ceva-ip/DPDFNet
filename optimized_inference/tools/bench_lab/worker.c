/* One measurement executable for every library. Kernel CPU-time accounting is
 * outside the window; batch samples contain no clock calls between inferences. */
#define _GNU_SOURCE
#include "../../include/dpdfnet.h"
#include "../support.h"
#include <dlfcn.h>
#include <link.h>
#include <errno.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <immintrin.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/personality.h>

#define CHECK(x) do {if(!(x)){fprintf(stderr,"bench worker line %d: %s\n",__LINE__,#x);return 1;}}while(0)
static int64_t ns(clockid_t c) {struct timespec t;if(clock_gettime(c,&t))abort();return (int64_t)t.tv_sec*INT64_C(1000000000)+t.tv_nsec;}
static int cmp(const void *a,const void *b){double x=*(const double *)a,y=*(const double *)b;return(x>y)-(x<y);}
static int feedback(int cpu,uint64_t *ref,uint64_t *del) {
    char path[160];snprintf(path,sizeof(path),"/sys/devices/system/cpu/cpu%d/acpi_cppc/feedback_ctrs",cpu);
    FILE *f=fopen(path,"r");if(!f)return -1;
    int n=fscanf(f,"ref:%"SCNu64" del:%"SCNu64,ref,del);fclose(f);return n==2?0:-1;
}
static int probe(int cpu,int milliseconds) {
    uint64_t r0,d0,r1,d1;CHECK(!feedback(cpu,&r0,&d0));
    int64_t begin=ns(CLOCK_MONOTONIC_RAW),end=begin;
    __m256i a=_mm256_set1_epi32(17),b=_mm256_set1_epi32(12345);
    do {
        for(int j=0;j<100000;++j) {
            a=_mm256_xor_si256(_mm256_add_epi32(a,b),_mm256_srli_epi32(a,3));
            __asm__ volatile("" : "+x"(a));
        }
        end=ns(CLOCK_MONOTONIC_RAW);
    } while(end-begin<(int64_t)milliseconds*1000000);
    CHECK(!feedback(cpu,&r1,&d1));CHECK(r1>r0&&d1>d0);
    printf("{\"reference_delta\":%"PRIu64",\"delivered_delta\":%"PRIu64",\"duration_ms\":%.6f,\"checksum\":%d}\n",r1-r0,d1-d0,(double)(end-begin)/1e6,_mm256_extract_epi32(a,0));return 0;
}
/* Prefault readable ELF mappings, including the worker's not-yet-executed
 * measurement path, before warmup. No page locking or global VM changes. */
static int prefault(struct dl_phdr_info *info,size_t size,void *opaque) {
    (void)size;volatile unsigned char *sink=opaque;
    for(int i=0;i<info->dlpi_phnum;++i) {
        const ElfW(Phdr) *p=info->dlpi_phdr+i;
        if(p->p_type!=PT_LOAD||!(p->p_flags&PF_R))continue;
        const volatile unsigned char *base=(const volatile unsigned char *)(info->dlpi_addr+p->p_vaddr);
        for(size_t j=0;j<p->p_memsz;j+=4096)*sink^=base[j];
    }
    return 0;
}
static int perf_command(FILE *ctl,FILE *ack,const char *s) {
    if(!ctl)return 0;
    if(fprintf(ctl,"%s\n",s)<0||fflush(ctl))return -1;
    int c;do{c=fgetc(ack);}while(c==0);
    if(c!='a'||fgetc(ack)!='c'||fgetc(ack)!='k'||fgetc(ack)!='\n')return -1;
    return 0;
}
int main(int argc,char **argv) {
    if(argc==4&&!strcmp(argv[1],"probe"))return probe(atoi(argv[2]),atoi(argv[3]));
    CHECK(argc==9); /* library model weights input calls mode batch raw-samples */
    char *end;unsigned long calls=strtoul(argv[5],&end,10);CHECK(!*end&&calls>=16&&calls<=1000000);
    unsigned long batch=strtoul(argv[7],&end,10);CHECK(!*end&&batch>=1&&batch<=256&&calls%batch==0);
    int paced=!strcmp(argv[6],"paced")||!strncmp(argv[6],"paced:",6),batched=!strcmp(argv[6],"batch");
    CHECK(paced||batched||!strcmp(argv[6],"continuous"));CHECK(batched||batch==1);
    int64_t period_ns=10000000;
    if(!strncmp(argv[6],"paced:",6)) {
        unsigned long value=strtoul(argv[6]+6,&end,10);
        CHECK(!*end&&value>=1000000&&value<=100000000);period_ns=(int64_t)value;
    }
    void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!lib){fprintf(stderr,"%s\n",dlerror());return 1;}
    const dpdfnet_model_info *(*infofn)(dpdfnet_model_id);
    dpdfnet_model *(*create)(dpdfnet_model_id,const float *,size_t);
    int (*reset)(dpdfnet_model_id,float *);
    int (*process)(dpdfnet_model *,const float *,const float *,float *,float *);
    void (*destroy)(dpdfnet_model *);
#define SYM(x,n) do{*(void **)(&x)=dlsym(lib,n);CHECK(x);}while(0)
    SYM(infofn,"dpdfnet_get_model_info");SYM(create,"dpdfnet_create");SYM(reset,"dpdfnet_init_state");SYM(process,"dpdfnet_process");SYM(destroy,"dpdfnet_destroy");
#undef SYM
    dpdfnet_model_id id=(dpdfnet_model_id)atoi(argv[2]);const dpdfnet_model_info *info=infofn(id);CHECK(info&&info->spectrum_floats==962);
    float *w=read_weights(argv[3],info);CHECK(w);dpdfnet_model *m=create(id,w,info->weight_floats);free(w);CHECK(m);
    FILE *f=fopen(argv[4],"rb");CHECK(f&&!fseek(f,0,SEEK_END));long bytes=ftell(f);CHECK(bytes>0&&bytes%(962*4)==0);rewind(f);
    size_t frames=(size_t)bytes/(962*4);float *input=malloc((size_t)bytes);CHECK(input);CHECK(fread(input,1,(size_t)bytes,f)==(size_t)bytes);fclose(f);
    for(size_t i=0;i<(size_t)bytes/4;++i)CHECK(isfinite(input[i]));
    size_t observations=calls/batch;CHECK(observations>1);double *times=calloc(observations,sizeof(double));float *state=calloc(info->state_floats,sizeof(float));float output[962]={0};CHECK(times&&state&&!reset(id,state));
    FILE *ctl=NULL,*ack=NULL;const char *cp=getenv("DPDF_PERF_CONTROL"),*ap=getenv("DPDF_PERF_ACK");
    if(cp||ap){CHECK(cp&&ap);ctl=fopen(cp,"w");ack=fopen(ap,"r");CHECK(ctl&&ack);}
    for(size_t i=0;i<observations;++i)((volatile double *)times)[i]=0;
    CHECK(!prctl(PR_SET_TIMERSLACK,1,0,0,0));
    volatile unsigned char prefault_sink=0;dl_iterate_phdr(prefault,(void *)&prefault_sink);
    (void)ns(CLOCK_MONOTONIC_RAW);(void)ns(CLOCK_MONOTONIC);(void)ns(CLOCK_THREAD_CPUTIME_ID);
    /* Warm state, weights, code and output pages before either measurement. */
    const size_t warmup=128;
    for(size_t i=0;i<warmup;++i)CHECK(!process(m,input+(i%frames)*962,state,output,state));
    uint64_t ref0=0,del0=0,ref1=0,del1=0;int cpu=sched_getcpu();
    int have_feedback=!feedback(cpu,&ref0,&del0);
    struct rusage before,after;CHECK(!perf_command(ctl,ack,"enable"));CHECK(!getrusage(RUSAGE_SELF,&before));
    int64_t cpu0=ns(CLOCK_THREAD_CPUTIME_ID),deadline=ns(CLOCK_MONOTONIC);size_t over=0,late=0;
    for(size_t n=0;n<observations;++n) {
        if(paced){struct timespec t={deadline/1000000000,deadline%1000000000};int e;do{e=clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&t,NULL);}while(e==EINTR);CHECK(!e);}
        int64_t t0=ns(CLOCK_MONOTONIC_RAW);
        for(size_t j=0;j<batch;++j){size_t i=warmup+n*batch+j;CHECK(!process(m,input+(i%frames)*962,state,output,state));}
        int64_t t1=ns(CLOCK_MONOTONIC_RAW);
        times[n]=(double)(t1-t0)/1e6/batch;
        if(!batched)over+=times[n]>=10;
        if(paced){int64_t finish=ns(CLOCK_MONOTONIC);late+=finish>deadline+period_ns;deadline+=period_ns;if(finish>deadline)deadline=finish;}
    }
    int64_t cpu1=ns(CLOCK_THREAD_CPUTIME_ID);CHECK(!getrusage(RUSAGE_SELF,&after));CHECK(!perf_command(ctl,ack,"disable"));
    have_feedback=have_feedback&&!feedback(cpu,&ref1,&del1)&&ref1>ref0&&del1>del0;
    if(ctl)fclose(ctl);
    if(ack)fclose(ack);
    f=fopen(argv[8],"wb");CHECK(f);CHECK(fwrite(times,sizeof(double),observations,f)==observations);CHECK(!fclose(f));
    double mean=0,m2=0;
    for(size_t i=0;i<observations;++i){double delta=times[i]-mean;mean+=delta/(i+1);m2+=delta*(times[i]-mean);}
    qsort(times,observations,sizeof(double),cmp);uint64_t hash=UINT64_C(1469598103934665603);
    for(size_t i=0;i<info->state_floats;++i){uint32_t u;memcpy(&u,state+i,4);hash=(hash^u)*UINT64_C(1099511628211);}
    for(size_t i=0;i<962;++i){uint32_t u;memcpy(&u,output+i,4);hash=(hash^u)*UINT64_C(1099511628211);}
    printf("{\"layout_randomization_disabled\":%s,\"api_address\":\"%p\",\"state_address\":\"%p\",",(personality(0xffffffffUL)&ADDR_NO_RANDOMIZE)?"true":"false",(void *)process,(void *)state);
    printf("\"period_ns\":%"PRId64",\"cpu\":%d,\"uid\":%u,\"no_new_privs\":%d,\"reference_delta\":%"PRIu64",\"delivered_delta\":%"PRIu64",\"hardware_feedback_available\":%s,",paced?period_ns:INT64_C(0),cpu,(unsigned)getuid(),prctl(PR_GET_NO_NEW_PRIVS,0,0,0,0),ref1-ref0,del1-del0,have_feedback?"true":"false");
    printf("\"schema_version\":1,\"mode\":\"%s\",\"model\":%d,\"calls\":%lu,\"warmup\":%zu,\"observations\":%zu,\"batch_size\":%lu,\"observation_unit\":\"%s\",\"mean_ms\":%.12f,\"sample_sd_ms\":%.12f,\"cpu_ms_per_call\":%.12f,\"p95_ms\":%.12f,\"p99_ms\":%.12f,\"max_ms\":%.12f,\"over_10ms\":%zu,\"late\":%zu,\"voluntary_switches\":%ld,\"involuntary_switches\":%ld,\"minor_faults\":%ld,\"major_faults\":%ld,\"checksum\":\"%016"PRIx64"\"}\n",argv[6],id,calls,warmup,observations,batch,batched?"batch_mean":"call",mean,sqrt(m2/(observations-1)),(double)(cpu1-cpu0)/1e6/calls,times[(observations*95)/100],times[(observations*99)/100],times[observations-1],over,late,after.ru_nvcsw-before.ru_nvcsw,after.ru_nivcsw-before.ru_nivcsw,after.ru_minflt-before.ru_minflt,after.ru_majflt-before.ru_majflt,hash);
    free(times);free(state);free(input);destroy(m);dlclose(lib);return 0;
}
