/* Diagnostic C-only loop: all buffers and records are allocated by the caller.
 * No model change, RT scheduling request, allocation, printing or Python call
 * occurs during this loop. Linux guest observations do not diagnose host stalls.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <sched.h>
#include <sys/resource.h>
#include <time.h>

typedef int (*process_fn)(void *,const float *,const float *,float *,float *);
static int64_t stamp(clockid_t clock) {
    struct timespec t;
    clock_gettime(clock,&t);
    return (int64_t)t.tv_sec*1000000000+t.tv_nsec;
}

int tail_loop(process_fn process,void *model,const float *frames,float *state,
              float *output,int count,int stride,int64_t period,int64_t *records) {
    int64_t start=stamp(CLOCK_MONOTONIC)+period;
    for (int i=0;i<count;++i) {
        int64_t deadline=start+(int64_t)i*period;
        struct timespec target={deadline/1000000000,deadline%1000000000};
        int rc;
        do { rc=clock_nanosleep(CLOCK_MONOTONIC,TIMER_ABSTIME,&target,0); } while (rc==EINTR);
        if (rc) return -100-rc;
        int64_t release=stamp(CLOCK_MONOTONIC);
        struct rusage before,after;
        getrusage(RUSAGE_THREAD,&before);
        int cpu0=sched_getcpu();
        int64_t wall0=stamp(CLOCK_MONOTONIC),cpu_start=stamp(CLOCK_THREAD_CPUTIME_ID);
        rc=process(model,frames+(int64_t)i*stride,state,output,state);
        int64_t cpu_end=stamp(CLOCK_THREAD_CPUTIME_ID),wall1=stamp(CLOCK_MONOTONIC);
        int cpu1=sched_getcpu();
        getrusage(RUSAGE_THREAD,&after);
        if (rc) return rc;
        int64_t *r=records+(int64_t)i*13;
        r[0]=wall1-wall0; r[1]=cpu_end-cpu_start;
        r[2]=release-deadline; r[3]=wall1-deadline;
        r[4]=cpu0; r[5]=cpu1;
        r[6]=after.ru_nvcsw-before.ru_nvcsw;
        r[7]=after.ru_nivcsw-before.ru_nivcsw;
        r[8]=after.ru_minflt-before.ru_minflt;
        r[9]=after.ru_majflt-before.ru_majflt;
        r[10]=0; r[11]=0; r[12]=i;
    }
    return 0;
}
