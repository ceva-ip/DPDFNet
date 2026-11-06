#define _POSIX_C_SOURCE 200809L
#include "internal.h"
#include "full_ops.h"
#include "int8_matrix.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint32_t seed=20261026;
static float sample(void) {
    seed=1664525u*seed+1013904223u;
    return ((int)(seed>>16)-32768)/32768.0f;
}
static int64_t ns(clockid_t clock) {
    struct timespec t;
    if(clock_gettime(clock,&t)) exit(2);
    return (int64_t)t.tv_sec*INT64_C(1000000000)+t.tv_nsec;
}

int main(int argc,char **argv) {
    if(argc!=3 || !dpdf_has_avx2()) return 2;
    int iterations=atoi(argv[2]);
    if(iterations<100 || iterations>10000000) return 2;
    int kind=-1;
    const char *names[]={"depthwise2","depthwise3","pair40","pair48","batch40","row64",
                         "gates1","gates40","gates48","norm40","transpose"};
    for(int i=0;i<11;++i) if(!strcmp(argv[1],names[i])) kind=i;
    if(kind<0) return 2;
    const size_t count=64*480;
    float *x=malloc(count*sizeof(float)),*b=malloc(count*sizeof(float));
    float *w=malloc(count*sizeof(float)),*y=malloc(count*sizeof(float)),*z=malloc(count*sizeof(float));
    if(!x||!b||!w||!y||!z) return 2;
    for(size_t i=0;i<count;++i){x[i]=sample();b[i]=sample();w[i]=sample();y[i]=0;z[i]=0;}
    dpdf_qmatrix *q0=dpdf_qcreate(w,64,192),*q1=dpdf_qcreate(w+64*192,64,192);
    if(!q0||!q1) return 2;
    int64_t start=0,cpu_start=0,warm_deadline=ns(CLOCK_MONOTONIC)+INT64_C(100000000);
    for(int i=-1000;i<iterations;++i) {
        if(i==0 && ns(CLOCK_MONOTONIC)<warm_deadline) i=-1000;
        if(i==0){cpu_start=ns(CLOCK_THREAD_CPUTIME_ID);start=ns(CLOCK_MONOTONIC);}
        switch(kind) {
        case 0: case 1: {
            int stride=kind==0?2:3;
            if(!dpdf_depthwise_stride_avx2(x,w,b,y,64,1,480,64,1,480/stride,1,3,1,stride,0,1,64)) return 3;
            break;
        }
        case 2: case 3:
            dpdf_qaffine_pair(q0,q1,x,b,b,y,z,kind==2?40:48); break;
        case 4: dpdf_qaffine(q0,x,b,y,40); break;
        case 5: dpdf_qaffine(q0,x,b,y,1); break;
        case 6: case 7: case 8: dpdf_gates_avx2(x,b,w,y,kind==6?1:kind==7?40:48); break;
        case 9: dpdf_norm_residual_avx2(y,x,b,w,w+64,1e-5f,40); break;
        case 10: dpdf_transpose_avx2(x,y,64,48); break;
        }
    }
    int64_t elapsed=ns(CLOCK_MONOTONIC)-start,cpu=ns(CLOCK_THREAD_CPUTIME_ID)-cpu_start;
    uint32_t checksum=0;
    for(size_t i=0;i<count;++i){uint32_t value;memcpy(&value,y+i,4);checksum=checksum*33+value;}
    printf("{\"kernel\":\"%s\",\"iterations\":%d,\"mean_ms\":%.12f,\"cpu_ms\":%.12f,\"checksum\":%u,\"measurement\":\"hot-cache amortized call time, not per-call latency\"}\n",
           names[kind],iterations,(double)elapsed/iterations/1e6,(double)cpu/iterations/1e6,checksum);
    dpdf_qdestroy(q0);dpdf_qdestroy(q1);free(x);free(b);free(w);free(y);free(z);
    return 0;
}
