/* Linux microbenchmark: compile with cc -O2 affine_benchmark.c -ldl -lm.
 * Arguments: baseline.so candidate.so [other.so]. Each library must expose
 * the internal dpdf_qcreate/qaffine/qdestroy API. No model weights required. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    void *lib,*matrix;
    void *(*create)(const float *,int,int);
    void (*affine)(const void *,const float *,const float *,float *,int);
    void (*destroy)(void *);
    int (*available)(void);
} backend;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e9+t.tv_nsec;
}

int main(int argc,char **argv) {
    if (argc<3 || argc>4) return 2;
    int count=argc-1;
    backend b[3]={0};
    for (int i=0;i<count;++i) {
        b[i].lib=dlopen(argv[i+1],RTLD_NOW|RTLD_LOCAL);
        if (!b[i].lib) { fprintf(stderr,"%s\n",dlerror()); return 1; }
        b[i].create=dlsym(b[i].lib,"dpdf_qcreate");
        b[i].affine=dlsym(b[i].lib,"dpdf_qaffine");
        b[i].destroy=dlsym(b[i].lib,"dpdf_qdestroy");
        b[i].available=dlsym(b[i].lib,"dpdf_has_avx2");
        if (!b[i].create || !b[i].affine || !b[i].destroy ||
            !b[i].available || !b[i].available()) return 1;
    }
    const int shapes[][2]={{64,192},{128,64},{256,768},{512,768}};
    puts("{\"iterations\":20000,\"warmup\":1000,\"repeats\":7,\"rows\":[");
    int first=1;
    for (size_t shape=0;shape<sizeof(shapes)/sizeof(shapes[0]);++shape) {
        int k=shapes[shape][0],n=shapes[shape][1];
        float *w=malloc((size_t)k*n*sizeof(float)),*x=malloc(k*sizeof(float));
        float *bias=malloc(n*sizeof(float)),*y=malloc(n*sizeof(float)),*ref=malloc(n*sizeof(float));
        if (!w || !x || !bias || !y || !ref) return 1;
        for (int j=0;j<k*n;++j) w[j]=((j*13%127)-63)/64.0f;
        for (int j=0;j<k;++j) x[j]=((j*7%61)-30)/32.0f;
        for (int j=0;j<n;++j) bias[j]=j/1024.0f;
        for (int i=0;i<count;++i) {
            b[i].matrix=b[i].create(w,k,n);
            if (!b[i].matrix) return 1;
            b[i].affine(b[i].matrix,x,bias,y,1);
            if (i==0) memcpy(ref,y,n*sizeof(float));
            else if (memcmp(ref,y,n*sizeof(float))) return 1;
        }
        for (int repeat=0;repeat<7;++repeat) for (int offset=0;offset<count;++offset) {
            int i=(repeat+offset)%count;
            for (int j=0;j<1000;++j) b[i].affine(b[i].matrix,x,bias,y,1);
            double start=now();
            for (int j=0;j<20000;++j) b[i].affine(b[i].matrix,x,bias,y,1);
            double elapsed=(now()-start)/20000;
            printf("%s{\"k\":%d,\"n\":%d,\"backend\":%d,\"repeat\":%d,\"ns_per_call\":%.3f}",
                   first?"":",\n",k,n,i,repeat,elapsed);
            first=0;
        }
        for (int i=0;i<count;++i) b[i].destroy(b[i].matrix);
        free(w);free(x);free(bias);free(y);free(ref);
    }
    puts("\n]}");
    for (int i=0;i<count;++i) dlclose(b[i].lib);
    return 0;
}
