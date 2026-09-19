/* Linux-only measurement harness. A fresh process per configuration avoids
 * allocator-history bias. Input weights are unmapped before RSS is sampled. */
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static size_t memory_status(const char *key) {
    FILE *f=fopen("/proc/self/status","r"); if (!f) return 0;
    char line[256];size_t kb=0;
    while (fgets(line,sizeof(line),f)) if (!strncmp(line,key,strlen(key)) && sscanf(line+strlen(key)," %zu kB",&kb)==1) break;
    fclose(f);return kb*1024;
}
int main(int argc,char **argv) {
    if (argc!=6) return 2;
    void *lib=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL); if (!lib) { fputs(dlerror(),stderr);return 3; }
    void *(*create)(const float *,size_t,int)=dlsym(lib,"dpdf_model_create");
    void *(*config)(const float *,size_t,int,int,unsigned)=dlsym(lib,"dpdf_model_create_config");
    void (*destroy)(void *)=dlsym(lib,"dpdf_model_destroy");
    int (*init)(float *)=dlsym(lib,"dpdf_model_init_state");
    size_t (*state_size)(void)=dlsym(lib,"dpdf_model_state_size");
    int (*process)(void *,const float *,const float *,float *,float *)=dlsym(lib,"dpdf_model_process");
    size_t (*owned)(const void *)=dlsym(lib,"dpdf_model_owned_bytes");
    if (!create || !destroy || !init || !process || !state_size) return 4;
    float *state=calloc(state_size(),sizeof(float)),x[962]={0},y[962];if (!state) return 5;
    init(state);size_t before=memory_status("VmRSS:");
    int fd=open(argv[2],O_RDONLY);struct stat st;
    if (fd<0 || fstat(fd,&st) || st.st_size<=0 || st.st_size%4) return 6;
    float *w=mmap(NULL,(size_t)st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);if (w==MAP_FAILED) return 7;
    int tier=atoi(argv[3]),precision=atoi(argv[4]);unsigned mask=(unsigned)atoi(argv[5]);
    void *m=config?config(w,(size_t)st.st_size/4,tier,precision,mask):create(w,(size_t)st.st_size/4,tier);
    if (!m) return 8;
    size_t peak=memory_status("VmHWM:");
    munmap(w,(size_t)st.st_size);
    for (int i=0;i<120;++i) { x[i%962]=.01f; if (process(m,x,state,y,state)) return 9; }
    size_t after=memory_status("VmRSS:"),end_peak=memory_status("VmHWM:");
    if (end_peak>peak) peak=end_peak;
    printf("{\"rss_before_bytes\":%zu,\"rss_after_bytes\":%zu,\"rss_delta_bytes\":%zu,\"peak_rss_bytes\":%zu,\"owned_bytes\":%zu}\n",before,after,after>before?after-before:0,peak,owned?owned(m):0);
    destroy(m);free(state);dlclose(lib);return 0;
}
