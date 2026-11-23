#define _GNU_SOURCE
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <unistd.h>
#ifdef DPDF_TEST_AVX_VNNI
#include "../src/common/int8_vnni.c"
#else
#include "../src/common/int8.c"
#endif
#include "quantize_reference.h"

static uint32_t seed=7711;
static float sample(void) {
    seed=seed*1664525u+1013904223u;
    return (int32_t)seed*0x1p-30f;
}
static void fill(float *x,int mask,int pattern) {
    for (int r=0;r<4;++r) for (int j=0;j<64;++j) {
        float v=sample();
        if (mask & (1<<r)) v=(j&1)?0.0f:-0.0f;
        else if (pattern==1) v=.5f;
        else if (pattern==2) v=-.5f;
        else if (pattern==3) v=(j&1)?1.f:-1.f;
        else if (pattern==4) v*=1e-30f;
        else if (pattern==5) v*=0x1p-130f;
        else if (pattern==6) v*=0x1p120f;
        else if (pattern>=7 && pattern<71) v=j==pattern-7 ? ((r&1)?-1.f:1.f) : 0;
        else if (pattern==71) v=(j&1)?0x1.fffffep-1f:-0x1.000002p0f;
        else if (pattern==72) v=(float)(j-32)*0x1p-140f;
        else if (pattern==73) v=(float)(j-32)*0x1p-120f;
        else if (pattern==74) v=(float)(j-32)/127.f;
        x[r*64+j]=v;
    }
}
int main(void) {
#ifdef DPDF_TEST_AVX_VNNI
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avxvnni")) return 77;
#endif
    long page=sysconf(_SC_PAGESIZE);
    if (page<2048) return 2;
    unsigned saved=_mm_getcsr();
    unsigned char *a=mmap(0,page*3,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    unsigned char *b=mmap(0,page*3,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (a==MAP_FAILED || b==MAP_FAILED || mprotect(a+page,page,PROT_READ|PROT_WRITE) ||
        mprotect(b+page,page,PROT_READ|PROT_WRITE)) return 2;
    unsigned rounds[]={_MM_ROUND_NEAREST,_MM_ROUND_DOWN,_MM_ROUND_UP,_MM_ROUND_TOWARD_ZERO};
    int cases=0,flag_differences=0;
    for (int place=0;place<3;++place) for (int mode=0;mode<4;++mode)
      for (int denorm=0;denorm<4;++denorm) for (int mask=0;mask<16;++mask)
        for (int pattern=0;pattern<96;++pattern) {
            float *x=(float *)(place==0 ? a+page : a+2*page-1024-(place==2?4:0));
            int8_t *actual=(int8_t *)(place==0 ? b+page : b+2*page-256-(place==2?1:0));
            int8_t reference[256],single[256];float backup[256],ref[4],ss[4];int rz[4],sz[4];
            struct {uint32_t head;float values[4];uint32_t tail;} s={0x1937,{0},0x7291};
            struct {uint32_t head;int values[4];uint32_t tail;} z={0x5173,{0},0x9371};
            unsigned control=(saved&~(0x603f|0x8040))|rounds[mode]|
                ((denorm&1)?0x8000:0)|((denorm&2)?0x40:0);
            _mm_setcsr(control);
            fill(x,mask,pattern);memcpy(backup,x,sizeof(backup));
            _mm_setcsr(control);
            for (int r=0;r<4;++r) reference_quantize(x+r*64,reference+r*64,64,ref+r,rz+r);
            unsigned reference_flags=_mm_getcsr()&63;
            _mm_setcsr(control);memset(b+page,0x55,(size_t)page);
            quantize4_64(x,actual,s.values,z.values);
            unsigned actual_csr=_mm_getcsr();
            flag_differences+=reference_flags!=(actual_csr&63);
            if ((actual_csr&~63u)!=control || memcmp(actual,reference,256) ||
                memcmp(s.values,ref,sizeof(ref)) || memcmp(z.values,rz,sizeof(rz)) ||
                memcmp(x,backup,sizeof(backup)) || s.head!=0x1937 || s.tail!=0x7291 ||
                z.head!=0x5173 || z.tail!=0x9371 || (mask==15 && (actual_csr&5))) goto fail;
            for (unsigned char *p=b+page;p<b+2*page;++p)
                if ((p<(unsigned char *)actual || p>=(unsigned char *)actual+256) && *p!=0x55) goto fail;
            _mm_setcsr(control);
            for (int r=0;r<4;++r) quantize(x+r*64,single+r*64,64,ss+r,sz+r);
            if (memcmp(single,reference,256)||memcmp(ss,ref,sizeof(ref))||memcmp(sz,rz,sizeof(rz))) goto fail;
            cases++;continue;
fail:
            _mm_setcsr(saved);
            fprintf(stderr,"quantize4 mismatch place=%d mode=%d denorm=%d mask=%d pattern=%d\n",place,mode,denorm,mask,pattern);
            return 1;
        }
    _mm_setcsr(saved);munmap(a,page*3);munmap(b,page*3);
    printf("{\"cases\":%d,\"empty_masks\":16,\"rounding_modes\":4,\"FTZ_DAZ_combinations\":4,\"guarded_both_ends\":true,\"bitwise\":true,\"sticky_flag_differences_diagnostic_only\":%d}\n",cases,flag_differences);
    return 0;
}
