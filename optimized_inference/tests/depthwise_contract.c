/* Independent ordered scalar oracle, exact input bounds and output canaries.
 * All rounding modes and all FTZ/DAZ combinations are exercised. */
#define _GNU_SOURCE
#include "full_ops.h"
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xmmintrin.h>

#define CHECK(x) do {if(!(x)){fprintf(stderr,"Depthwise contract line %d\n",__LINE__);return 1;}}while(0)
static uint32_t random_state=0x194812;
static float sample(void) {
    random_state=1664525u*random_state+1013904223u;
    return ((int)(random_state>>16)-32768)/8192.0f;
}
static void reference(const float *x,const float *w,const float *bias,float *y,
                      int ci,int wi,int co,int wo,int sw) {
    for(int oc=0;oc<co;++oc)for(int ow=0;ow<wo;++ow){
        float value=bias?bias[oc]:0;
        for(int k=0;k<3;++k){
            int ix=ow*sw+k-1;
            if(ix>=0 && ix<wi)value+=w[oc*3+k]*x[(oc/(co/ci))*wi+ix];
        }
        y[oc*wo+ow]=value;
    }
}
int main(void) {
    if(!dpdf_has_avx2()) return 77;
    const int widths[]={8,9,15,16,17,23,24,25,47,48,49,95,96,97,479,480,481};
    const int channels[]={1,3,64},rounding[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    unsigned original=_mm_getcsr(); int original_round=fegetround(),cases=0;
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    for(int ri=0;ri<4;++ri)for(int denorm=0;denorm<4;++denorm){
        CHECK(!fesetround(rounding[ri]));
        unsigned csr=_mm_getcsr();csr&=~((1u<<15)|(1u<<6));
        if(denorm&1)csr|=1u<<15;
        if(denorm&2)csr|=1u<<6;
        _mm_setcsr(csr);
        for(size_t ix=0;ix<sizeof(widths)/sizeof(widths[0]);++ix)
        for(int ic=0;ic<3;++ic)for(int multiplier=1;multiplier<=2;++multiplier)
        for(int sw=2;sw<=3;++sw)for(int use_bias=0;use_bias<2;++use_bias)for(int pattern=0;pattern<4;++pattern){
            int wi=widths[ix],ci=channels[ic],co=ci*multiplier,wo=(wi-1)/sw+1;
            size_t elements=(size_t)ci*wi,bytes=elements*sizeof(float),pages=(bytes+page-1)/page;
            unsigned char *mapping=mmap(NULL,(pages+2)*page,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
            CHECK(mapping!=MAP_FAILED);CHECK(!mprotect(mapping+page,pages*page,PROT_READ|PROT_WRITE));
            /* Exact end boundary; varying lengths also vary vector alignment. */
            float *x=(float *)(mapping+(pages+1)*page-bytes);
            float *wa=malloc(((size_t)co*3+1)*sizeof(float)),*ba=malloc((co+1)*sizeof(float));
            float *actual=malloc(((size_t)co*wo+2)*sizeof(float)),*expected=malloc((size_t)co*wo*sizeof(float));
            CHECK(wa&&ba&&actual&&expected);float *w=wa+1,*bias=ba+1;
            for(size_t j=0;j<elements;++j)x[j]=pattern==0?sample():pattern==1?(j&1?-0.0f:+0.0f):
                pattern==2?(j&1?-0x1p-140f:0x1p-140f):(j&1?100.0f:-100.0f);
            for(int j=0;j<co*3;++j)w[j]=pattern==2?.5f:sample();
            for(int j=0;j<co;++j)bias[j]=pattern==1?-0.0f:pattern==2?0:sample();
            actual[0]=12345;actual[co*wo+1]=-12345;
            reference(x,w,use_bias?bias:NULL,expected,ci,wi,co,wo,sw);
            CHECK(!mprotect(mapping+page,pages*page,PROT_READ));
            CHECK(dpdf_depthwise_stride_avx2(x,w,use_bias?bias:NULL,actual+1,ci,1,wi,co,1,wo,1,3,1,sw,0,1,ci)==1);
            if(memcmp(actual+1,expected,(size_t)co*wo*sizeof(float))){
                fprintf(stderr,"wi=%d ci=%d co=%d sw=%d bias=%d pattern=%d rounding=%d denorm=%d\n",wi,ci,co,sw,use_bias,pattern,ri,denorm);
                return 1;
            }
            CHECK(actual[0]==12345&&actual[co*wo+1]==-12345);
            free(wa);free(ba);free(actual);free(expected);CHECK(!munmap(mapping,(pages+2)*page));++cases;
        }
    }
    CHECK(!fesetround(original_round));_mm_setcsr(original);
    printf("%d exact guarded depthwise cases\n",cases);return 0;
}
