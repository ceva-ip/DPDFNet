#define _GNU_SOURCE
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <unistd.h>
#include "../src/common/int8.c"
int main(void) {
 long page=sysconf(_SC_PAGESIZE);unsigned saved=_mm_getcsr();
 unsigned char *a=mmap(0,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 unsigned char *b=mmap(0,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 if(a==MAP_FAILED||b==MAP_FAILED||mprotect(a+page,page,PROT_NONE)||mprotect(b+page,page,PROT_NONE))return 2;
 float *x=(float *)(a+page-1028);int8_t *actual=(int8_t *)(b+page-256),reference[256];float s[4],ref[4];int z[4],rz[4];uint32_t seed=7711;int cases=0;
 unsigned rounds[]={_MM_ROUND_NEAREST,_MM_ROUND_DOWN,_MM_ROUND_UP,_MM_ROUND_TOWARD_ZERO};
 for(int mode=0;mode<4;mode++)for(int denorm=0;denorm<4;denorm++) {
  _mm_setcsr((saved&~(0x6000|0x8040))|rounds[mode]|((denorm&1)?0x8000:0)|((denorm&2)?0x40:0));
  for(int test=0;test<1024;test++) {
   for(int i=0;i<256;i++){seed=seed*1664525u+1013904223u;x[i]=(int32_t)seed*0x1p-30f;if(test%7==0)x[i]*=1e-30f;if(test%11==0)x[i]=(i&1)?0.0f:-0.0f;if(test%13==0&&i<64)x[i]=0;}
   for(int r=0;r<4;r++)quantize(x+r*64,reference+r*64,64,ref+r,rz+r);
   memset(actual,0x55,256);quantize4_64(x,actual,s,z);
   if(memcmp(actual,reference,256)||memcmp(s,ref,sizeof(s))||memcmp(z,rz,sizeof(z))){fprintf(stderr,"Mismatch mode=%d denorm=%d case=%d\n",mode,denorm,test);_mm_setcsr(saved);return 1;}cases++;
  }
 }
 _mm_setcsr(saved);printf("{\"cases\":%d,\"rounding_modes\":4,\"FTZ_DAZ_combinations\":4,\"guarded_input_output\":true,\"bitwise\":true}\n",cases);munmap(a,page*2);munmap(b,page*2);return 0;
}
