#define _GNU_SOURCE
#include "int8_matrix.h"
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(X) do {if(!(X)){fprintf(stderr,"CM contract failed line %d\n",__LINE__);return 1;}} while(0)
static float from_bits(uint32_t bits) {float v;memcpy(&v,&bits,4);return v;}
static uint32_t seed=841009;
static float sample(void){seed=seed*1664525u+1013904223u;return ((int)(seed>>16)-32768)/8192.0f;}
int main(void) {
 if(!dpdf_has_avx2()){puts("CM AVX2/FMA unavailable; skipped");return 77;}
#ifdef DPDF_TEST_AVX_VNNI
 if(!dpdf_has_avx_vnni()){puts("CM AVX-VNNI unavailable; skipped");return 77;}
#endif
 unsigned saved=_mm_getcsr(),rounds[]={_MM_ROUND_NEAREST,_MM_ROUND_DOWN,_MM_ROUND_UP,_MM_ROUND_TOWARD_ZERO};
 const int widths[]={8,40,48,80,96,160,480};size_t page=(size_t)sysconf(_SC_PAGESIZE),capacity=480*64*sizeof(float),span=(capacity+page-1)/page*page+page;
 char *xm=mmap(0,span+page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0),*ym=mmap(0,span+page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);REQUIRE(xm!=MAP_FAILED && ym!=MAP_FAILED);REQUIRE(!mprotect(xm+span,page,PROT_NONE) && !mprotect(ym+span,page,PROT_NONE));
 float *xr=malloc(capacity),*yr=malloc(capacity),*expected=malloc(capacity);REQUIRE(xr&&yr&&expected);int cases=0;
 for(int mode=0;mode<4;++mode) for(int denorm=0;denorm<4;++denorm) {
  _mm_setcsr((saved&~(0x6000|0x8040))|rounds[mode]|((denorm&1)?0x8000:0)|((denorm&2)?0x40:0));
  float w[64*64],bias[64];for(int i=0;i<64*64;++i)w[i]=sample();for(int i=0;i<64;++i)bias[i]=sample();
  dpdf_qmatrix *q=dpdf_qcreate(w,64,64);REQUIRE(q);
#ifdef DPDF_TEST_AVX_VNNI
  q->affine=dpdf_qaffine_vnni;
#else
  q->affine=dpdf_qaffine_avx2;
#endif
  REQUIRE(dpdf_qaffine_cm64(NULL,xr,bias,expected,8)==0);
  const int invalid_rows[]={-8,0,7,9,488};
  memset(expected,0xa5,32);
  for(size_t i=0;i<sizeof(invalid_rows)/sizeof(invalid_rows[0]);++i)
   REQUIRE(dpdf_qaffine_cm64(q,xr,bias,expected,invalid_rows[i])==0);
  dpdf_qmatrix wrong=*q;wrong.k=128;REQUIRE(dpdf_qaffine_cm64(&wrong,xr,bias,expected,8)==0);
  wrong=*q;wrong.n=72;REQUIRE(dpdf_qaffine_cm64(&wrong,xr,bias,expected,8)==0);
  wrong=*q;wrong.affine=NULL;REQUIRE(dpdf_qaffine_cm64(&wrong,xr,bias,expected,8)==0);
  for(int i=0;i<32;++i)REQUIRE(((unsigned char *)expected)[i]==0xa5);

  size_t packed_span=(4096+page-1)/page*page;void *packed_map=mmap(0,packed_span+page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);REQUIRE(packed_map!=MAP_FAILED);REQUIRE(!mprotect((char *)packed_map+packed_span,page,PROT_NONE));int8_t *packed_end=(int8_t *)packed_map+packed_span-4096;memcpy(packed_end,q->packed,4096);REQUIRE(!mprotect(packed_map,packed_span,PROT_READ));int8_t *old=q->packed;q->packed=packed_end;
  for(size_t wi=0;wi<sizeof(widths)/sizeof(widths[0]);++wi)for(int pattern=0;pattern<12;++pattern)for(int placement=0;placement<2;++placement) {
   int rows=widths[wi];size_t bytes=(size_t)rows*64*sizeof(float);float *x=(float *)(xm+span-bytes-placement*4),*y=(float *)(ym+span-bytes-placement*4);
   REQUIRE(!mprotect(xm,span,PROT_READ|PROT_WRITE));memset(xm,0x5a,span);memset(ym,0xa5,span);
   for(int c=0;c<64;++c)for(int f=0;f<rows;++f) {
    float v=pattern==0 ? ((c&1)?0.0f:-0.0f):pattern==1 ? sample():pattern==2 ? .25f:pattern==3 ? sample()*1e-30f:pattern==4 ? (f%3?sample():0.0f):(c%2?1000.0f:-1000.0f);
    if(pattern==6)v=sample()*1e-40f;
    if(pattern==7)v=c%3 ? sample():from_bits(0x7fc12345u);
    if(pattern==8)v=c%2 ? INFINITY:-INFINITY;
    if(pattern==9)v=c%2 ? 3e38f:-3e38f;
    if(pattern==10)v=c%3 ? sample():from_bits(0x7f812345u);
    if(pattern==11)v=from_bits(0x7fc12345u);
    x[c*rows+f]=v;xr[f*64+c]=v;
   }
   REQUIRE(!mprotect(xm,span,PROT_READ));
   for(int f=0;f<rows;f+=48){int m=rows-f;if(m>48)m=48;dpdf_qaffine(q,xr+f*64,bias,yr+f*64,m);}
   for(int c=0;c<64;++c)for(int f=0;f<rows;++f)expected[c*rows+f]=yr[f*64+c];
   REQUIRE(dpdf_qaffine_cm64(q,x,bias,y,rows)==1);REQUIRE(!memcmp(y,expected,bytes));
   for(unsigned char *z=(unsigned char *)ym;z<(unsigned char *)y;++z)REQUIRE(*z==0xa5);
   if(placement)for(int i=0;i<4;++i)REQUIRE(((unsigned char *)y)[bytes+i]==0xa5);
   REQUIRE(dpdf_qaffine_cm64(q,x,bias,(float *)x+4,rows)==0);
   REQUIRE(dpdf_qaffine_cm64(q,x,bias,(float *)x,rows)==0);
   REQUIRE(dpdf_qaffine_cm64(q,x,bias,(float *)x-4,rows)==0);cases++;
  }
  q->packed=old;munmap(packed_map,packed_span+page);dpdf_qdestroy(q);
 }
 _mm_setcsr(saved);munmap(xm,span+page);munmap(ym,span+page);free(xr);free(yr);free(expected);
 printf("{\"cases\":%d,\"bitwise\":true,\"rounding_modes\":4,\"ftz_daz\":4,\"guarded_readonly_input_weights\":true,\"guarded_output\":true,\"unaligned\":true}\n",cases);return 0;
}
