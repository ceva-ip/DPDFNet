/* Independent scalar int32 oracle and protected-buffer checks for pair ASM.
 * The model/runtime and C quantized kernel are not linked into this contract. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#if !defined(__linux__) || !defined(__x86_64__)
#error Fixed pair contract requires Linux x86-64.
#endif

void dpdf_qdot4pair64_avx2(const int8_t *,const int8_t *,const int8_t *,int32_t *);
#ifdef DPDF_PAIR_TEST96
void dpdf_qdot4pair96_avx2(const int8_t *,const int8_t *,const int8_t *,int32_t *);
#endif
#define CHECK(v) do { if (!(v)) { fprintf(stderr,"Pair contract line %d\n",__LINE__); return 1; } } while (0)
typedef struct { unsigned char *mapping,*writable,*payload; size_t mapping_bytes,writable_bytes,bytes; } region;
static uint32_t seed=UINT32_C(0xa721bd19);
static uint32_t random_u32(void) { seed=seed*1664525u+1013904223u; return seed; }
static int allocate(region *r,size_t bytes,int at_end,size_t gap) {
    long page=sysconf(_SC_PAGESIZE); if (page<=0) return -1;
    size_t p=(size_t)page,pages=(bytes+gap+p-1)/p;
    r->mapping_bytes=(pages+2)*p;
    r->mapping=mmap(NULL,r->mapping_bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (r->mapping==MAP_FAILED) return -1;
    r->writable=r->mapping+p; r->writable_bytes=pages*p; r->bytes=bytes;
    r->payload=at_end?r->writable+r->writable_bytes-bytes-gap:r->writable+gap;
    if (mprotect(r->writable,r->writable_bytes,PROT_READ|PROT_WRITE)) {
        munmap(r->mapping,r->mapping_bytes); return -1;
    }
    return 0;
}
static int fill(region *r,const void *payload) {
    if (mprotect(r->writable,r->writable_bytes,PROT_READ|PROT_WRITE)) return -1;
    memset(r->writable,0xa5,r->writable_bytes);
    if (payload) memcpy(r->payload,payload,r->bytes); else memset(r->payload,0xcd,r->bytes);
    return 0;
}
static int canaries(const region *r) {
    for (const unsigned char *p=r->writable;p<r->payload;++p) if (*p!=0xa5) return 0;
    for (const unsigned char *p=r->payload+r->bytes;p<r->writable+r->writable_bytes;++p)
        if (*p!=0xa5) return 0;
    return 1;
}
static void patterns(unsigned char *a,int8_t w[2][96*8],int k,int pattern) {
    for (int r=0;r<4;++r) for (int j=0;j<k;++j) {
        a[r*k+j]=pattern==0?0:pattern<=3?254:
            pattern==4?((j+r)%2?254:0):pattern==5?254:
            pattern==6?(unsigned char)((j+r*37)%255):pattern==7?((j+r)%2?253:254):
            pattern==8?(r%2?254:0):(unsigned char)(random_u32()%255u);
    }
    for (int t=0;t<2;++t) for (int j=0;j<k;++j) for (int c=0;c<8;++c)
        w[t][j*8+c]=pattern==0 || pattern==1?63:pattern==2?-63:
            pattern==3?((t+j)%2?-63:63):pattern==4?((t+c)%2?-63:63):
            pattern==5?(t?-63:63):pattern==6?((j+c+t)%2?-63:63):
            pattern==7?((j/2+c+t)%2?-63:63):pattern==8?0:
            (int8_t)((int)(random_u32()%127u)-63);
}
static void oracle(const unsigned char *a,const int8_t w[2][96*8],int k,
                   int8_t packed[2][96*8],int32_t expected[64]) {
    for (int t=0;t<2;++t) for (int c=0;c<8;++c) {
        for (int j=0;j<k;++j) packed[t][(j/4)*32+c*4+j%4]=w[t][j*8+c];
        for (int r=0;r<4;++r) {
            int32_t sum=0;
            for (int j=0;j<k;++j) sum+=(int32_t)a[r*k+j]*(int32_t)w[t][j*8+c];
            expected[(t*4+r)*8+c]=sum;
        }
    }
}
int main(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) { fputs("Pair assembly oracle requires AVX2\n",stderr); return 77; }
    int widths[]={64
#ifdef DPDF_PAIR_TEST96
        ,96
#endif
    };
    int calls=0,patterns_per_placement=265;
    unsigned char activation[4*96]; int8_t weights[2][96*8],packed[2][96*8];
    int32_t expected[64];
    for (size_t width=0;width<sizeof(widths)/sizeof(widths[0]);++width) {
        int k=widths[width];
        for (int placement=0;placement<3;++placement) {
            region a={0},w0={0},w1={0},out={0};
            int end=placement!=1;
            CHECK(allocate(&a,(size_t)k*4,end,placement==2?1:0)==0);
            CHECK(allocate(&w0,(size_t)k*8,end,placement==2?3:0)==0);
            CHECK(allocate(&w1,(size_t)k*8,end,placement==2?5:0)==0);
            CHECK(allocate(&out,256,end,placement==2?4:0)==0);
            if (placement==2) {
                CHECK((uintptr_t)a.payload%32 && (uintptr_t)w0.payload%32 && (uintptr_t)w1.payload%32);
                CHECK((uintptr_t)out.payload%32 && (uintptr_t)out.payload%4==0);
            }
            for (int pattern=0;pattern<patterns_per_placement;++pattern) {
                patterns(activation,weights,k,pattern); oracle(activation,weights,k,packed,expected);
                CHECK(fill(&a,activation)==0 && fill(&w0,packed[0])==0 && fill(&w1,packed[1])==0 && fill(&out,NULL)==0);
                CHECK(mprotect(a.writable,a.writable_bytes,PROT_READ)==0);
                CHECK(mprotect(w0.writable,w0.writable_bytes,PROT_READ)==0);
                CHECK(mprotect(w1.writable,w1.writable_bytes,PROT_READ)==0);
                if (k==64) dpdf_qdot4pair64_avx2((const int8_t *)a.payload,(const int8_t *)w0.payload,
                                               (const int8_t *)w1.payload,(int32_t *)out.payload);
#ifdef DPDF_PAIR_TEST96
                else dpdf_qdot4pair96_avx2((const int8_t *)a.payload,(const int8_t *)w0.payload,
                                          (const int8_t *)w1.payload,(int32_t *)out.payload);
#endif
                CHECK(!memcmp(out.payload,expected,sizeof(expected)));
                CHECK(!memcmp(a.payload,activation,(size_t)k*4));
                CHECK(!memcmp(w0.payload,packed[0],(size_t)k*8) && !memcmp(w1.payload,packed[1],(size_t)k*8));
                CHECK(canaries(&a) && canaries(&w0) && canaries(&w1) && canaries(&out));
                ++calls;
            }
            CHECK(munmap(a.mapping,a.mapping_bytes)==0 && munmap(w0.mapping,w0.mapping_bytes)==0);
            CHECK(munmap(w1.mapping,w1.mapping_bytes)==0 && munmap(out.mapping,out.mapping_bytes)==0);
        }
    }
    printf("{\"passed\":true,\"calls_checked\":%d,\"widths_checked\":%zu,\"patterns_per_placement\":%d,"
           "\"placements\":3,\"output_elements_per_call\":64,\"input_pages_read_only\":true,"
           "\"guard_edges\":\"immediate before and after aligned payloads\","
           "\"unaligned_gap_bytes\":[1,3,5,4],\"canaries_checked\":true}\n",
           calls,sizeof(widths)/sizeof(widths[0]),patterns_per_placement);
    return 0;
}
