/* Direct, independent integer oracle and guarded-buffer check for the frozen
 * Linux x86-64 SysV K=64 W7A8 assembler. No model runtime is linked. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#if !defined(__x86_64__) || !defined(__linux__)
#error This guarded assembly contract requires Linux x86-64.
#endif

void dpdf_qdot64_fixed_avx2(const int8_t *,const int8_t *,int32_t *);
#define CHECK(value) do { if (!(value)) { fprintf(stderr,"Assembly contract line %d\n",__LINE__); return 1; } } while (0)

typedef struct {
    unsigned char *mapping,*writable,*payload;
    size_t mapping_bytes,writable_bytes,payload_bytes;
} region;

static uint32_t random_state=0x489163c7u;
static uint32_t random_u32(void) {
    random_state=random_state*1664525u+1013904223u;
    return random_state;
}

static int allocate_region(region *r,size_t bytes,int at_end,size_t gap) {
    long page=sysconf(_SC_PAGESIZE);
    if (page<=0) return -1;
    size_t page_bytes=(size_t)page;
    size_t pages=(bytes+gap+page_bytes-1)/page_bytes;
    r->mapping_bytes=(pages+2)*page_bytes;
    r->mapping=mmap(NULL,r->mapping_bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    if (r->mapping==MAP_FAILED) return -1;
    r->writable=r->mapping+page_bytes;
    r->writable_bytes=pages*page_bytes;
    r->payload=at_end?r->writable+r->writable_bytes-bytes-gap:r->writable+gap;
    r->payload_bytes=bytes;
    if (mprotect(r->writable,r->writable_bytes,PROT_READ|PROT_WRITE)) {
        munmap(r->mapping,r->mapping_bytes); return -1;
    }
    return 0;
}

static int fill_region(region *r,const void *payload) {
    if (mprotect(r->writable,r->writable_bytes,PROT_READ|PROT_WRITE)) return -1;
    memset(r->writable,0xa5,r->writable_bytes);
    if (payload) memcpy(r->payload,payload,r->payload_bytes);
    else memset(r->payload,0xcd,r->payload_bytes);
    return 0;
}

static int padding_unchanged(const region *r) {
    for (const unsigned char *p=r->writable;p<r->payload;++p)
        if (*p!=0xa5) return 0;
    for (const unsigned char *p=r->payload+r->payload_bytes;p<r->writable+r->writable_bytes;++p)
        if (*p!=0xa5) return 0;
    return 1;
}

static void make_pattern(unsigned char *activation,int8_t *weights,int pattern) {
    for (int j=0;j<64;++j) {
        activation[j]=pattern==0?0:pattern<=3?254:
                      pattern==4?(j%2?254:0):pattern==5?254:
                      pattern==6?(unsigned char)j:pattern==7?(j%2?253:254):pattern==8?254:
                      (unsigned char)(random_u32()%255u);
        for (int c=0;c<64;++c)
            weights[j*64+c]=pattern==0 || pattern==1?63:pattern==2?-63:
                            pattern==3 || pattern==4?(j%2?-63:63):
                            pattern==5?(c%2?-63:63):
                            pattern==6?((j+c)%2?-63:63):
                            pattern==7?((j/2+c)%2?-63:63):pattern==8?0:
                            (int8_t)((int)(random_u32()%127u)-63);
    }
}

static void pack_and_oracle(const unsigned char *activation,const int8_t *weights,
                            int8_t *packed,int32_t *expected) {
    for (int c=0;c<64;++c) {
        int32_t sum=0;
        for (int j=0;j<64;++j) {
            int8_t value=weights[j*64+c];
            packed[(c/8)*512+(j/4)*32+(c%8)*4+j%4]=value;
            sum+=(int32_t)activation[j]*(int32_t)value;
        }
        expected[c]=sum;
    }
}

int main(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) {
        fputs("Direct assembler contract requires AVX2; not tested.\n",stderr);
        return 77;
    }
    const int patterns=265; /* Nine extremes/signed/zero +256 random cases. */
    unsigned char activation[64];
    int8_t weights[64*64],packed[64*64];
    int32_t expected[64];
    int calls=0;
    for (int placement=0;placement<3;++placement) {
        region a={0},w={0},out={0};
        int at_end=placement!=1;
        size_t a_gap=placement==2?1:0,w_gap=placement==2?3:0,out_gap=placement==2?4:0;
        CHECK(allocate_region(&a,64,at_end,a_gap)==0);
        CHECK(allocate_region(&w,4096,at_end,w_gap)==0);
        CHECK(allocate_region(&out,256,at_end,out_gap)==0);
        if (placement==2) {
            CHECK((uintptr_t)a.payload%32!=0 && (uintptr_t)w.payload%32!=0);
            CHECK((uintptr_t)out.payload%32!=0 && (uintptr_t)out.payload%4==0);
        }
        for (int pattern=0;pattern<patterns;++pattern) {
            make_pattern(activation,weights,pattern);
            pack_and_oracle(activation,weights,packed,expected);
            CHECK(fill_region(&a,activation)==0);
            CHECK(fill_region(&w,packed)==0);
            CHECK(fill_region(&out,NULL)==0);
            /* Input writes fault independently of guard-page edge checks. */
            CHECK(mprotect(a.writable,a.writable_bytes,PROT_READ)==0);
            CHECK(mprotect(w.writable,w.writable_bytes,PROT_READ)==0);
            dpdf_qdot64_fixed_avx2((const int8_t *)a.payload,
                                   (const int8_t *)w.payload,(int32_t *)out.payload);
            CHECK(!memcmp(out.payload,expected,sizeof(expected)));
            CHECK(!memcmp(a.payload,activation,sizeof(activation)));
            CHECK(!memcmp(w.payload,packed,sizeof(packed)));
            CHECK(padding_unchanged(&a) && padding_unchanged(&w) && padding_unchanged(&out));
            ++calls;
        }
        CHECK(munmap(a.mapping,a.mapping_bytes)==0);
        CHECK(munmap(w.mapping,w.mapping_bytes)==0);
        CHECK(munmap(out.mapping,out.mapping_bytes)==0);
    }
    printf("{\"passed\":true,\"calls_checked\":%d,\"patterns_per_placement\":%d,"
           "\"placements\":3,\"activation_bytes\":64,\"packed_weight_bytes\":4096,"
           "\"output_bytes\":256,\"output_elements_per_call\":64,"
           "\"seed_hex\":\"489163c7\",\"input_pages_read_only\":true,"
           "\"guard_edges\":\"immediate after and before aligned payloads\","
           "\"unaligned_gap_bytes\":[1,3,4],\"canaries_checked\":true}\n",calls,patterns);
    return 0;
}
