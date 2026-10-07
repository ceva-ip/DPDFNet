/* Independent scalar integer and FP32 oracle for fused K64 row assembly.
 * mmap guards, read-only source pages, all caller rounding/FTZ/DAZ controls.
 * This direct contract links the generated .S; no model runtime is involved. */
#define _GNU_SOURCE
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xmmintrin.h>

#if !defined(__x86_64__) || !defined(__linux__)
#error This direct fused contract requires Linux x86-64 SysV.
#endif

void dpdf_qaffine64_fused_avx2(const int8_t *,const int8_t *,const int32_t *,
        const float *,const float *,float *,float,int,int);
#define CHECK(value) do { if (!(value)) { fprintf(stderr,"Fused contract line %d\n",__LINE__); return 1; } } while (0)

typedef struct {
    unsigned char *mapping,*writable,*payload;
    size_t mapping_bytes,writable_bytes,payload_bytes;
} region;

static uint32_t random_state=0x85163942u;
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

static void pattern_data(unsigned char *a,int8_t *w,float *scales,float *bias,
                         int n,int pattern) {
    for (int j=0;j<64;++j) {
        a[j]=pattern==0?0:pattern<=3?254:pattern==4?(j%2?254:0):
             pattern==5?254:pattern==6?(unsigned char)j:pattern==7?(j%2?253:254):
             pattern==8?254:(unsigned char)(random_u32()%255u);
        for (int c=0;c<n;++c)
            w[j*n+c]=pattern<=1?63:pattern==2?-63:
                     pattern==3 || pattern==4?(j%2?-63:63):
                     pattern==5?(c%2?-63:63):pattern==6?((j+c)%2?-63:63):
                     pattern==7?((j/2+c)%2?-63:63):pattern==8?0:
                     (int8_t)((int)(random_u32()%127u)-63);
    }
    for (int c=0;c<n;++c) {
        scales[c]=c%9==0?+0.0f:c%9==1?-0.0f:c%9==2?1e-20f:
                  c%9==3?FLT_MIN:((float)(1+random_u32()%1024u))/8192.0f;
        bias[c]=c%5==0?-0.0f:c%5==1?+0.0f:
                ((int)(random_u32()%65536u)-32768)/8192.0f;
    }
}

static void independent_reference(const unsigned char *a,const int8_t *w,
        const float *scales,const float *bias,float activation_scale,int zp,int n,
        int8_t *packed,int32_t *weight_sums,float *expected) {
    for (int c=0;c<n;++c) {
        int32_t sum=0,weight_sum=0;
        for (int j=0;j<64;++j) {
            int8_t value=w[j*n+c];
            packed[(c/8)*512+(j/4)*32+(c%8)*4+j%4]=value;
            sum+=((int32_t)a[j]-zp)*(int32_t)value;
            weight_sum+=value;
        }
        weight_sums[c]=weight_sum;
        /* Deliberately round the scalar FP32 scale product before bias FMA. */
        volatile float combined_scale=activation_scale*scales[c];
        expected[c]=fmaf((float)sum,combined_scale,bias[c]);
    }
}

int main(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        fputs("Fused assembler contract requires AVX2/FMA; not tested.\n",stderr);
        return 77;
    }
    unsigned char a[64];
    int8_t w[64*192],packed[64*192];
    int32_t sums[192];
    float scales[192],bias[192],expected[192];
    const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    const float activation_scales[]={1.0f,1e-20f,1e10f,+0.0f,-0.0f};
    fenv_t saved;
    CHECK(fegetenv(&saved)==0);
    unsigned original_mxcsr=_mm_getcsr();
    int calls=0,outputs=0;
    for (int rounding=0;rounding<4;++rounding) for (int denorm=0;denorm<2;++denorm) {
        CHECK(fesetround(modes[rounding])==0);
        _mm_setcsr((_mm_getcsr()&~0x8040u)|(denorm?0x8040u:0u));
        unsigned controls=_mm_getcsr()&0xffc0u;
        for (int tiles=1;tiles<=3;tiles+=2) {
            int n=tiles*64;
            for (int placement=0;placement<3;++placement) {
                region buffers[6]={{0}};
                const size_t lengths[]={64,(size_t)64*n,(size_t)n*4,(size_t)n*4,(size_t)n*4,(size_t)n*4};
                const size_t gaps[]={1,3,4,4,4,4};
                for (int b=0;b<6;++b) {
                    CHECK(allocate_region(buffers+b,lengths[b],placement!=1,
                                          placement==2?gaps[b]:0)==0);
                    if (placement==2) CHECK((uintptr_t)buffers[b].payload%32!=0);
                    if (b>=2) CHECK((uintptr_t)buffers[b].payload%4==0);
                }
                for (int pattern=0;pattern<32;++pattern) {
                    int zp=pattern%4==0?0:pattern%4==1?127:pattern%4==2?254:
                           (int)(random_u32()%255u);
                    float activation_scale=activation_scales[pattern%5];
                    pattern_data(a,w,scales,bias,n,pattern);
                    independent_reference(a,w,scales,bias,activation_scale,zp,n,
                                          packed,sums,expected);
                    const void *payloads[]={a,packed,sums,scales,bias,NULL};
                    for (int b=0;b<6;++b) {
                        CHECK(fill_region(buffers+b,payloads[b])==0);
                        if (b<5) CHECK(mprotect(buffers[b].writable,buffers[b].writable_bytes,PROT_READ)==0);
                    }
                    dpdf_qaffine64_fused_avx2((const int8_t *)buffers[0].payload,
                            (const int8_t *)buffers[1].payload,(const int32_t *)buffers[2].payload,
                            (const float *)buffers[3].payload,(const float *)buffers[4].payload,
                            (float *)buffers[5].payload,activation_scale,zp,tiles);
                    if (memcmp(buffers[5].payload,expected,(size_t)n*4)) {
                        const unsigned char *actual=buffers[5].payload;
                        for (int c=0;c<n;++c) if (memcmp(actual+c*4,expected+c,4)) {
                            uint32_t actual_bits,expected_bits;
                            memcpy(&actual_bits,actual+c*4,4); memcpy(&expected_bits,expected+c,4);
                            fprintf(stderr,"round=%d ftz_daz=%d tiles=%d placement=%d pattern=%d col=%d bits=%08x/%08x\n",
                                    rounding,denorm,tiles,placement,pattern,c,actual_bits,expected_bits);
                            return 1;
                        }
                    }
                    for (int b=0;b<6;++b) CHECK(padding_unchanged(buffers+b));
                    for (int b=0;b<5;++b) CHECK(!memcmp(buffers[b].payload,payloads[b],lengths[b]));
                    CHECK(fegetround()==modes[rounding] && (_mm_getcsr()&0xffc0u)==controls);
                    ++calls; outputs+=n;
                }
                for (int b=0;b<6;++b) CHECK(munmap(buffers[b].mapping,buffers[b].mapping_bytes)==0);
            }
        }
    }
    CHECK(fesetenv(&saved)==0); _mm_setcsr(original_mxcsr);
    printf("{\"passed\":true,\"calls_checked\":%d,\"float_outputs_checked\":%d,"
           "\"rounding_modes\":4,\"ftz_daz_modes\":2,\"tiles\":[1,3],"
           "\"guarded_placements\":3,\"patterns_per_placement\":32,"
           "\"input_pages_read_only\":true,\"padding_canaries\":true,"
           "\"caller_controls_preserved\":true,\"separate_fp32_scale_product\":true}\n",calls,outputs);
    return 0;
}
