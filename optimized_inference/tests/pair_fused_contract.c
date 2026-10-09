/* Independent corrected-integer + scalar FP32/fmaf oracle for fused pair.
 * Exact-length mmap guards and read-only source/argument pages check assembler
 * accesses that ASAN cannot instrument. Only Linux x86-64 SysV is supported. */
#define _GNU_SOURCE
#include "pair_fused_args.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xmmintrin.h>

#if !defined(__x86_64__) || !defined(__linux__)
#error This direct contract requires Linux x86-64 SysV.
#endif
#define CHECK(value) do { if (!(value)) { fprintf(stderr,"Pair fused contract line %d\n",__LINE__); return 1; } } while (0)

typedef struct {
    unsigned char *mapping,*writable,*payload;
    size_t mapping_bytes,writable_bytes,payload_bytes;
} region;

static uint32_t random_state=0x73958341u;
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

static int fill_region(region *r,const void *source) {
    if (mprotect(r->writable,r->writable_bytes,PROT_READ|PROT_WRITE)) return -1;
    memset(r->writable,0xa5,r->writable_bytes);
    if (source) memcpy(r->payload,source,r->payload_bytes);
    return 0;
}

static int padding_unchanged(const region *r) {
    for (const unsigned char *p=r->writable;p<r->payload;++p)
        if (*p!=0xa5) return 0;
    for (const unsigned char *p=r->payload+r->payload_bytes;p<r->writable+r->writable_bytes;++p)
        if (*p!=0xa5) return 0;
    return 1;
}

static void pattern_data(unsigned char *a,int8_t w[2][64*8],float scales[2][8],
        float bias[2][8],float activation_scales[4],int zp[4],int pattern) {
    const float choices[]={1.0f,1e-20f,1e10f,+0.0f,-0.0f};
    for (int r=0;r<4;++r) {
        activation_scales[r]=choices[(pattern+r)%5];
        zp[r]=(pattern+r)%4==0?0:(pattern+r)%4==1?127:(pattern+r)%4==2?254:
              (int)(random_u32()%255u);
        for (int j=0;j<64;++j)
            a[r*64+j]=pattern==0?0:pattern<=3?254:
                pattern==4?((j+r)%2?254:0):pattern==5?(r%2?0:254):
                pattern==6?(unsigned char)j:pattern==7?((j+r)%2?253:254):
                (unsigned char)(random_u32()%255u);
    }
    for (int tile=0;tile<2;++tile) {
        for (int j=0;j<64;++j) for (int c=0;c<8;++c)
            w[tile][j*8+c]=pattern<=1?63:pattern==2?-63:
                pattern==3 || pattern==4?((j+tile)%2?-63:63):
                pattern==5?((c+tile)%2?-63:63):pattern==6?((j+c+tile)%2?-63:63):
                pattern==7?((j/2+c+tile)%2?-63:63):pattern==8?0:
                (int8_t)((int)(random_u32()%127u)-63);
        for (int c=0;c<8;++c) {
            scales[tile][c]=(c+tile)%9==0?+0.0f:(c+tile)%9==1?-0.0f:
                (c+tile)%9==2?1e-20f:(c+tile)%9==3?FLT_MIN:
                ((float)(1+random_u32()%1024u))/8192.0f;
            bias[tile][c]=(c+tile)%5==0?-0.0f:(c+tile)%5==1?+0.0f:
                ((int)(random_u32()%65536u)-32768)/8192.0f;
        }
    }
}

static void independent_reference(const unsigned char *a,const int8_t w[2][64*8],
        const float scales[2][8],const float bias[2][8],const float activation_scales[4],
        const int zp[4],int8_t packed[2][64*8],int32_t sums[2][8],float expected[2][4*8]) {
    for (int tile=0;tile<2;++tile) for (int c=0;c<8;++c) {
        int32_t weight_sum=0;
        for (int j=0;j<64;++j) {
            int8_t value=w[tile][j*8+c];
            packed[tile][(j/4)*32+c*4+j%4]=value;
            weight_sum+=value;
        }
        sums[tile][c]=weight_sum;
        for (int r=0;r<4;++r) {
            int32_t sum=0;
            for (int j=0;j<64;++j)
                sum+=((int32_t)a[r*64+j]-zp[r])*(int32_t)w[tile][j*8+c];
            volatile float product=activation_scales[r]*scales[tile][c];
            expected[tile][r*8+c]=fmaf((float)sum,product,bias[tile][c]);
        }
    }
}

int main(void) {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) {
        fputs("Fused pair contract requires AVX2/FMA; not tested.\n",stderr);
        return 77;
    }
    unsigned char activation[4*64];
    int8_t weights[2][64*8],packed[2][64*8];
    int32_t sums[2][8];
    float scales[2][8],bias[2][8],activation_scales[4],expected[2][4*8];
    int zero_point[4];
    const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    const int strides[]={8,16,192};
    fenv_t saved;
    CHECK(fegetenv(&saved)==0);
    unsigned original_mxcsr=_mm_getcsr();
    int calls=0,outputs=0;
    for (int rounding=0;rounding<4;++rounding) for (int denorm=0;denorm<2;++denorm) {
        CHECK(fesetround(modes[rounding])==0);
        _mm_setcsr((_mm_getcsr()&~0x8040u)|(denorm?0x8040u:0u));
        unsigned controls=_mm_getcsr()&0xffc0u;
        for (int stride_index=0;stride_index<3;++stride_index) {
            int stride=strides[stride_index];
            for (int shared=0;shared<2;++shared) {
                if (shared && stride<16) continue;
                size_t output_bytes=(size_t)(3*stride+(shared?16:8))*4;
                for (int placement=0;placement<3;++placement) {
                    /* Buffers0..8 are activation/weights/sums/scales/bias.
                     * Buffer9 is the immutable pointer/FP metadata contract.
                     * Buffers10/11 are outputs; shared layout uses10 only. */
                    region buffers[12]={{0}};
                    const size_t lengths[]={256,512,512,32,32,32,32,32,32,128,
                                            output_bytes,output_bytes};
                    const size_t gaps[]={1,3,3,4,4,4,4,4,4,8,4,4};
                    for (int b=0;b<12;++b) {
                        CHECK(allocate_region(buffers+b,lengths[b],placement!=1,
                                              placement==2?gaps[b]:0)==0);
                        if (placement==2) CHECK((uintptr_t)buffers[b].payload%32!=0);
                        if (b>=3 && b!=9) CHECK((uintptr_t)buffers[b].payload%4==0);
                        if (b==9) CHECK((uintptr_t)buffers[b].payload%8==0);
                    }
                    unsigned char expected_bytes[2][(3*192+16)*4];
                    for (int pattern=0;pattern<32;++pattern) {
                        pattern_data(activation,weights,scales,bias,activation_scales,zero_point,pattern);
                        independent_reference(activation,weights,scales,bias,activation_scales,
                                              zero_point,packed,sums,expected);
                        const void *sources[]={activation,packed[0],packed[1],sums[0],sums[1],
                                               scales[0],scales[1],bias[0],bias[1]};
                        for (int b=0;b<9;++b) {
                            CHECK(fill_region(buffers+b,sources[b])==0);
                            CHECK(mprotect(buffers[b].writable,buffers[b].writable_bytes,PROT_READ)==0);
                        }
                        CHECK(fill_region(buffers+10,NULL)==0 && fill_region(buffers+11,NULL)==0);
                        memset(expected_bytes,0xa5,sizeof(expected_bytes));
                        for (int r=0;r<4;++r) for (int tile=0;tile<2;++tile) {
                            int output_index=shared?0:tile;
                            size_t offset=(size_t)(r*stride+(shared?tile*8:0))*4;
                            memcpy(expected_bytes[output_index]+offset,expected[tile]+r*8,32);
                        }
                        dpdf_pair_fused_args args={
                            (const int8_t *)buffers[0].payload,
                            (const int8_t *)buffers[1].payload,(const int8_t *)buffers[2].payload,
                            (const int32_t *)buffers[3].payload,(const int32_t *)buffers[4].payload,
                            (const float *)buffers[5].payload,(const float *)buffers[6].payload,
                            (const float *)buffers[7].payload,(const float *)buffers[8].payload,
                            (float *)buffers[10].payload,
                            (float *)(shared?buffers[10].payload+32:buffers[11].payload),
                            (size_t)stride*4,{0},{0}
                        };
                        memcpy(args.activation_scale,activation_scales,sizeof(activation_scales));
                        memcpy(args.zero_point,zero_point,sizeof(zero_point));
                        CHECK(fill_region(buffers+9,&args)==0);
                        CHECK(mprotect(buffers[9].writable,buffers[9].writable_bytes,PROT_READ)==0);
                        dpdf_qaffine4pair64_fused_avx2((const dpdf_pair_fused_args *)buffers[9].payload);
                        for (int output=0;output<2;++output) {
                            if (memcmp(buffers[10+output].payload,expected_bytes[output],output_bytes)) {
                                fprintf(stderr,"round=%d ftz_daz=%d stride=%d shared=%d placement=%d pattern=%d output=%d\n",
                                        rounding,denorm,stride,shared,placement,pattern,output);
                                return 1;
                            }
                        }
                        for (int b=0;b<12;++b) CHECK(padding_unchanged(buffers+b));
                        for (int b=0;b<9;++b) CHECK(!memcmp(buffers[b].payload,sources[b],lengths[b]));
                        CHECK(!memcmp(buffers[9].payload,&args,sizeof(args)));
                        CHECK(fegetround()==modes[rounding] && (_mm_getcsr()&0xffc0u)==controls);
                        ++calls; outputs+=64;
                    }
                    for (int b=0;b<12;++b) CHECK(munmap(buffers[b].mapping,buffers[b].mapping_bytes)==0);
                }
            }
        }
    }
    CHECK(fesetenv(&saved)==0); _mm_setcsr(original_mxcsr);
    printf("{\"passed\":true,\"k\":64,\"rows\":4,\"tiles\":2,\"calls_checked\":%d,\"float_outputs_checked\":%d,"
           "\"rounding_modes\":4,\"ftz_daz_modes\":2,\"output_strides\":[8,16,192],"
           "\"independent_and_adjacent_outputs\":true,\"guarded_placements\":3,\"patterns\":32,"
           "\"input_and_argument_pages_read_only\":true,\"padding_canaries\":true,"
           "\"caller_controls_preserved\":true,\"separate_fp32_scale_product\":true}\n",calls,outputs);
    return 0;
}
