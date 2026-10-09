/* Independent staged oracle for the exact generated 256-wide GRU helper.
 * Correctness only. Compile against the isolated candidate dpdf_full library.
 * This deliberately keeps the generated graph's separate array passes.
 */
#include "full_ops.h"
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64)
#include <xmmintrin.h>
#define HAVE_MXCSR 1
#endif

static uint32_t next(uint32_t *seed) {
    *seed=*seed*1664525u+1013904223u;
    return *seed;
}
static float bits_float(uint32_t bits) {
    float value;
    memcpy(&value,&bits,sizeof(value));
    return value;
}
static void staged(const float *a,const float *b,const float *old,float *out) {
    float reset_sum[256],reset[256],update_sum[256],update[256];
    float product[256],candidate_sum[256],candidate[256],difference[256],updated[256];
    for (int i=0;i<256;++i) reset_sum[i]=b[i]+a[i];
    for (int i=0;i<256;++i) reset[i]=1.0f/(1.0f+expf(-reset_sum[i]));
    for (int i=0;i<256;++i) update_sum[i]=b[256+i]+a[256+i];
    for (int i=0;i<256;++i) update[i]=1.0f/(1.0f+expf(-update_sum[i]));
    for (int i=0;i<256;++i) product[i]=b[512+i]*reset[i];
    for (int i=0;i<256;++i) candidate_sum[i]=a[512+i]+product[i];
    for (int i=0;i<256;++i) candidate[i]=tanhf(candidate_sum[i]);
    for (int i=0;i<256;++i) difference[i]=old[i]-candidate[i];
    for (int i=0;i<256;++i) updated[i]=difference[i]*update[i];
    for (int i=0;i<256;++i) out[i]=updated[i]+candidate[i];
}

static int check(int pattern,int alias,uint32_t *seed) {
    /* Padding permits a valid float pointer with 4-byte displacement from a
     * normally aligned stack array. Raw whole-buffer comparison is a canary. */
    float a0[772],b0[772],old0[260],out0[260];
    float a[772],b[772],old[260],out[260],expected[256];
    for (int i=0;i<772;++i) {
        uint32_t r=next(seed),s=next(seed);
        float av=(float)((int)(r%20001u)-10000)*0.001f;
        float bv=(float)((int)(s%20001u)-10000)*0.001f;
        if (pattern==1) { av=bits_float(r&0x80000000u);bv=bits_float(s&0x80000000u); }
        if (pattern==2) { av=bits_float((r&0x80000000u)|1u);bv=bits_float((s&0x80000000u)|0x007fffffu); }
        if (pattern==3) { av*=0x1p-50f;bv*=0x1p-50f; }
        if (pattern==4) { av*=10000.0f;bv*=10000.0f; }
        if (pattern==5) { av=(i&1)?87.0f:-87.0f;bv=(i&2)?1.0f:-1.0f; }
        a0[i]=av;b0[i]=bv;
    }
    for (int i=0;i<260;++i) {
        uint32_t r=next(seed);
        old0[i]=pattern==1?bits_float(r&0x80000000u):(float)((int)(r%4001u)-2000)*0.001f;
        out0[i]=bits_float(0x4b654321u);
    }
    memcpy(a,a0,sizeof(a));memcpy(b,b0,sizeof(b));
    memcpy(old,old0,sizeof(old));memcpy(out,out0,sizeof(out));
    staged(a0+1,b0+1,old0+1,expected);
    float *actual=out+1;
    float *wanted=out0+1;
    switch (alias) {
        case 0: break;
        case 1: actual=old+1;wanted=old0+1;break;
        case 2: actual=a+1;wanted=a0+1;break;
        case 3: actual=a+257;wanted=a0+257;break;
        case 4: actual=a+513;wanted=a0+513;break;
        case 5: actual=b+1;wanted=b0+1;break;
        case 6: actual=b+257;wanted=b0+257;break;
        case 7: actual=b+513;wanted=b0+513;break;
        default: return 0;
    }
    memcpy(wanted,expected,sizeof(expected));
    dpdf_generated_gru256_fused(a+1,b+1,old+1,actual);
    if (memcmp(a,a0,sizeof(a)) || memcmp(b,b0,sizeof(b)) ||
        memcmp(old,old0,sizeof(old)) || memcmp(out,out0,sizeof(out))) {
        fprintf(stderr,"Generated GRU mismatch: pattern=%d alias=%d\n",pattern,alias);
        return 0;
    }
    return 1;
}

int main(void) {
    const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    const int saved_round=fegetround();
#ifdef HAVE_MXCSR
    const unsigned saved_mxcsr=_mm_getcsr();
    const int denormal_modes=4;
#else
    const int denormal_modes=1;
#endif
    uint32_t seed=0x53a2148du;
    int calls=0;
    for (int mode=0;mode<4;++mode) {
        if (fesetround(modes[mode])) return 2;
        for (int denormal=0;denormal<denormal_modes;++denormal) {
#ifdef HAVE_MXCSR
            unsigned controls=_mm_getcsr();
            controls=(controls&~((1u<<15)|(1u<<6)))|
                     ((denormal&1)?(1u<<15):0u)|((denormal&2)?(1u<<6):0u);
            _mm_setcsr(controls);
#endif
            for (int pattern=0;pattern<6;++pattern) for (int alias=0;alias<8;++alias) {
                if (!check(pattern,alias,&seed)) return 1;
                ++calls;
            }
        }
    }
    if (fesetround(saved_round)) return 2;
#ifdef HAVE_MXCSR
    _mm_setcsr(saved_mxcsr);
#endif
    printf("{\"passed\":true,\"calls_checked\":%d,\"rounding_modes\":4,\"denormal_modes\":%d,\"patterns\":6,\"output_aliases\":8}\n",calls,denormal_modes);
    return 0;
}
