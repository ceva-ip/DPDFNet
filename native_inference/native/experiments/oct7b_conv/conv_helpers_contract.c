/* Independent bit/shape/canary oracle for Oct7b convolution helpers.
 * No timing. x/y are disjoint as required by the original convolution and
 * transpose APIs. Every payload starts one float past an allocation boundary.
 */
#include "full_ops.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t next(uint32_t *seed) {
    *seed=*seed*1664525u+1013904223u;
    return *seed;
}
static float bits_float(uint32_t bits) {
    float result;
    memcpy(&result,&bits,sizeof(result));
    return result;
}
static float ordinary(uint32_t *seed) {
    return (float)((int)(next(seed)%20001u)-10000)*0.0001f;
}

#ifdef DPDF_OCT7B_TRANSPOSE_RELU
static int transpose_check(int rows,int cols,int pattern,int avx,uint32_t *seed) {
    size_t count=(size_t)rows*cols;
    float *x=malloc((count+2)*sizeof(float)),*before=malloc((count+2)*sizeof(float));
    float *y=malloc((count+2)*sizeof(float)),*expected=malloc(count*sizeof(float));
    if (!x || !before || !y || !expected) return 0;
    x[0]=x[count+1]=y[0]=y[count+1]=bits_float(0x4b654321u);
    for (size_t i=0;i<count;++i) {
        float value=ordinary(seed);
        if (pattern==1) value=bits_float(next(seed)&0x80000000u);
        if (pattern==2) value=bits_float((next(seed)&0x80000000u)|(uint32_t)(i%0x007fffffu+1u));
        if (pattern==3) value=bits_float((next(seed)&0x80000000u)|0x7f800000u);
        if (pattern==4) value=bits_float((next(seed)&0x80000000u)|0x7fc00001u|(uint32_t)(i%100u));
        x[1+i]=value;
        y[1+i]=bits_float(0x4b123456u);
    }
    memcpy(before,x,(count+2)*sizeof(float));
    /* Deliberately use the original separate transpose and Relu passes. */
    for (int r=0;r<rows;++r) for (int c=0;c<cols;++c)
        expected[c*rows+r]=x[1+r*cols+c];
    for (size_t i=0;i<count;++i) expected[i]=expected[i]>0?expected[i]:0.0f;
#ifdef DPDF_X86_DISPATCH
    if (avx) dpdf_transpose_relu_avx2(x+1,y+1,rows,cols);
    else
#else
    (void)avx;
#endif
    dpdf_transpose_relu_scalar(x+1,y+1,rows,cols);
    int ok=!memcmp(before,x,(count+2)*sizeof(float)) &&
           !memcmp(expected,y+1,count*sizeof(float)) &&
           !memcmp(y,before,sizeof(float)) && !memcmp(y+count+1,before,sizeof(float));
    if (!ok) fprintf(stderr,"Transpose/ReLU mismatch rows=%d cols=%d pattern=%d avx=%d\n",rows,cols,pattern,avx);
    free(x);free(before);free(y);free(expected);
    return ok;
}
#endif

#ifdef DPDF_OCT7B_CONV_PAIR
static int pair_check(int cig,int kh,int cog,int group,int width,int pattern,int avx,uint32_t *seed) {
    int ci=cig*group,co=cog*group,k=cig*kh;
    size_t input_count=(size_t)ci*kh*width,weight_count=(size_t)co*k;
    size_t output_count=(size_t)co*width;
    float *x=malloc((input_count+2)*sizeof(float));
    float *before=malloc((input_count+2)*sizeof(float));
    float *w=malloc((weight_count+2)*sizeof(float));
    float *bias=malloc((co+2)*sizeof(float));
    float *y=malloc((output_count+2)*sizeof(float));
    float *expected=malloc(output_count*sizeof(float));
    if (!x || !before || !w || !bias || !y || !expected) return 0;
    x[0]=x[input_count+1]=w[0]=w[weight_count+1]=bias[0]=bias[co+1]=
        y[0]=y[output_count+1]=bits_float(0x4b654321u);
    for (size_t i=0;i<input_count;++i) {
        x[1+i]=ordinary(seed);
        if (pattern==1) x[1+i]=bits_float(next(seed)&0x80000000u);
        if (pattern==2) x[1+i]*=0x1p-50f;
    }
    for (size_t i=0;i<weight_count;++i) w[1+i]=ordinary(seed);
    for (int i=0;i<co;++i) bias[1+i]=pattern==1?bits_float(next(seed)&0x80000000u):ordinary(seed);
    memcpy(before,x,(input_count+2)*sizeof(float));
    const float *used_bias=pattern==2?NULL:bias+1;
    /* Independent ordered scalar oracle models the baseline's eight-wide
     * FMA groups and separate FP32 tails for this unpadded unit-stride shape. */
    for (int oc=0;oc<co;++oc) for (int col=0;col<width;++col) {
        float sum=used_bias?used_bias[oc]:0.0f;
        for (int ic=0;ic<cig;++ic) for (int ky=0;ky<kh;++ky) {
            float weight=w[1+oc*k+ic*kh+ky];
            float value=x[1+((oc/cog*cig+ic)*kh+ky)*width+col];
            if (avx && col<width/8*8) sum=fmaf(weight,value,sum);
            else { volatile float product=weight*value;sum+=product; }
        }
        expected[oc*width+col]=sum;
    }
    dpdf_axpy_fn axpy=dpdf_axpy_scalar;
#ifdef DPDF_X86_DISPATCH
    if (avx) axpy=dpdf_axpy_avx2;
#else
    (void)avx;
#endif
    dpdf_conv(axpy,x+1,w+1,used_bias,y+1,ci,kh,width,co,1,width,kh,1,1,1,0,0,group);
    int ok=!memcmp(x,before,(input_count+2)*sizeof(float)) &&
           !memcmp(y+1,expected,output_count*sizeof(float)) &&
           !memcmp(y,before,sizeof(float)) && !memcmp(y+output_count+1,before,sizeof(float));
    if (!ok) fprintf(stderr,"Pair convolution mismatch ci/group=%d kh=%d co/group=%d group=%d width=%d pattern=%d avx=%d\n",
                    cig,kh,cog,group,width,pattern,avx);
    free(x);free(before);free(w);free(bias);free(y);free(expected);
    return ok;
}
#endif

int main(void) {
    uint32_t seed=0x7926184bu;
    int implementations=1,transpose_calls=0,pair_calls=0;
#ifdef DPDF_X86_DISPATCH
    if (dpdf_has_avx2()) implementations=2;
#endif
#ifdef DPDF_OCT7B_TRANSPOSE_RELU
    const int dimensions[]={1,7,8,9,15,16,17,31,32,33,40,48,64,80,96,160,480};
    const int dimension_count=(int)(sizeof(dimensions)/sizeof(dimensions[0]));
    for (int avx=0;avx<implementations;++avx)
        for (int r=0;r<dimension_count;++r) for (int c=0;c<dimension_count;++c)
            for (int pattern=0;pattern<5;++pattern) {
                if (!transpose_check(dimensions[r],dimensions[c],pattern,avx,&seed)) return 1;
                ++transpose_calls;
            }
#endif
#ifdef DPDF_OCT7B_CONV_PAIR
    const int widths[]={7,8,9,15,16,17,31,32,33,63,64,65,96,97};
    const int inputs[]={1,3,32},heights[]={1,3,5},outputs[]={1,2,3,5};
    for (int avx=0;avx<implementations;++avx)
        for (int width=0;width<(int)(sizeof(widths)/sizeof(widths[0]));++width)
            for (int input=0;input<3;++input) for (int height=0;height<3;++height)
                for (int output=0;output<4;++output) for (int group=1;group<=2;++group)
                    for (int pattern=0;pattern<3;++pattern) {
                        if (!pair_check(inputs[input],heights[height],outputs[output],group,widths[width],pattern,avx,&seed)) return 1;
                        ++pair_calls;
                    }
#endif
    printf("{\"passed\":true,\"transpose_calls\":%d,\"pair_calls\":%d,\"implementations\":%d}\n",
           transpose_calls,pair_calls,implementations);
    return 0;
}
