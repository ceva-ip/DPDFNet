/* Independent bitwise oracles for layout movement and ordered LayerNorm. */
#include "full_ops.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"Layout contract line %d\n",__LINE__); return 1; } } while (0)

static void norm_reference(float *out,const float *p,const float *skip,
        const float *scale,const float *bias,float eps,int rows) {
    for (int r=0;r<rows;++r) {
        double mean=0,var=0;
        for (int c=0;c<64;++c) mean+=p[r*64+c];
        mean/=64;
        for (int c=0;c<64;++c) { double d=p[r*64+c]-mean; var+=d*d; }
        float inv=(float)(1.0/sqrt(var/64+eps));
        for (int c=0;c<64;++c)
            out[r*64+c]=((p[r*64+c]-(float)mean)*inv)*scale[c]+bias[c]+skip[r*64+c];
    }
}

/* Output-by-output oracle for the original convolution's rounding contract:
 * stride-one AVX2 axpy fuses full groups of eight; all other terms do not. */
static void conv_reference(const float *x,const float *w,const float *bias,float *y,
        int ci,int hi,int wi,int co,int ho,int wo,int kh,int kw,int sw,int pw,int group,int avx) {
    int cig=ci/group,cog=co/group;
    for (int oc=0;oc<co;++oc) for (int oh=0;oh<ho;++oh) for (int ow=0;ow<wo;++ow) {
        float sum=bias?bias[oc]:0;
        for (int ic=0;ic<cig;++ic) for (int ky=0;ky<kh;++ky) for (int kx=0;kx<kw;++kx) {
            int ih=oh+ky,iw=ow*sw+kx-pw;
            if (ih>=hi || iw<0 || iw>=wi) continue;
            float value=x[((oc/cog*cig+ic)*hi+ih)*wi+iw];
            float weight=w[((oc*cig+ic)*kh+ky)*kw+kx];
            int begin=0,end=wo;
            while (begin<end && begin*sw+kx-pw<0) ++begin;
            while (end>begin && (end-1)*sw+kx-pw>=wi) --end;
            if (avx && sw==1 && ow<begin+(end-begin)/8*8) sum=fmaf(weight,value,sum);
            else sum+=weight*value;
        }
        y[(oc*ho+oh)*wo+ow]=sum;
    }
}

static int conv_contract(void) {
    const int widths[]={1,7,8,9,16,17,40,48,65,80,96,160,480};
    int cases=0;
    for (size_t i=0;i<sizeof(widths)/sizeof(widths[0]);++i)
      for (int pattern=0;pattern<3;++pattern) for (int kw=1;kw<=5;kw+=2) for (int sw=1;sw<=3;++sw) {
        int wi=widths[i],pw=kw/2,wo=(wi+2*pw-kw)/sw+1;
        int ci=pattern==1?4:3,co=pattern==1?8:2,group=pattern==1?4:1;
        int kh=pattern==1?1:3,hi=kh+(pattern==2),ho=hi-kh+1;
        int nx=ci*hi*wi,nw=co*(ci/group)*kh*kw,ny=co*ho*wo;
        float *xa=malloc((nx+2)*4),*wa=malloc((nw+1)*4),*ya=malloc((ny+2)*4),*expected=malloc(ny*4);
        CHECK(xa && wa && ya && expected);
        float *x=xa+1,*w=wa+1,*y=ya+1,bias[8];
        for (int j=0;j<nx;++j) x[j]=sinf(j*.137f)*(1+j%7);
        for (int j=0;j<nw;++j) w[j]=cosf(j*.253f)*.17f;
        for (int j=0;j<co;++j) bias[j]=(j-3)*.017f;
        const float *b=i%2?bias:NULL;
        for (int tier=0;tier<2;++tier) {
            dpdf_axpy_fn axpy=dpdf_axpy_scalar;
#ifdef DPDF_X86_DISPATCH
            if (tier==1) { if (!dpdf_has_avx2()) continue; axpy=dpdf_axpy_avx2; }
#else
            if (tier==1) continue;
#endif
            ya[0]=12345;ya[ny+1]=-12345;
            conv_reference(x,w,b,expected,ci,hi,wi,co,ho,wo,kh,kw,sw,pw,group,tier);
            dpdf_conv(axpy,x,w,b,y,ci,hi,wi,co,ho,wo,kh,kw,1,sw,0,pw,group);
            if (memcmp(y,expected,ny*4)) {
                fprintf(stderr,"Convolution differs: width=%d pattern=%d kw=%d sw=%d tier=%d\n",wi,pattern,kw,sw,tier);
                CHECK(0);
            }
            CHECK(ya[0]==12345 && ya[ny+1]==-12345);
        }
        free(xa);free(wa);free(ya);free(expected);++cases;
      }
    printf("%d convolution shapes match the original per-tap rounding contract\n",cases);
    return 0;
}

int main(void) {
    CHECK(conv_contract()==0);
    const int dims[]={1,3,7,8,9,40,48,64,96,160,480,481};
    for (size_t a=0;a<sizeof(dims)/sizeof(dims[0]);++a)
      for (size_t b=0;b<sizeof(dims)/sizeof(dims[0]);++b) {
        int rows=dims[a],cols=dims[b],n=rows*cols;
        float *storage=malloc((n+2)*sizeof(float)),*dest=malloc((n+2)*sizeof(float));
        CHECK(storage && dest);
        float *x=storage+1,*y=dest+1;
        for (int i=0;i<n;++i) {
            uint32_t bits=(uint32_t)i*2654435761u;
            if (i%13==0) bits=0x7fc01234u; /* Preserve NaN payloads as well. */
            memcpy(x+i,&bits,4);
        }
        for (int tier=0;tier<2;++tier) {
            dpdf_transpose_fn transpose=dpdf_transpose_scalar;
#ifdef DPDF_X86_DISPATCH
            if (tier==1) { if (!dpdf_has_avx2()) continue; transpose=dpdf_transpose_avx2; }
#else
            if (tier==1) continue;
#endif
            dest[0]=12345; dest[n+1]=-12345;
            transpose(x,y,rows,cols);
            CHECK(dest[0]==12345 && dest[n+1]==-12345);
            for (int r=0;r<rows;++r) for (int c=0;c<cols;++c)
                CHECK(!memcmp(y+c*rows+r,x+r*cols+c,4));
        }
        free(storage);free(dest);
      }
#ifdef DPDF_X86_DISPATCH
    if (dpdf_has_avx2()) {
        float p[48*64],skip[48*64],a[48*64],b[48*64],scale[64],bias[64];
        for (int pattern=0;pattern<8;++pattern) {
            for (int i=0;i<48*64;++i) {
                p[i]=pattern==0 ? 0 : pattern==1 ? 1 :
                     pattern==2 ? (i%2 ? 1e30f : -1e30f) : sinf((float)(i+pattern)*.134f)*(i%31);
                skip[i]=cosf(i*.002f);
            }
            for (int c=0;c<64;++c) { scale[c]=sinf(c*.12f); bias[c]=(c-32)*.001f; }
            for (int rows=4;rows<=48;rows+=4) {
                norm_reference(a,p,skip,scale,bias,1e-5f,rows);
                dpdf_norm_residual_avx2(b,p,skip,scale,bias,1e-5f,rows);
                CHECK(!memcmp(a,b,(size_t)rows*64*sizeof(float)));
            }
        }
    }
#else
    (void)norm_reference;
#endif
    puts("Transpose edges/NaN payloads and ordered-double SIMD normalization passed");
    return 0;
}
