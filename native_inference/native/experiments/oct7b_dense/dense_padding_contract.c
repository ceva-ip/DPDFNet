/* Independent public-dense-API oracle, loaded from two isolated libraries.
 * All comparisons use float object bits, including signed zero. No timings.
 * INT8 alias cases consume a whole <=48-row input chunk before any output.
 */
#include <dlfcn.h>
#include <fenv.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define DPDF_DENSE_MXCSR 1
#endif

typedef struct {
    void *library;
    void *(*create)(const float *,const float *,int,int,int,int);
    void (*destroy)(void *);
    void (*run)(void *,const float *,float *,int);
    size_t (*bytes)(const void *);
    int (*avx2)(void);
    int (*fp16)(void);
} api;
typedef struct { float *allocation,*data; size_t count; } buffer;
static size_t calls,memory_checks,alias_calls,fp_control_calls;
static long largest_saving;
static uint32_t random_state=UINT32_C(2026100717);
static const uint32_t guard=UINT32_C(0x4b5aa55a);

static void fail(const char *message,int k,int n,int rows,int precision) {
    fprintf(stderr,"%s: k=%d n=%d rows=%d precision=%d\n",message,k,n,rows,precision);
    exit(1);
}
static uint32_t bits(float v) { uint32_t u; memcpy(&u,&v,4); return u; }
static float from_bits(uint32_t u) { float v; memcpy(&v,&u,4); return v; }
static uint32_t random_u32(void) {
    random_state^=random_state<<13; random_state^=random_state>>17;
    random_state^=random_state<<5; return random_state;
}
static float value(int index,int pattern) {
    if (pattern==0) return from_bits(index%2 ? UINT32_C(0x80000000) : 0);
    float v=(float)((int)(random_u32()%4097)-2048)/2048.0f;
    if (index%17==0) v=from_bits(index%2 ? UINT32_C(0x80000000) : 0);
    return pattern==2 ? v*0x1p-20f : pattern==3 ? v*64.0f : v;
}
static buffer allocate(size_t count) {
    buffer b; b.count=count; b.allocation=malloc((count+2)*sizeof(float));
    if (!b.allocation) { fputs("Allocation failed\n",stderr); exit(1); }
    b.data=b.allocation+1; /* Deliberately not 16/32-byte aligned. */
    for (size_t i=0;i<count+2;++i) b.allocation[i]=from_bits(guard);
    return b;
}
static void check_guard(buffer b,int k,int n,int rows,int precision) {
    if (bits(b.allocation[0])!=guard || bits(b.data[b.count])!=guard)
        fail("Buffer guard overwritten",k,n,rows,precision);
}
static api load(const char *path) {
    api a={0}; a.library=dlopen(path,RTLD_NOW|RTLD_LOCAL);
    if (!a.library) { fprintf(stderr,"dlopen: %s\n",dlerror()); exit(1); }
#define LOAD(member,symbol) do { \
    void *address=dlsym(a.library,symbol); \
    if (!address) { fprintf(stderr,"Missing symbol %s\n",symbol); exit(1); } \
    memcpy(&a.member,&address,sizeof(address)); \
} while (0)
    LOAD(create,"dpdf_dense_create"); LOAD(destroy,"dpdf_dense_destroy");
    LOAD(run,"dpdf_dense_run"); LOAD(bytes,"dpdf_dense_bytes");
    LOAD(avx2,"dpdf_has_avx2"); LOAD(fp16,"dpdf_has_fp16");
#undef LOAD
    return a;
}
static void exact(const float *a,const float *b,size_t count,
                  int k,int n,int rows,int precision) {
    for (size_t i=0;i<count;++i) {
        if (bits(a[i])!=bits(b[i])) {
            fprintf(stderr,"Unequal output index=%zu bits=%08x/%08x\n",i,bits(a[i]),bits(b[i]));
            fail("Output differs",k,n,rows,precision);
        }
        if (!isfinite(a[i]) || !isfinite(b[i])) fail("Nonfinite output",k,n,rows,precision);
    }
}
static void one_call(api a,api b,void *left,void *right,const float *input,
                     int k,int n,int rows,int precision,int alias,int fp_control) {
    size_t input_count=(size_t)rows*k,output_count=(size_t)rows*n;
    size_t shared=input_count>output_count?input_count:output_count;
    buffer x[2]={allocate(shared),allocate(shared)};
    buffer y[2]={allocate(output_count),allocate(output_count)};
    memcpy(x[0].data,input,input_count*4); memcpy(x[1].data,input,input_count*4);
    float *out0=alias?x[0].data:y[0].data,*out1=alias?x[1].data:y[1].data;
    a.run(left,x[0].data,out0,rows); b.run(right,x[1].data,out1,rows);
    exact(out0,out1,output_count,k,n,rows,precision);
    if (alias) {
        buffer expected=allocate(output_count);
        a.run(left,input,expected.data,rows);
        exact(expected.data,out0,output_count,k,n,rows,precision);
        check_guard(expected,k,n,rows,precision);
        free(expected.allocation);
    }
    for (int j=0;j<2;++j) {
        check_guard(x[j],k,n,rows,precision); check_guard(y[j],k,n,rows,precision);
        if (!alias && memcmp(x[j].data,input,input_count*4))
            fail("Disjoint input changed",k,n,rows,precision);
    }
    if (alias) alias_calls++;
    if (fp_control) fp_control_calls++;
    calls++;
    for (int j=0;j<2;++j) { free(x[j].allocation); free(y[j].allocation); }
}
static void shape(api a,api b,int k,int n,int precision,int tiled,int controls) {
    float *w=malloc((size_t)k*n*4),*packed=malloc((size_t)k*n*4),*bias=malloc((size_t)n*4);
    if (!w || !packed || !bias) fail("Weight allocation failed",k,n,0,precision);
    for (int j=0;j<k;++j) for (int c=0;c<n;++c)
        w[j*n+c]=c%17==16 ? from_bits((j+c)%2?UINT32_C(0x80000000):0) : value(j*n+c,1);
    for (int c=0;c<n;++c) bias[c]=value(c,c%3==0?0:1);
    if (tiled) for (int c=0;c<n;++c) for (int j=0;j<k;++j)
        packed[(c/64)*k*64+j*64+c%64]=w[j*n+c];
    const float *weights=tiled?packed:w;
    if (fesetround(FE_TONEAREST)) fail("Cannot set creation rounding",k,n,0,precision);
#ifdef DPDF_DENSE_MXCSR
    _mm_setcsr(_mm_getcsr()&~UINT32_C(0x8040));
#endif
    void *left=a.create(weights,bias,k,n,precision,tiled);
    void *right=b.create(weights,bias,k,n,precision,tiled);
    if (!left || !right) fail("Dense creation failed",k,n,0,precision);
    int kp=(k+7)/8*8,old_np=(n+31)/32*32,new_np=precision==8?(n+7)/8*8:old_np;
    long expected=(long)(old_np-new_np)*(kp+12)+
        192L*((old_np!=n?old_np:0)-(new_np!=n?new_np:0));
    long actual=(long)a.bytes(left)-(long)b.bytes(right);
    if (actual!=expected) {
        fprintf(stderr,"Owned-byte saving expected=%ld actual=%ld\n",expected,actual);
        fail("Owned-byte accounting differs",k,n,0,precision);
    }
    if (actual>largest_saving) largest_saving=actual;
    memory_checks++;
    const int regular_rows[]={1,2,4,5,47,48,49,96,97};
    const int control_rows[]={1,4,49};
    const int rounding[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    int row_count=controls?3:9,env_count=controls?8:1;
#ifndef DPDF_DENSE_MXCSR
    if (controls) env_count=4;
#endif
    for (int e=0;e<env_count;++e) {
        if (fesetround(rounding[controls?e%4:0])) fail("Cannot set process rounding",k,n,0,precision);
#ifdef DPDF_DENSE_MXCSR
        unsigned csr=_mm_getcsr()&~UINT32_C(0x8040);
        if (controls && e>=4) csr|=UINT32_C(0x8040);
        _mm_setcsr(csr);
#endif
        for (int r=0;r<row_count;++r) {
            int rows=controls?control_rows[r]:regular_rows[r];
            float *input=malloc((size_t)rows*k*4);
            if (!input) fail("Input allocation failed",k,n,rows,precision);
            for (int pattern=0;pattern<4;++pattern) {
                for (int i=0;i<rows*k;++i) input[i]=value(i,pattern);
                one_call(a,b,left,right,input,k,n,rows,precision,0,controls);
                /* FP32/FP16 APIs do not promise alias safety. For INT8, all
                 * rows in a chunk are quantized before output writes; with
                 * multiple chunks, n<=k prevents clobbering future inputs. */
                if (precision==8 && (rows<=48 || n<=k))
                    one_call(a,b,left,right,input,k,n,rows,precision,1,controls);
            }
            free(input);
        }
    }
    a.destroy(left); b.destroy(right); free(w); free(packed); free(bias);
}
int main(int argc,char **argv) {
    if (argc!=3) { fputs("Usage: dense_padding_contract baseline.so candidate.so\n",stderr); return 2; }
    int original_round=fegetround();
#ifdef DPDF_DENSE_MXCSR
    unsigned original_csr=_mm_getcsr();
#endif
    api a=load(argv[1]),b=load(argv[2]);
    if (a.avx2()!=b.avx2() || a.fp16()!=b.fp16()) {
        fputs("Baseline and candidate dispatch capabilities differ\n",stderr); return 1;
    }
    const int ns[]={1,7,8,16,24,31,32,64,192};
    const int ks[]={1,7,8,17,64,127,512};
    const int precisions[]={0,16,8};
    for (int p=0;p<3;++p) {
        int precision=precisions[p];
        if ((precision==16 && !a.fp16()) || (precision==8 && !a.avx2())) continue;
        for (int n=0;n<9;++n) for (int k=0;k<7;++k)
            shape(a,b,ks[k],ns[n],precision,0,0);
        /* Target every narrow/padded N under all rounding/denormal controls;
         * K=17 requires activation padding, K=64 exercises its exact kernel. */
        for (int n=0;n<9;++n) shape(a,b,n%2?64:17,ns[n],precision,0,1);
        shape(a,b,64,64,precision,64,0); shape(a,b,127,192,precision,64,0);
    }
    if (fesetround(original_round)) { fputs("Cannot restore rounding\n",stderr); return 1; }
#ifdef DPDF_DENSE_MXCSR
    _mm_setcsr(original_csr);
#endif
    dlclose(a.library); dlclose(b.library);
    printf("{\"passed\":true,\"byte_exact_calls\":%zu,\"memory_checks\":%zu,"
           "\"alias_calls\":%zu,\"fp_control_calls\":%zu,\"largest_shape_owned_saving\":%ld,"
           "\"signed_zero_bits\":true,\"unaligned_buffers\":true,\"buffer_canaries\":true}\n",
           calls,memory_checks,alias_calls,fp_control_calls,largest_saving);
    return 0;
}
