/* Bounds/aliasing/API tests intended to run under ASan + UBSan too. */
#include "dpdf_dprnn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #x); return 1; } } while (0)

int main(int argc, char **argv) {
    CHECK(argc == 2);
    FILE *f = fopen(argv[1], "rb"); CHECK(f);
    float *w = malloc(DPDF_WEIGHT_FLOATS * sizeof(float)); CHECK(w);
    CHECK(fread(w, sizeof(float), DPDF_WEIGHT_FLOATS, f) == DPDF_WEIGHT_FLOATS);
    CHECK(fgetc(f) == EOF); fclose(f);
    CHECK(!dpdf_create(41, w, DPDF_WEIGHT_FLOATS, 1e-5f, 1e-5f, 0));
    CHECK(!dpdf_create(40, w, DPDF_WEIGHT_FLOATS-1, 1e-5f, 1e-5f, 0));
    CHECK(!dpdf_create(40, w, DPDF_WEIGHT_FLOATS, 0, 1e-5f, 0));
    CHECK(!dpdf_create(40, w, DPDF_WEIGHT_FLOATS, 1e-5f, 1e-5f, 3));
    CHECK(!dpdf_create(40, NULL, DPDF_WEIGHT_FLOATS, 1e-5f, 1e-5f, 0));
    float saved = w[0]; w[0] = NAN;
    CHECK(!dpdf_create(40, w, DPDF_WEIGHT_FLOATS, 1e-5f, 1e-5f, 0)); w[0] = saved;
    CHECK(dpdf_process(NULL, NULL, NULL, NULL, NULL) == -1);
    dpdf_destroy(NULL);
    for (int freq = 40; freq <= 48; freq += 8) {
        const size_t count = (size_t)freq*64, bytes = count*sizeof(float);
        float *x=malloc(bytes), *y=malloc(bytes), *alias=malloc(bytes);
        float *s=calloc(count,sizeof(float)), *next=malloc(bytes), *inplace=malloc(bytes);
        float *freq_in=malloc(bytes), *freq_out=malloc(bytes), *freq_next=malloc(bytes);
        CHECK(x && y && alias && s && next && inplace && freq_in && freq_out && freq_next);
        for (size_t i=0; i<count; ++i) x[i]=sinf((float)i*0.01f);
        for (int r=0;r<freq;++r) for (int c=0;c<64;++c) freq_in[r*64+c]=x[c*freq+r];
        for (int tier=1; tier<=2; ++tier) {
            if (tier==2 && !dpdf_has_avx2()) continue;
            dpdf_block *b=dpdf_create(freq,w,DPDF_WEIGHT_FLOATS,1e-5f,1e-5f,tier); CHECK(b);
            memset(s,0,bytes);
            for (int frame=0; frame<100; ++frame) {
                memcpy(alias,x,bytes); memcpy(inplace,s,bytes);
                CHECK(dpdf_process(b,x,s,y,next)==0);
                CHECK(dpdf_process(b,alias,inplace,alias,inplace)==0);
                CHECK(memcmp(y,alias,bytes)==0 && memcmp(next,inplace,bytes)==0);
                CHECK(dpdf_process_layout(b,freq_in,s,freq_out,freq_next,
                                          DPDF_INPUT_FREQ_MAJOR|DPDF_OUTPUT_FREQ_MAJOR)==0);
                CHECK(memcmp(next,freq_next,bytes)==0);
                for (int r=0;r<freq;++r) for (int c=0;c<64;++c)
                    CHECK(y[c*freq+r]==freq_out[r*64+c]);
                for (size_t i=0; i<count; ++i) CHECK(isfinite(y[i]) && isfinite(next[i]));
                memcpy(s,next,bytes);
            }
            CHECK(dpdf_process(b,NULL,s,y,next)==-1);
            CHECK(dpdf_process_layout(b,x,s,y,next,4)==-1);
            dpdf_destroy(b);
        }
        free(x); free(y); free(alias); free(s); free(next); free(inplace);
        free(freq_in); free(freq_out); free(freq_next);
    }
    free(w);
    puts("Native API, bounds, finite output, and in-place alias tests passed.");
    return 0;
}
