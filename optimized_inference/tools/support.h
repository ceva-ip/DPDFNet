#ifndef DPDFNET_TOOL_SUPPORT_H
#define DPDFNET_TOOL_SUPPORT_H
#include "dpdfnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
static inline float *read_weights(const char *path, const dpdfnet_model_info *info) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    float *w = malloc(info->weight_floats * sizeof(float));
    if (!w) { fclose(f); return NULL; }
    int ok = fread(w, sizeof(float), info->weight_floats, f) == info->weight_floats;
    ok = ok && fgetc(f) == EOF && !ferror(f);
    fclose(f);
    if (!ok) { fprintf(stderr, "Invalid weight file length: %s\n", path); free(w); return NULL; }
    return w;
}
/* Deterministic synthetic spectra for training and smoke benchmarks only. */
static inline void synthetic_spectrum(float *x, size_t frame, float scale) {
    uint32_t random = (uint32_t)frame + UINT32_C(20261009);
    for (size_t i = 0; i < 962; ++i) {
        random = random * UINT32_C(1664525) + UINT32_C(1013904223);
        float noise = (float)((int)(random >> 16) - 32768) / 32768.0f;
        x[i] = scale * (0.15f * noise + 0.3f * sinf((float)(i * 7 + frame) * 0.013f));
    }
    x[1] = x[961] = 0.0f;
}
#endif
