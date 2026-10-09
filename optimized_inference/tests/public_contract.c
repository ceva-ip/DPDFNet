#include "../tools/support.h"
#include <string.h>
#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "Failed line %d: %s\n", __LINE__, #expr); exit(1); } } while (0)
int main(int argc, char **argv) {
    CHECK(argc == 3);
    CHECK(!dpdfnet_get_model_info((dpdfnet_model_id)3));
    CHECK(!dpdfnet_create((dpdfnet_model_id)3, NULL, 0));
    CHECK(dpdfnet_init_state((dpdfnet_model_id)3, NULL) == -1);
    CHECK(dpdfnet_process(NULL, NULL, NULL, NULL, NULL) == -1);
    CHECK(dpdfnet_owned_bytes(NULL) == 0);
    dpdfnet_destroy(NULL);
    dpdfnet_model *contexts[2] = {NULL, NULL};
    float *states[2] = {NULL, NULL};
    for (int index = 0; index < 2; ++index) {
        dpdfnet_model_id id = index ? DPDFNET_MODEL_8 : DPDFNET_MODEL_2;
        const dpdfnet_model_info *info = dpdfnet_get_model_info(id);
        CHECK(info && info->abi_version == DPDFNET_ABI_VERSION);
        CHECK(info->sample_rate == 48000 && info->hop_samples == 480 && info->spectrum_floats == 962);
        CHECK(info->state_floats == (index ? 90228U : 56436U));
        CHECK(dpdfnet_init_state(id, NULL) == -1);
        float *w = read_weights(argv[index + 1], info);
        CHECK(w);
        CHECK(!dpdfnet_create(id, w, info->weight_floats - 1));
        CHECK(!dpdfnet_create(id, NULL, info->weight_floats));
        float saved = w[17]; w[17] = NAN;
        CHECK(!dpdfnet_create(id, w, info->weight_floats)); w[17] = saved;
        if (!dpdfnet_cpu_supported()) {
            CHECK(!dpdfnet_create(id, w, info->weight_floats)); free(w); continue;
        }
        dpdfnet_model *a = dpdfnet_create(id, w, info->weight_floats);
        dpdfnet_model *b = dpdfnet_create(id, w, info->weight_floats);
        CHECK(a && b);
        CHECK(dpdfnet_owned_bytes(a) == (index ? 5323412U : 3984320U));
        memset(w, 0, info->weight_floats * sizeof(float)); free(w); /* Contexts own their weights. */
        float *s = malloc(info->state_floats * sizeof(float));
        float *t = malloc(info->state_floats * sizeof(float));
        float *u = malloc(info->state_floats * sizeof(float));
        CHECK(s && t && u);
        CHECK(!dpdfnet_init_state(id, s)); CHECK(s[0] != 0.0f);
        memcpy(t, s, info->state_floats * sizeof(float));
        CHECK(dpdfnet_process(a, NULL, s, NULL, t) == -1);
        for (size_t hop = 0; hop < 12; ++hop) {
            float x[962], y[962], z[962]; synthetic_spectrum(x, hop, hop < 4 ? 0.001f : 1.0f);
            memcpy(z, x, sizeof(x));
            CHECK(!dpdfnet_process(a, x, s, y, u));
            CHECK(!dpdfnet_process(b, z, t, z, t));
            CHECK(!memcmp(y, z, sizeof(y)));
            CHECK(!memcmp(u, t, info->state_floats * sizeof(float)));
            for (size_t i = 0; i < 962; ++i) CHECK(isfinite(y[i]));
            memcpy(s, u, info->state_floats * sizeof(float));
        }
        /* Reset reproduces the first hop despite the context's reused scratch. */
        float x[962], y[962], z[962]; synthetic_spectrum(x, 0, 1.0f);
        CHECK(!dpdfnet_init_state(id, s)); CHECK(!dpdfnet_init_state(id, t));
        CHECK(!dpdfnet_process(a, x, s, y, s)); CHECK(!dpdfnet_process(b, x, t, z, t));
        CHECK(!memcmp(y, z, sizeof(y)) && !memcmp(s, t, info->state_floats * sizeof(float)));
        dpdfnet_destroy(b); free(t); free(u);
        contexts[index] = a; states[index] = s;
    }
    /* Both graphs remain live in one process and can be called alternately. */
    for (int i = 0; i < 2; ++i) if (contexts[i]) {
        float x[962]; synthetic_spectrum(x, 13, 1.0f);
        CHECK(!dpdfnet_process(contexts[i], x, states[i], x, states[i]));
        dpdfnet_destroy(contexts[i]); free(states[i]);
    }
    puts("Public API, both graphs, weight ownership, reset and in-place contracts passed.");
    return 0;
}
