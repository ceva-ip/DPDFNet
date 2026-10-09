#include "support.h"
int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "Usage: %s model8/weights.f32\n", argv[0]); return 2; }
    const dpdfnet_model_info *info = dpdfnet_get_model_info(DPDFNET_MODEL_8);
    float *weights = read_weights(argv[1], info);
    dpdfnet_model *m = weights ? dpdfnet_create(DPDFNET_MODEL_8, weights, info->weight_floats) : NULL;
    free(weights);
    float *state = calloc(info->state_floats, sizeof(float));
    if (!m || !state) { free(state); dpdfnet_destroy(m); return 1; }
    const float scales[] = {1.0f, 0.001f, 8.0f, 0.0f};
    float spec[962];
    for (size_t stream = 0; stream < 4; ++stream) {
        if (dpdfnet_init_state(DPDFNET_MODEL_8, state)) return 1;
        for (size_t frame = 0; frame < 1000; ++frame) {
            synthetic_spectrum(spec, frame, scales[stream]);
            if (dpdfnet_process(m, spec, state, spec, state)) return 1;
        }
    }
    dpdfnet_destroy(m);
    free(state);
    puts("Model-8 profile collected: four independent streams, 4000 hops.");
    return 0;
}
