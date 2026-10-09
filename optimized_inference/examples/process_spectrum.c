/* Raw spectrum example; FFT, windowing and PCM synthesis belong to the host. */
#include "../tools/support.h"
#include <sys/stat.h>
int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s 2|8 weights.f32 input_spectra.f32 output_spectra.f32\n", argv[0]);
        return 2;
    }
    dpdfnet_model_id id = (dpdfnet_model_id)atoi(argv[1]);
    const dpdfnet_model_info *info = dpdfnet_get_model_info(id);
    if (!info) { fputs("Model ID must be 2 or 8.\n", stderr); return 2; }
    struct stat input_stat, output_stat;
    if (!stat(argv[3], &input_stat) && !stat(argv[4], &output_stat) &&
            input_stat.st_dev == output_stat.st_dev && input_stat.st_ino == output_stat.st_ino) {
        fputs("Input and output must be different files.\n", stderr); return 2;
    }
    float *weights = read_weights(argv[2], info);
    dpdfnet_model *m = weights ? dpdfnet_create(id, weights, info->weight_floats) : NULL;
    free(weights);
    float *state = calloc(info->state_floats, sizeof(float));
    FILE *input = NULL, *output = NULL;
    int status = 1;
    if (!m || !state || dpdfnet_init_state(id, state)) goto cleanup;
    input = fopen(argv[3], "rb");
    if (!input) { perror(argv[3]); goto cleanup; }
    output = fopen(argv[4], "wb");
    if (!output) { perror(argv[4]); goto cleanup; }
    float spectrum[962];
    size_t count, hops = 0;
    while ((count = fread(spectrum, 1, sizeof(spectrum), input)) == sizeof(spectrum)) {
        for (size_t i = 0; i < 962; ++i) if (!isfinite(spectrum[i])) goto cleanup;
        if (dpdfnet_process(m, spectrum, state, spectrum, state) ||
            fwrite(spectrum, sizeof(float), 962, output) != 962) goto cleanup;
        ++hops;
    }
    if (count || ferror(input)) { fputs("Truncated spectrum or input error.\n", stderr); goto cleanup; }
    printf("Processed %zu hops with %s.\n", hops, info->name);
    status = 0;
cleanup:
    if (input && fclose(input)) status = 1;
    if (output && fclose(output)) status = 1;
    free(state);
    dpdfnet_destroy(m);
    return status;
}
