#define _POSIX_C_SOURCE 200809L
#include "../tools/support.h"
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/resource.h>
static int64_t nanoseconds(clockid_t clock) {
    struct timespec t; if (clock_gettime(clock, &t)) { perror("clock_gettime"); exit(1); }
    return (int64_t)t.tv_sec * INT64_C(1000000000) + t.tv_nsec;
}
static size_t rss(void) {
    FILE *f = fopen("/proc/self/statm", "r"); unsigned long total, resident;
    if (!f) return 0;
    int ok = fscanf(f, "%lu %lu", &total, &resident) == 2; fclose(f);
    return ok ? (size_t)resident * (size_t)sysconf(_SC_PAGESIZE) : 0;
}
static int compare(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y);
}
/* perf stat/record --delay=-1 --control=fifo:CONTROL,ACK. Optional control is
 * outside the timed region; ordinary latency runs never attach a profiler. */
static int profiler_command(FILE *control, FILE *ack, const char *command) {
    if (!control) return 0;
    if (fprintf(control, "%s\n", command) < 0 || fflush(control)) return -1;
    char response[32];
    /* perf versions may NUL-pad their acknowledgement. */
    int first;
    do { first = fgetc(ack); } while (first == 0);
    if (first == EOF) {
        fprintf(stderr, "perf acknowledgement missing after %s\n", command); return -1;
    }
    response[0] = (char)first;
    if (!fgets(response + 1, sizeof(response) - 1, ack)) return -1;
    if (strcmp(response, "ack\n")) {
        fprintf(stderr, "Unexpected perf acknowledgement after %s: %s\n", command, response); return -1;
    }
    return 0;
}
int main(int argc, char **argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "Usage: %s 2|8 weights.f32 [hops=1000] [paced|continuous] [spectra.f32]\n", argv[0]); return 2;
    }
    dpdfnet_model_id id = (dpdfnet_model_id)atoi(argv[1]);
    const dpdfnet_model_info *info = dpdfnet_get_model_info(id);
    size_t hops = argc > 3 ? (size_t)strtoul(argv[3], NULL, 10) : 1000;
    int paced = argc < 5 || !strcmp(argv[4], "paced");
    if (!info || hops < 100 || hops > 1000000 || (argc > 4 && strcmp(argv[4], "paced") && strcmp(argv[4], "continuous"))) return 2;
    const char *control_path = getenv("DPDF_PERF_CONTROL"), *ack_path = getenv("DPDF_PERF_ACK");
    FILE *control = control_path ? fopen(control_path, "w") : NULL;
    FILE *ack = ack_path ? fopen(ack_path, "r") : NULL;
    if ((control_path || ack_path) && (!control || !ack)) return 1;
    size_t frame_count = 256;
    FILE *input = argc > 5 ? fopen(argv[5], "rb") : NULL;
    if (argc > 5) {
        if (!input || fseek(input, 0, SEEK_END)) return 1;
        long bytes = ftell(input); if (bytes <= 0 || bytes % (962 * 4)) return 1;
        frame_count = (size_t)bytes / (962 * 4); rewind(input);
    }
    float *frames = malloc(frame_count * 962 * sizeof(float));
    double *times = malloc(hops * sizeof(double));
    float *state = calloc(info->state_floats, sizeof(float));
    if (!frames || !times || !state) return 1;
    if (input) {
        if (fread(frames, sizeof(float), frame_count * 962, input) != frame_count * 962) return 1;
        fclose(input);
    } else for (size_t i = 0; i < frame_count; ++i) synthetic_spectrum(frames + i * 962, i, 1.0f);
    for (size_t i = 0; i < frame_count * 962; ++i) if (!isfinite(frames[i])) return 1;
    size_t before = rss();
    float *weights = read_weights(argv[2], info);
    dpdfnet_model *m = weights ? dpdfnet_create(id, weights, info->weight_floats) : NULL;
    free(weights);
    if (!m || dpdfnet_init_state(id, state)) return 1;
    double sum = 0, cpu = 0, mean = 0, m2 = 0; size_t over = 0, late = 0;
    struct rusage usage_before = {0}, usage_after = {0};
    float output[962];
    int64_t deadline = nanoseconds(CLOCK_MONOTONIC);
    for (size_t i = 0; i < hops + 120; ++i) {
        if (paced) {
            struct timespec t = {deadline / INT64_C(1000000000), deadline % INT64_C(1000000000)};
            while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL)) {}
        }
        if (i == 120 && (profiler_command(control, ack, "enable") || getrusage(RUSAGE_SELF, &usage_before))) return 1;
        int64_t c0 = nanoseconds(CLOCK_THREAD_CPUTIME_ID), t0 = nanoseconds(CLOCK_MONOTONIC);
        if (dpdfnet_process(m, frames + (i % frame_count) * 962, state, output, state)) return 1;
        int64_t t1 = nanoseconds(CLOCK_MONOTONIC), c1 = nanoseconds(CLOCK_THREAD_CPUTIME_ID);
        if (i >= 120) {
            double ms = (double)(t1 - t0) / 1e6; times[i - 120] = ms; sum += ms;
            double delta = ms - mean; mean += delta / (double)(i - 119); m2 += delta * (ms - mean);
            cpu += (double)(c1 - c0) / 1e6; over += ms > 10;
            late += paced && t1 > deadline + INT64_C(10000000);
        }
        deadline += INT64_C(10000000);
        if (paced && t1 > deadline) deadline = t1; /* Record miss before restarting cadence. */
    }
    if (profiler_command(control, ack, "disable") || getrusage(RUSAGE_SELF, &usage_after)) return 1;
    if (control) fclose(control);
    if (ack) fclose(ack);
    const char *samples = getenv("DPDF_BENCH_SAMPLES");
    if (samples) {
        FILE *f = fopen(samples, "wb");
        if (!f) return 1;
        size_t written = fwrite(times, sizeof(double), hops, f);
        int closed = fclose(f);
        if (written != hops || closed) return 1;
    }
    size_t after = rss(); qsort(times, hops, sizeof(double), compare);
    printf("{\"model\":\"%s\",\"input\":\"%s\",\"mode\":\"%s\",\"hops\":%zu,"
           "\"mean_ms\":%.9f,\"cpu_ms\":%.9f,\"p99_ms\":%.9f,\"max_ms\":%.9f,"
           "\"min_ms\":%.9f,\"median_ms\":%.9f,\"p95_ms\":%.9f,\"p999_ms\":%.9f,\"hop_sd_ms\":%.9f,"
           "\"voluntary_switches\":%ld,\"involuntary_switches\":%ld,\"minor_faults\":%ld,\"major_faults\":%ld,"
           "\"over_10ms\":%zu,\"late\":%zu,\"owned_bytes\":%zu,\"rss_bytes\":%zu,\"rss_increment_bytes\":%zd}\n",
           info->name, input ? "file" : "synthetic", paced ? "paced" : "continuous", hops,
           sum / hops, cpu / hops, times[(hops * 99) / 100], times[hops - 1],
           times[0], (times[(hops - 1) / 2] + times[hops / 2]) / 2,
           times[(hops * 95) / 100], times[(hops * 999) / 1000], sqrt(m2 / (double)(hops - 1)),
           usage_after.ru_nvcsw - usage_before.ru_nvcsw, usage_after.ru_nivcsw - usage_before.ru_nivcsw,
           usage_after.ru_minflt - usage_before.ru_minflt, usage_after.ru_majflt - usage_before.ru_majflt, over, late,
           dpdfnet_owned_bytes(m), after, (ptrdiff_t)after - (ptrdiff_t)before);
    dpdfnet_destroy(m); free(state); free(times); free(frames);
    return 0;
}
