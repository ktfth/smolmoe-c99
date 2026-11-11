/* Microbenchmark of the Q8 matmul. Not part of `make test`.
 * Reports weight bandwidth and the decode/prefill rate a 556 MiB model would
 * see if every token were this kernel. The "before" loops are the previous
 * one-row dot / x4 scans; "after" is the tiled gemm. */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"
#include "platform.h"

#define MODEL_BYTES (556ull << 20)

typedef struct Job {
    void (*fn)(const struct Job *, uint32_t index, uint32_t count);
    const unsigned char *w;
    size_t row_bytes;
    const float *xd;
    const int8_t *xq;
    float *out;
    unsigned char *scratch;
    size_t scratch_stride;
    uint32_t rows, nb, ntok;
    DotQ8 dot;
    DotQ8x4 x4;
    GemmQ8 gemm;
} Job;

static void split(uint32_t total, uint32_t index, uint32_t count, uint32_t *b, uint32_t *e) {
    uint32_t base = total / count, extra = total % count;
    *b = index * base + (index < extra ? index : extra);
    *e = *b + base + (index < extra ? 1 : 0);
}

static void old_rows(const Job *j, uint32_t index, uint32_t count) {
    uint32_t b, e, r, t;
    split(j->rows, index, count, &b, &e);
    for (r = b; r < e; ++r) {
        const unsigned char *row = j->w + (size_t)r * j->row_bytes;
        for (t = 0; t + 4 <= j->ntok; t += 4) {
            float y[4];
            j->x4(row, j->xd + (size_t)t * j->nb, j->xq + (size_t)t * j->nb * QK, j->nb, j->nb, y);
            if (j->out) memcpy(j->out + (size_t)t * j->rows + r, y, sizeof(y)); /* wrong layout, just a sink */
        }
        for (; t < j->ntok; ++t) {
            float y = j->dot(row, j->xd + (size_t)t * j->nb, j->xq + (size_t)t * j->nb * QK, j->nb);
            if (j->out) j->out[(size_t)t * j->rows + r] = y;
        }
    }
}

static void new_rows(const Job *j, uint32_t index, uint32_t count) {
    uint32_t b, e;
    unsigned char *scratch = j->scratch ? j->scratch + (size_t)index * j->scratch_stride : NULL;
    split(j->rows, index, count, &b, &e);
    if (b < e)
        j->gemm(j->w + (size_t)b * j->row_bytes, j->row_bytes, j->xd, j->xq, e - b, j->ntok, j->nb,
                j->out ? j->out + b : NULL, j->rows, scratch);
}

typedef struct Thread {
    const Job *job;
    uint32_t index, count;
} Thread;

static void *thread_main(void *arg) {
    Thread *t = (Thread *)arg;
    t->job->fn(t->job, t->index, t->count);
    return NULL;
}

static double once(Job *job, uint32_t threads) {
    pthread_t th[32];
    Thread arg[32];
    uint32_t i;
    double t0, t1;
    if (threads > 32) threads = 32;
    if (threads < 1) threads = 1;
    t0 = now_seconds();
    for (i = 1; i < threads; ++i) {
        arg[i].job = job;
        arg[i].index = i;
        arg[i].count = threads;
        pthread_create(&th[i], NULL, thread_main, &arg[i]);
    }
    job->fn(job, 0, threads);
    for (i = 1; i < threads; ++i) pthread_join(th[i], NULL);
    t1 = now_seconds();
    return t1 - t0;
}

static double best(Job *job, uint32_t threads, int reps) {
    double b = 1e300;
    int i;
    once(job, threads); /* warmup */
    for (i = 0; i < reps; ++i) {
        double d = once(job, threads);
        if (d < b) b = d;
    }
    return b;
}

static void report(const char *name, double sec, uint64_t weight_bytes, uint32_t ntok) {
    double gbps = (double)weight_bytes / sec / 1e9;
    double model = (double)MODEL_BYTES / (double)weight_bytes * sec;
    printf("%-22s %7.3f s  %6.1f GB/s  %6.1f tok/s-eq (556 MiB, %u tok/pass)\n", name, sec, gbps,
           ntok / model, ntok);
}

int main(void) {
    const uint32_t nb = 152; /* Qwen2.5-0.5B FFN width */
    const uint32_t rows = 65536; /* ~342 MiB, past the per-core caches */
    const uint32_t ntok_prefill = 32;
    const uint32_t threads = 4;
    const size_t row_bytes = (size_t)nb * Q8_BLOCK_BYTES;
    const uint64_t weight_bytes = (uint64_t)rows * row_bytes;
    unsigned char *w;
    float *xd;
    int8_t *xq;
    float *out;
    unsigned char *scratch;
    uint32_t i, t;
    Job job;
    const char *backend;
    printf("matrix %u x %u  %.1f MiB  threads %u\n", rows, nb * QK, (double)weight_bytes / (1 << 20), threads);
    w = (unsigned char *)malloc(weight_bytes);
    xd = (float *)malloc((size_t)ntok_prefill * nb * sizeof(float));
    xq = (int8_t *)malloc((size_t)ntok_prefill * nb * QK);
    out = (float *)malloc((size_t)ntok_prefill * rows * sizeof(float));
    scratch = (unsigned char *)malloc((size_t)threads * gemm_scratch_bytes(nb));
    if (!w || !xd || !xq || !out || !scratch) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    for (i = 0; i < weight_bytes; ++i) w[i] = (unsigned char)(i * 17u + 3u);
    for (t = 0; t < ntok_prefill; ++t)
        for (i = 0; i < nb; ++i) {
            xd[(size_t)t * nb + i] = 0.01f + 0.0001f * (float)((t + i) % 50);
            memset(xq + ((size_t)t * nb + i) * QK, (int)(t + i) % 29 - 14, QK);
        }
    /* Finite scales, so the dots do real work. */
    for (i = 0; i < rows * nb; ++i) {
        float s = 0.02f;
        memcpy(w + (size_t)i * Q8_BLOCK_BYTES, &s, 4);
    }
    backend = kernels_select(0);
    printf("dispatched backend: %s\n", backend);
    memset(&job, 0, sizeof(job));
    job.w = w;
    job.row_bytes = row_bytes;
    job.xd = xd;
    job.xq = xq;
    job.out = out;
    job.scratch = scratch;
    job.scratch_stride = gemm_scratch_bytes(nb);
    job.rows = rows;
    job.nb = nb;
    job.dot = dot_q8_avx2;
    job.x4 = dot_q8_avx2_x4;

    job.fn = old_rows;
    job.gemm = NULL;
    job.ntok = 1;
    report("before decode avx2", best(&job, threads, 3), weight_bytes, 1);
    job.ntok = ntok_prefill;
    report("before prefill avx2", best(&job, threads, 2), weight_bytes, ntok_prefill);

    job.fn = new_rows;
    job.gemm = gemm_q8_avx2;
    job.ntok = 1;
    report("after decode avx2", best(&job, threads, 3), weight_bytes, 1);
    job.ntok = ntok_prefill;
    report("after prefill avx2", best(&job, threads, 2), weight_bytes, ntok_prefill);

    job.gemm = gemm_q8;
    job.ntok = 1;
    report("after decode best", best(&job, threads, 3), weight_bytes, 1);
    job.ntok = ntok_prefill;
    report("after prefill best", best(&job, threads, 2), weight_bytes, ntok_prefill);

    kernels_set_fast(1);
    printf("fast backend: %s\n", kernels_select(0));
    job.gemm = gemm_q8;
    job.ntok = 1;
    report("fast decode", best(&job, threads, 3), weight_bytes, 1);
    job.ntok = ntok_prefill;
    report("fast prefill", best(&job, threads, 2), weight_bytes, ntok_prefill);

    /* Touch the sink so the stores cannot be deleted. */
    volatile float sink = out[0] + out[(size_t)ntok_prefill * rows - 1];
    if (sink == 1e30f) printf("sink\n");
    free(w);
    free(xd);
    free(xq);
    free(out);
    free(scratch);
    return 0;
}
