/* Unit tests for the compute kernels and the thread pool. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"
#include "threadpool.h"
#include "util.h"

#define MAX_BLOCKS 64u

static int failures;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                      \
            fputc('\n', stderr);                               \
            ++failures;                                        \
        }                                                      \
    } while (0)

static uint32_t rng = 12345;
static uint32_t next(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng >> 8;
}
static float uniform(float lo, float hi) {
    return lo + (hi - lo) * (float)next() / 16777216.0f;
}

static void random_weights(unsigned char *w, uint32_t nb, int extremes) {
    uint32_t b, j;
    for (b = 0; b < nb; ++b) {
        float scale = uniform(0.0f, 0.05f);
        unsigned char *blk = w + b * Q8_BLOCK_BYTES;
        memcpy(blk, &scale, 4);
        for (j = 0; j < QK; ++j) {
            int v = (int)(next() % 255) - 127;
            if (extremes && j % 7 == 0) v = (j % 2) ? -128 : 127;
            blk[4 + j] = (unsigned char)(int8_t)v;
        }
    }
}

static void test_quantize(void) {
    float x[100], d[4];
    int8_t q[4 * QK];
    uint32_t i;
    for (i = 0; i < 100; ++i) x[i] = uniform(-3.0f, 3.0f);
    CHECK(quantize_q8(x, 100, d, q) == 0, "quantize failed");
    for (i = 0; i < 100; ++i) {
        float back = d[i / QK] * (float)q[i];
        CHECK(fabsf(back - x[i]) <= d[i / QK] * 0.5f + 1e-6f, "round trip error at %u", i);
        CHECK(q[i] >= -127 && q[i] <= 127, "range at %u", i);
    }
    for (i = 100; i < 4 * QK; ++i) CHECK(q[i] == 0, "padding not zero at %u", i);
    memset(x, 0, sizeof(x));
    CHECK(quantize_q8(x, 32, d, q) == 0 && d[0] == 0 && q[5] == 0, "zero block");
    x[3] = INFINITY;
    CHECK(quantize_q8(x, 32, d, q) == -1, "infinite input accepted");
    x[3] = NAN;
    CHECK(quantize_q8(x, 32, d, q) == -1, "NaN input accepted");
}

static void test_dot_against_reference(void) {
    unsigned char w[MAX_BLOCKS * Q8_BLOCK_BYTES];
    float x[MAX_BLOCKS * QK], xd[MAX_BLOCKS];
    int8_t xq[MAX_BLOCKS * QK];
    uint32_t nb, i;
    for (nb = 1; nb <= MAX_BLOCKS; ++nb) {
        double exact = 0;
        float got;
        random_weights(w, nb, 0);
        for (i = 0; i < nb * QK; ++i) x[i] = uniform(-2.0f, 2.0f);
        quantize_q8(x, nb * QK, xd, xq);
        for (i = 0; i < nb * QK; ++i) {
            const unsigned char *blk = w + (i / QK) * Q8_BLOCK_BYTES;
            exact += (double)rd_f32(blk) * (double)(int8_t)blk[4 + i % QK] * (double)xd[i / QK] * (double)xq[i];
        }
        got = dot_q8_scalar(w, xd, xq, nb);
        CHECK(fabs((double)got - exact) <= 1e-4 * (1.0 + fabs(exact)), "nb=%u got %g want %g", nb, got, exact);
    }
}

/* Each x4 output must equal the single-row scalar kernel bit for bit. */
static void test_x4(DotQ8x4 fn, const char *name) {
    static unsigned char w[MAX_BLOCKS * Q8_BLOCK_BYTES];
    static float x[4 * MAX_BLOCKS * QK], xd[4 * MAX_BLOCKS];
    static int8_t xq[4 * MAX_BLOCKS * QK];
    uint32_t trial, i, t;
    for (trial = 0; trial < 100; ++trial) {
        uint32_t nb = 1 + next() % MAX_BLOCKS;
        float out[4];
        random_weights(w, nb, trial % 2);
        for (i = 0; i < 4 * nb * QK; ++i) x[i] = uniform(-20.0f, 20.0f);
        quantize_q8(x, 4 * nb * QK, xd, xq);
        fn(w, xd, xq, nb, nb, out);
        for (t = 0; t < 4; ++t) {
            float want = dot_q8_scalar(w, xd + t * nb, xq + t * nb * QK, nb);
            CHECK(memcmp(&want, &out[t], sizeof(want)) == 0, "%s x4 trial %u row %u", name, trial, t);
        }
    }
}

static void test_backends_bit_identical(void) {
#if defined(SM_HAVE_AVX2)
    unsigned char w[MAX_BLOCKS * Q8_BLOCK_BYTES];
    float x[MAX_BLOCKS * QK], xd[MAX_BLOCKS];
    int8_t xq[MAX_BLOCKS * QK];
    uint32_t nb, trial;
    if (strcmp(kernels_select(0), "avx2")) {
        puts("avx2 unavailable on this CPU; backend identity skipped");
        return;
    }
    for (trial = 0; trial < 200; ++trial) {
        uint32_t i;
        float a, b;
        nb = 1 + next() % MAX_BLOCKS;
        random_weights(w, nb, trial % 3 == 0);
        for (i = 0; i < nb * QK; ++i) x[i] = uniform(-50.0f, 50.0f);
        quantize_q8(x, nb * QK, xd, xq);
        a = dot_q8_scalar(w, xd, xq, nb);
        b = dot_q8_avx2(w, xd, xq, nb);
        CHECK(memcmp(&a, &b, sizeof(a)) == 0, "trial %u nb=%u scalar %a avx2 %a", trial, nb, a, b);
    }
    test_x4(dot_q8_avx2_x4, "avx2");
#else
    puts("built without SIMD; backend identity skipped");
#endif
}

static void test_dot_f32(void) {
    float a[19], b[19];
    uint32_t i;
    double exact = 0;
    for (i = 0; i < 19; ++i) {
        a[i] = uniform(-1, 1);
        b[i] = uniform(-1, 1);
        exact += (double)a[i] * b[i];
    }
    CHECK(fabs(dot_f32(a, b, 19) - exact) < 1e-5, "dot_f32");
}

typedef struct SumTask {
    const float *values;
    uint32_t n;
    float *partial;
    uint32_t *hits;
} SumTask;

static void sum_task(void *ctx, uint32_t index, uint32_t count) {
    SumTask *t = (SumTask *)ctx;
    uint32_t begin, end, i;
    split_range(t->n, index, count, &begin, &end);
    for (i = begin; i < end; ++i) {
        t->partial[i] = t->values[i] * 2.0f;
        __atomic_add_fetch(&t->hits[i], 1, __ATOMIC_RELAXED);
    }
}

static void test_pool(void) {
    static float values[1000], partial[1000];
    static uint32_t hits[1000];
    uint32_t threads, round, i;
    for (i = 0; i < 1000; ++i) values[i] = (float)i;
    for (threads = 1; threads <= 8; ++threads) {
        Memory mem;
        Pool *pool;
        SumTask task;
        memset(&mem, 0, sizeof(mem));
        mem.limit = 1u << 20;
        pool = pool_create(threads, &mem);
        task.values = values;
        task.n = 1000;
        task.partial = partial;
        task.hits = hits;
        memset(hits, 0, sizeof(hits));
        for (round = 0; round < 500; ++round) pool_run(pool, sum_task, &task);
        for (i = 0; i < 1000; ++i) {
            CHECK(hits[i] == 500, "threads=%u index %u ran %u times", threads, i, hits[i]);
            CHECK(partial[i] == 2.0f * (float)i, "threads=%u wrong value", threads);
        }
        pool_destroy(pool);
        mem_release(&mem);
    }
}

static void test_split_range(void) {
    uint32_t total, count, index, covered;
    for (total = 0; total < 50; ++total)
        for (count = 1; count < 9; ++count) {
            uint32_t prev_end = 0;
            covered = 0;
            for (index = 0; index < count; ++index) {
                uint32_t b, e;
                split_range(total, index, count, &b, &e);
                CHECK(b == prev_end && b <= e, "split gap");
                covered += e - b;
                prev_end = e;
            }
            CHECK(covered == total, "split coverage");
        }
}

int main(void) {
    test_quantize();
    test_dot_against_reference();
    test_x4(dot_q8_scalar_x4, "scalar");
    test_backends_bit_identical();
    test_dot_f32();
    test_split_range();
    test_pool();
    if (failures) {
        fprintf(stderr, "%d kernel test(s) failed\n", failures);
        return 1;
    }
    puts("kernel tests passed");
    return 0;
}
