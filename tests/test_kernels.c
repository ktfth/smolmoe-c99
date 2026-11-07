/* Unit tests for the compute kernels and the thread pool. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "f16.h"
#include "kernels.h"
#include "threadpool.h"
#include "util.h"
#include "vecmath.h"

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

static void test_f16(void) {
    uint32_t h;
    for (h = 0; h < 65536; ++h) {
        float f = f16_to_f32((uint16_t)h);
        if (f != f) continue; /* NaN payloads need not round-trip */
        CHECK(f32_to_f16(f) == h, "f16 round trip 0x%04x", h);
    }
    CHECK(f32_to_f16(1.0f) == 0x3c00 && f32_to_f16(-2.0f) == 0xc000, "f16 constants");
    CHECK(f32_to_f16(65520.0f) == 0x7c00, "f16 overflow rounds to inf");
    CHECK(f32_to_f16(1.0f + 1.0f / 2048) == 0x3c00, "f16 ties to even");
}

/* Random rows packed with quantize_q5_row. */
static void random_q5(unsigned char *w, uint32_t nb, float spread) {
    static float row[MAX_BLOCKS * QK];
    uint32_t i;
    for (i = 0; i < nb * QK; ++i) row[i] = uniform(-spread, spread);
    quantize_q5_row(row, nb * QK, w);
}

static void test_q5_roundtrip(void) {
    static float x[MAX_BLOCKS * QK], back[MAX_BLOCKS * QK];
    static unsigned char w[MAX_BLOCKS * Q5_BLOCK_BYTES];
    uint32_t i, cols = 100;
    for (i = 0; i < cols; ++i) x[i] = uniform(-1.0f, 1.0f);
    x[7] = 3.0f; /* outlier sets the first block scale */
    quantize_q5_row(x, cols, w);
    dequantize_q5_row(w, cols, back);
    for (i = 0; i < cols; ++i) {
        float amax = i < 32 ? 3.0f : 1.0f;
        CHECK(fabsf(back[i] - x[i]) <= amax / 15.0f, "q5 error at %u: %g vs %g", i, back[i], x[i]);
    }
    for (i = 0; i < 32; ++i) x[i] = 0;
    quantize_q5_row(x, 32, w);
    dequantize_q5_row(w, 32, back);
    CHECK(back[0] == 0 && back[31] == 0, "zero block");
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

static void test_x4_q5(DotQ8x4 fn, const char *name) {
    static unsigned char w[MAX_BLOCKS * Q5_BLOCK_BYTES];
    static float x[4 * MAX_BLOCKS * QK], xd[4 * MAX_BLOCKS];
    static int8_t xq[4 * MAX_BLOCKS * QK];
    uint32_t trial, i, t;
    for (trial = 0; trial < 100; ++trial) {
        uint32_t nb = 1 + next() % MAX_BLOCKS;
        float out[4];
        random_q5(w, nb, 2.0f);
        for (i = 0; i < 4 * nb * QK; ++i) x[i] = uniform(-20.0f, 20.0f);
        quantize_q8(x, 4 * nb * QK, xd, xq);
        fn(w, xd, xq, nb, nb, out);
        for (t = 0; t < 4; ++t) {
            float want = dot_q5_scalar(w, xd + t * nb, xq + t * nb * QK, nb);
            CHECK(memcmp(&want, &out[t], sizeof(want)) == 0, "%s q5 x4 trial %u row %u", name, trial, t);
        }
    }
}

/* The SIMD backend this machine selects (AVX2, NEON or SIMD128), reached through the
 * dispatch pointers; NULL when only the scalar code runs here, which fails
 * under SM_EXPECT_SIMD=1 (CI) so the comparisons cannot vanish silently. */
static const char *simd_backend(void) {
    const char *name = kernels_select(0), *expect = getenv("SM_EXPECT_SIMD");
    if (!strcmp(name, "scalar")) {
        CHECK(!(expect && !strcmp(expect, "1")), "SM_EXPECT_SIMD=1 but the scalar backend was selected");
        puts("no SIMD backend on this build or CPU; backend identity skipped");
        return NULL;
    }
    /* A pointer left on the scalar version would make every comparison pass trivially. */
#if defined(SM_HAVE_AVX2)
    CHECK(dot_q8 == dot_q8_avx2 && dot_q8_x4 == dot_q8_avx2_x4 && dot_q5 == dot_q5_avx2 &&
          dot_q5_x4 == dot_q5_avx2_x4 && quantize_q8 == quantize_q8_avx2 && silu_mul == silu_mul_avx2,
          "%s: a dispatch pointer does not reach the SIMD code", name);
#elif defined(SM_HAVE_NEON)
    CHECK(dot_q8 == dot_q8_neon && dot_q8_x4 == dot_q8_neon_x4 && dot_q5 == dot_q5_neon &&
          dot_q5_x4 == dot_q5_neon_x4 && quantize_q8 == quantize_q8_neon && silu_mul == silu_mul_neon,
          "%s: a dispatch pointer does not reach the SIMD code", name);
#elif defined(SM_HAVE_WASM_SIMD)
    CHECK(dot_q8 == dot_q8_wasm && dot_q8_x4 == dot_q8_wasm_x4 && dot_q5 == dot_q5_wasm &&
          dot_q5_x4 == dot_q5_wasm_x4 && quantize_q8 == quantize_q8_wasm && silu_mul == silu_mul_wasm,
          "%s: a dispatch pointer does not reach the SIMD code", name);
#endif
    return name;
}

/* Lane 0 gets -4098 from the first block, then d * s = (1 + 2^-12) * 4097 =
 * 4098 + 2^-12 from the second, which rounds to 4098 (a tie, to even). With
 * the kernels' two roundings the result is exactly 0; a fused multiply-add
 * gives 2^-12. Fails if -ffp-contract=off is lost from the build. */
static void test_no_fused_multiply_add(void) {
    unsigned char w[2 * Q8_BLOCK_BYTES];
    float xd[4 * 2], scale1 = 1.0f, scale2 = 1.0f + 1.0f / 4096, out[4];
    int8_t xq[4 * 2 * QK];
    uint32_t t;
    memset(w, 0, sizeof(w));
    memset(xq, 0, sizeof(xq));
    memcpy(w, &scale1, 4);
    memcpy(w + Q8_BLOCK_BYTES, &scale2, 4);
    w[4] = 127, w[5] = 17;                                          /* 127 * -32 + 17 * -2 = -4098 */
    w[Q8_BLOCK_BYTES + 4] = 127, w[Q8_BLOCK_BYTES + 5] = 1;         /* 127 * 32 + 1 * 33 = 4097 */
    for (t = 0; t < 4; ++t) {
        int8_t *row = xq + t * 2 * QK;
        row[0] = -32, row[1] = -2, row[QK] = 32, row[QK + 1] = 33;
        xd[2 * t] = xd[2 * t + 1] = 1.0f;
    }
    CHECK(dot_q8_scalar(w, xd, xq, 2) == 0.0f, "scalar dot fused a multiply-add: %a",
          (double)dot_q8_scalar(w, xd, xq, 2));
    if (!simd_backend()) return;
    CHECK(dot_q8(w, xd, xq, 2) == 0.0f, "simd dot fused a multiply-add: %a", (double)dot_q8(w, xd, xq, 2));
    dot_q8_x4(w, xd, xq, 2, 2, out);
    for (t = 0; t < 4; ++t) CHECK(out[t] == 0.0f, "simd x4 row %u fused a multiply-add: %a", t, (double)out[t]);
}

/* Q5 blocks with every high bit set, none set and alternating, and Q8 weights
 * at both extremes against activations of -127: the edges random data misses. */
static void test_backend_edges(void) {
    static const uint32_t HIGH[3] = {0xFFFFFFFFu, 0u, 0xAAAAAAAAu};
    static const unsigned char LOW[3] = {0xF0, 0xFF, 0x5A};
    unsigned char w5[3 * Q5_BLOCK_BYTES], w8[3 * Q8_BLOCK_BYTES];
    float xd[3] = {1.0f, 0.5f, 2.0f}, one = 1.0f;
    int8_t xq[3 * QK];
    uint32_t b, j;
    if (!simd_backend()) return;
    for (b = 0; b < 3; ++b) {
        unsigned char *blk = w5 + b * Q5_BLOCK_BYTES;
        blk[0] = 0x00, blk[1] = 0x3c; /* f16 1.0 */
        memcpy(blk + 2, &HIGH[b], 4);
        memset(blk + 6, LOW[b], 16);
        memcpy(w8 + b * Q8_BLOCK_BYTES, &one, 4);
        for (j = 0; j < QK; ++j) {
            w8[b * Q8_BLOCK_BYTES + 4 + j] = (unsigned char)(int8_t)(j % 2 ? -128 : 127);
            xq[b * QK + j] = (int8_t)(b == 2 ? (j % 3 ? 127 : -127) : -127);
        }
    }
    {
        float a5 = dot_q5_scalar(w5, xd, xq, 3), b5 = dot_q5(w5, xd, xq, 3);
        float a8 = dot_q8_scalar(w8, xd, xq, 3), b8 = dot_q8(w8, xd, xq, 3);
        CHECK(memcmp(&a5, &b5, sizeof(a5)) == 0, "q5 high-bit edges: scalar %a simd %a", (double)a5, (double)b5);
        CHECK(memcmp(&a8, &b8, sizeof(a8)) == 0, "q8 extreme weights: scalar %a simd %a", (double)a8, (double)b8);
    }
}

static void test_backends_bit_identical(void) {
    unsigned char w[MAX_BLOCKS * Q8_BLOCK_BYTES];
    float x[MAX_BLOCKS * QK], xd[MAX_BLOCKS];
    int8_t xq[MAX_BLOCKS * QK];
    uint32_t nb, trial;
    const char *simd = simd_backend();
    if (!simd) return;
    for (trial = 0; trial < 200; ++trial) {
        uint32_t i;
        float a, b;
        nb = 1 + next() % MAX_BLOCKS;
        random_weights(w, nb, trial % 3 == 0);
        for (i = 0; i < nb * QK; ++i) x[i] = uniform(-50.0f, 50.0f);
        quantize_q8(x, nb * QK, xd, xq);
        a = dot_q8_scalar(w, xd, xq, nb);
        b = dot_q8(w, xd, xq, nb);
        CHECK(memcmp(&a, &b, sizeof(a)) == 0, "trial %u nb=%u scalar %a %s %a", trial, nb, a, simd, b);
    }
    test_x4(dot_q8_x4, simd);
    for (trial = 0; trial < 200; ++trial) {
        static unsigned char w5[MAX_BLOCKS * Q5_BLOCK_BYTES];
        uint32_t i;
        float a, b;
        nb = 1 + next() % MAX_BLOCKS;
        random_q5(w5, nb, 0.5f + (float)(trial % 7));
        for (i = 0; i < nb * QK; ++i) x[i] = uniform(-50.0f, 50.0f);
        quantize_q8(x, nb * QK, xd, xq);
        a = dot_q5_scalar(w5, xd, xq, nb);
        b = dot_q5(w5, xd, xq, nb);
        CHECK(memcmp(&a, &b, sizeof(a)) == 0, "q5 trial %u nb=%u scalar %a %s %a", trial, nb, a, simd, b);
    }
    test_x4_q5(dot_q5_x4, simd);
}

static int same_bits(const void *a, const void *b, size_t n) { return memcmp(a, b, n) == 0; }

/* Distance in units in the last place between two finite floats of any sign. */
static long ulp_distance(float a, float b) {
    int32_t ia, ib;
    long d;
    memcpy(&ia, &a, sizeof(ia));
    memcpy(&ib, &b, sizeof(ib));
    d = (long)(ia < 0 ? INT32_MIN - ia : ia) - (long)(ib < 0 ? INT32_MIN - ib : ib);
    return d < 0 ? -d : d;
}

static void test_exp_det(void) {
    const uint32_t samples = 1000000;
    uint32_t i, off = 0, far = 0;
    float x;
    /* Whole range, including the gradual underflow below -87.3. */
    for (i = 0; i <= samples; ++i) {
        float want, got;
        long d;
        x = -103.5f + 192.2f * (float)i / (float)samples;
        want = (float)exp((double)x); /* binary64 exp, rounded once: the correctly rounded value */
        got = exp_det(x);
        d = ulp_distance(got, want);
        off += d != 0;
        far += d > 1;
    }
    CHECK(far == 0, "exp_det: %u results more than 1 ulp off", far);
    CHECK(off <= samples / 10000, "exp_det: %u of %u results not correctly rounded", off, samples);
    CHECK(exp_det(0.0f) == 1.0f, "exp_det(0) = %a", (double)exp_det(0.0f));
    CHECK(exp_det(89.0f) == INFINITY && exp_det(INFINITY) == INFINITY, "exp_det overflow");
    CHECK(exp_det(-110.0f) == 0.0f && exp_det(-INFINITY) == 0.0f, "exp_det underflow");
    CHECK(exp_det(-100.0f) > 0.0f, "exp_det keeps subnormal results");
    x = exp_det(NAN);
    CHECK(x != x, "exp_det(NaN)");
}

/* Inputs that cover the polynomial, both clamps, infinities and NaN. */
static float special_or_uniform(uint32_t i, float lo, float hi) {
    /* silu evaluates e^-gate: negative gates reach the overflow side, positive ones the underflow side. */
    static const float special[] = {0.0f, -0.0f, 88.7f, 88.8f, -88.7f, -88.72f, -88.8f, -89.0f, 103.0f, 120.0f,
                                    1e-30f, -1e-30f, 200.0f, -200.0f, INFINITY, -INFINITY, NAN};
    return i % 9 == 0 ? special[(i / 9) % (sizeof(special) / sizeof(special[0]))] : uniform(lo, hi);
}

static void test_silu_backends(void) {
    static float gate[1003], up_scalar[1003], up_simd[1003];
    uint32_t n, i;
    for (i = 0; i < 1003; ++i) {
        gate[i] = special_or_uniform(i, -30.0f, 30.0f);
        up_scalar[i] = up_simd[i] = uniform(-4.0f, 4.0f);
    }
    silu_mul_scalar(up_scalar, gate, 1003);
    for (i = 0; i < 1003; ++i) {
        float want = up_simd[i] * (gate[i] / (1.0f + (float)exp(-(double)gate[i]))), got = up_scalar[i];
        int ok = isnan(want) ? isnan(got) : isinf(want) ? got == want : fabsf(got - want) <= 1e-6f * fabsf(want) + 1e-30f;
        CHECK(ok, "silu %u (gate %a): %a vs %a", i, (double)gate[i], (double)got, (double)want);
    }
    if (!simd_backend()) return;
    for (n = 0; n <= 19; ++n) { /* every tail length */
        for (i = 0; i < n; ++i) up_scalar[i] = up_simd[i] = uniform(-4.0f, 4.0f);
        silu_mul_scalar(up_scalar, gate, n);
        silu_mul(up_simd, gate, n);
        CHECK(same_bits(up_scalar, up_simd, n * sizeof(float)), "silu simd differs at n=%u", n);
    }
    for (i = 0; i < 1003; ++i) up_scalar[i] = up_simd[i] = uniform(-4.0f, 4.0f);
    silu_mul_scalar(up_scalar, gate, 1003);
    silu_mul(up_simd, gate, 1003);
    CHECK(same_bits(up_scalar, up_simd, sizeof(up_simd)), "silu simd differs");
}

static void test_quantize_backends(void) {
    static float x[MAX_BLOCKS * QK + 7];
    static int8_t qa[MAX_BLOCKS * QK + QK], qb[MAX_BLOCKS * QK + QK];
    float da[MAX_BLOCKS + 1], db[MAX_BLOCKS + 1];
    uint32_t trial, i;
    if (!simd_backend()) return;
    for (trial = 0; trial < 300; ++trial) {
        uint32_t n = 1 + next() % (MAX_BLOCKS * QK + 7), nb = (n + QK - 1) / QK;
        /* 1e-37 gives scales near FLT_MIN, 1e-39 subnormal inputs whose blocks quantize to zeros. */
        float spread = trial % 6 == 0 ? 1e-30f : trial % 6 == 1 ? 1e30f : trial % 6 == 2 ? 1e-37f
                     : trial % 6 == 3 ? 1e-39f : 50.0f;
        for (i = 0; i < n; ++i) x[i] = trial % 5 == 0 && i % 33 == 0 ? 0.0f : uniform(-spread, spread);
        CHECK(quantize_q8_scalar(x, n, da, qa) == 0 && quantize_q8(x, n, db, qb) == 0, "quantize trial %u", trial);
        CHECK(same_bits(da, db, nb * sizeof(float)) && same_bits(qa, qb, (size_t)nb * QK),
              "quantize simd differs: trial %u n=%u", trial, n);
    }
    /* Non-finite values in the first and last lanes of a vector, in a later block and in the partial tail. */
    {
        static const uint32_t AT[] = {0, 7, 31, 40, 66};
        uint32_t k;
        for (k = 0; k < sizeof(AT) / sizeof(AT[0]); ++k) {
            for (i = 0; i < 67; ++i) x[i] = 1.0f;
            x[AT[k]] = k % 2 ? NAN : INFINITY;
            CHECK(quantize_q8(x, 67, db, qb) == -1, "simd quantize accepted %s at %u", k % 2 ? "NaN" : "inf", AT[k]);
            CHECK(quantize_q8_scalar(x, 67, da, qa) == -1, "scalar quantize accepted a non-finite value at %u", AT[k]);
        }
    }
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
    test_f16();
    test_quantize();
    test_q5_roundtrip();
    test_dot_against_reference();
    test_x4(dot_q8_scalar_x4, "scalar");
    test_x4_q5(dot_q5_scalar_x4, "scalar");
    test_backends_bit_identical();
    test_no_fused_multiply_add();
    test_backend_edges();
    test_exp_det();
    test_silu_backends();
    test_quantize_backends();
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
