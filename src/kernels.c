#include "kernels.h"
#include "util.h"

#include <math.h>

#if defined(SM_HAVE_AVX2)
#  include <immintrin.h>
#endif

DotQ8 dot_q8 = dot_q8_scalar;
DotQ8x4 dot_q8_x4 = dot_q8_scalar_x4;

/* Lane reduction shared by every backend: ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)). */
static float reduce8(const float a[8]) {
    float s0 = a[0] + a[4], s1 = a[1] + a[5], s2 = a[2] + a[6], s3 = a[3] + a[7];
    return (s0 + s2) + (s1 + s3);
}

/* Lane l of a block holds the exact integer sum of products 4l..4l+3,
 * matching the AVX2 maddubs/madd layout. */
float dot_q8_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t b, lane, k;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK) {
        const int8_t *wq = (const int8_t *)(w + 4);
        float d = rd_f32(w) * xd[b];
        for (lane = 0; lane < 8; ++lane) {
            int32_t sum = 0;
            for (k = 0; k < 4; ++k) sum += (int32_t)wq[4 * lane + k] * (int32_t)xq[4 * lane + k];
            acc[lane] += d * (float)sum;
        }
    }
    return reduce8(acc);
}

void dot_q8_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]) {
    uint32_t t;
    for (t = 0; t < 4; ++t)
        out[t] = dot_q8_scalar(w, xd + (size_t)t * stride, xq + (size_t)t * stride * QK, nb);
}

#if defined(SM_HAVE_AVX2)
__attribute__((target("avx2")))
static float reduce_avx2(__m256 acc) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

/* Exact integer products of one block: |w| * sign(x, w) == w * x. */
#define Q8_BLOCK_PRODUCT(aw, wv, xv, ones) \
    _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16((aw), _mm256_sign_epi8((xv), (wv))), (ones)))

__attribute__((target("avx2")))
void dot_q8_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]) {
    const __m256i ones = _mm256_set1_epi16(1);
    const size_t qs = (size_t)stride * QK;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK, ++xd) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)(const void *)(w + 4));
        __m256i aw = _mm256_sign_epi8(wv, wv);
        float ws = rd_f32(w);
        __m256i x0 = _mm256_loadu_si256((const __m256i *)(const void *)xq);
        __m256i x1 = _mm256_loadu_si256((const __m256i *)(const void *)(xq + qs));
        __m256i x2 = _mm256_loadu_si256((const __m256i *)(const void *)(xq + 2 * qs));
        __m256i x3 = _mm256_loadu_si256((const __m256i *)(const void *)(xq + 3 * qs));
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_set1_ps(ws * xd[0]), Q8_BLOCK_PRODUCT(aw, wv, x0, ones)));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_set1_ps(ws * xd[stride]), Q8_BLOCK_PRODUCT(aw, wv, x1, ones)));
        a2 = _mm256_add_ps(a2, _mm256_mul_ps(_mm256_set1_ps(ws * xd[2 * stride]), Q8_BLOCK_PRODUCT(aw, wv, x2, ones)));
        a3 = _mm256_add_ps(a3, _mm256_mul_ps(_mm256_set1_ps(ws * xd[3 * stride]), Q8_BLOCK_PRODUCT(aw, wv, x3, ones)));
    }
    out[0] = reduce_avx2(a0);
    out[1] = reduce_avx2(a1);
    out[2] = reduce_avx2(a2);
    out[3] = reduce_avx2(a3);
}

__attribute__((target("avx2")))
float dot_q8_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK) {
        __m256i wv = _mm256_loadu_si256((const __m256i *)(const void *)(w + 4));
        __m256i xv = _mm256_loadu_si256((const __m256i *)(const void *)xq);
        __m256 d = _mm256_set1_ps(rd_f32(w) * xd[b]);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, Q8_BLOCK_PRODUCT(_mm256_sign_epi8(wv, wv), wv, xv, ones)));
    }
    return reduce_avx2(acc);
}
#endif

const char *kernels_select(int force_scalar) {
    dot_q8 = dot_q8_scalar;
    dot_q8_x4 = dot_q8_scalar_x4;
#if defined(SM_HAVE_AVX2)
    if (!force_scalar && __builtin_cpu_supports("avx2")) {
        dot_q8 = dot_q8_avx2;
        dot_q8_x4 = dot_q8_avx2_x4;
        return "avx2";
    }
#else
    (void)force_scalar;
#endif
    return "scalar";
}

int quantize_q8(const float *x, uint32_t n, float *d, int8_t *q) {
    uint32_t nb = (n + QK - 1) / QK, b, j;
    for (b = 0; b < nb; ++b, x += QK, q += QK) {
        uint32_t count = n - b * QK < QK ? n - b * QK : QK;
        float amax = 0, scale, inv;
        for (j = 0; j < count; ++j) {
            float a = fabsf(x[j]);
            if (!(a <= FLT_MAX)) return -1; /* rejects both inf and NaN */
            if (a > amax) amax = a;
        }
        scale = amax / 127.0f;
        inv = scale > 0 ? 1.0f / scale : 0.0f;
        d[b] = scale;
        for (j = 0; j < QK; ++j) {
            long v = j < count ? lrintf(x[j] * inv) : 0;
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            q[j] = (int8_t)v;
        }
    }
    return 0;
}

void dequantize_q8_row(const unsigned char *w, uint32_t cols, float *out) {
    uint32_t b, j, nb = (cols + QK - 1) / QK;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES) {
        float scale = rd_f32(w);
        uint32_t n = cols - b * QK < QK ? cols - b * QK : QK;
        for (j = 0; j < n; ++j) out[b * QK + j] = scale * (float)((const int8_t *)(w + 4))[j];
    }
}

float dot_f32(const float *a, const float *b, uint32_t n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}, sum;
    uint32_t i, lane;
    for (i = 0; i + 8 <= n; i += 8)
        for (lane = 0; lane < 8; ++lane) acc[lane] += a[i + lane] * b[i + lane];
    sum = reduce8(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
