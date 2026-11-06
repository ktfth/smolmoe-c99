#include "kernels.h"
#include "f16.h"
#include "util.h"

#include <math.h>

#if defined(SM_HAVE_AVX2)
#  include <immintrin.h>
#endif

DotQ8 dot_q8 = dot_q8_scalar, dot_q5 = dot_q5_scalar;
DotQ8x4 dot_q8_x4 = dot_q8_scalar_x4, dot_q5_x4 = dot_q5_scalar_x4;

static uint16_t rd_u16(const unsigned char *p) {
    return (uint16_t)(p[0] | p[1] << 8);
}

/* Lane reduction shared by every backend: ((l0+l4)+(l2+l6)) + ((l1+l5)+(l3+l7)). */
static float reduce8(const float a[8]) {
    float s0 = a[0] + a[4], s1 = a[1] + a[5], s2 = a[2] + a[6], s3 = a[3] + a[7];
    return (s0 + s2) + (s1 + s3);
}

/* Lane l of a block holds the exact integer sum of products 4l..4l+3,
 * matching the AVX2 maddubs/madd layout. */
static void accumulate_block(float acc[8], const int8_t *wq, const int8_t *xq, float d) {
    uint32_t lane, k;
    for (lane = 0; lane < 8; ++lane) {
        int32_t sum = 0;
        for (k = 0; k < 4; ++k) sum += (int32_t)wq[4 * lane + k] * (int32_t)xq[4 * lane + k];
        acc[lane] += d * (float)sum;
    }
}

/* Block body after the f16 scale: u32 high bits, then 16 bytes of low nibbles. */
static void unpack_q5(const unsigned char *body, int8_t out[QK]) {
    uint32_t high = rd_u32(body), j;
    const unsigned char *low = body + 4;
    for (j = 0; j < QK / 2; ++j) {
        out[j] = (int8_t)(((low[j] & 15) | ((high >> j) & 1) << 4) - 16);
        out[j + QK / 2] = (int8_t)(((low[j] >> 4) | ((high >> (j + QK / 2)) & 1) << 4) - 16);
    }
}

float dot_q8_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK)
        accumulate_block(acc, (const int8_t *)(w + 4), xq, rd_f32(w) * xd[b]);
    return reduce8(acc);
}

float dot_q5_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int8_t wq[QK];
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q5_BLOCK_BYTES, xq += QK) {
        unpack_q5(w + 2, wq);
        accumulate_block(acc, wq, xq, f16_to_f32(rd_u16(w)) * xd[b]);
    }
    return reduce8(acc);
}

void dot_q8_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]) {
    uint32_t t;
    for (t = 0; t < 4; ++t)
        out[t] = dot_q8_scalar(w, xd + (size_t)t * stride, xq + (size_t)t * stride * QK, nb);
}

void dot_q5_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]) {
    uint32_t t;
    for (t = 0; t < 4; ++t)
        out[t] = dot_q5_scalar(w, xd + (size_t)t * stride, xq + (size_t)t * stride * QK, nb);
}

#if defined(SM_HAVE_AVX2)
__attribute__((target("avx2,f16c")))
static float reduce_avx2(__m256 acc) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

/* Exact integer products of one block: |w| * sign(x, w) == w * x. */
#define BLOCK_PRODUCT(aw, wv, xv, ones) \
    _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_maddubs_epi16((aw), _mm256_sign_epi8((xv), (wv))), (ones)))

#define LOAD256(p) _mm256_loadu_si256((const __m256i *)(const void *)(p))

/* Block body -> 32 int8 in natural element order, minus 16. Port-5 friendly:
 * the 128-bit broadcast is a pure load and the lane split is a blend; the
 * high bits use one shuffle/compare (byte j becomes 0xFF iff bit j is set). */
__attribute__((target("avx2,f16c")))
static __m256i unpack_q5_avx2(const unsigned char *body) {
    const __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)(body + 4)));
    const __m256i low = _mm256_and_si256(_mm256_blend_epi32(v, _mm256_srli_epi16(v, 4), 0xF0), _mm256_set1_epi8(15));
    const __m256i spread = _mm256_shuffle_epi8(_mm256_set1_epi32((int)rd_u32(body)),
        _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0));
    const __m256i bits = _mm256_cmpeq_epi8(_mm256_or_si256(spread, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL)),
                                           _mm256_set1_epi8(-1));
    return _mm256_sub_epi8(_mm256_or_si256(low, _mm256_and_si256(bits, _mm256_set1_epi8(16))),
                           _mm256_set1_epi8(16));
}

/* Q5 rows are processed in chunks whose block scales are converted eight at a
 * time and pre-multiplied by the activation scales, so the block loop only
 * issues a broadcast load per block. Products are the same IEEE operations as
 * the scalar kernel (f16 -> f32 is exact). */
#define Q5_CHUNK 64u

__attribute__((target("avx2,f16c")))
static void q5_scales(const unsigned char *w, uint32_t n, float *out) {
    uint32_t j = 0, k;
    for (; j + 8 <= n; j += 8) {
        uint64_t lo = 0, hi = 0;
        for (k = 0; k < 4; ++k) {
            lo |= (uint64_t)rd_u16(w + (size_t)(j + k) * Q5_BLOCK_BYTES) << (16 * k);
            hi |= (uint64_t)rd_u16(w + (size_t)(j + 4 + k) * Q5_BLOCK_BYTES) << (16 * k);
        }
        _mm256_storeu_ps(out + j, _mm256_cvtph_ps(_mm_set_epi64x((long long)hi, (long long)lo)));
    }
    for (; j < n; ++j) out[j] = _cvtsh_ss(rd_u16(w + (size_t)j * Q5_BLOCK_BYTES));
}

__attribute__((target("avx2,f16c")))
static void scale_products(const float *ws, const float *xd, uint32_t n, float *out) {
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8) _mm256_storeu_ps(out + j, _mm256_mul_ps(_mm256_loadu_ps(ws + j), _mm256_loadu_ps(xd + j)));
    for (; j < n; ++j) out[j] = ws[j] * xd[j];
}

#define ACCUMULATE(acc, scale, aw, wv, xv, ones) \
    (acc) = _mm256_add_ps((acc), _mm256_mul_ps(_mm256_set1_ps(scale), BLOCK_PRODUCT((aw), (wv), (xv), (ones))))

#define ACCUMULATE_AT(acc, scale_ptr, aw, wv, xv, ones) \
    (acc) = _mm256_add_ps((acc), _mm256_mul_ps(_mm256_broadcast_ss(scale_ptr), BLOCK_PRODUCT((aw), (wv), (xv), (ones))))

__attribute__((target("avx2,f16c")))
float dot_q8_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK) {
        __m256i wv = LOAD256(w + 4);
        ACCUMULATE(acc, rd_f32(w) * xd[b], _mm256_sign_epi8(wv, wv), wv, LOAD256(xq), ones);
    }
    return reduce_avx2(acc);
}

__attribute__((target("avx2,f16c")))
float dot_q5_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    float ws[Q5_CHUNK], s[Q5_CHUNK];
    uint32_t b0, j;
    for (b0 = 0; b0 < nb; b0 += Q5_CHUNK) {
        uint32_t n = nb - b0 < Q5_CHUNK ? nb - b0 : Q5_CHUNK;
        q5_scales(w, n, ws);
        scale_products(ws, xd + b0, n, s);
        for (j = 0; j < n; ++j, w += Q5_BLOCK_BYTES, xq += QK) {
            __m256i wv = unpack_q5_avx2(w + 2);
            ACCUMULATE_AT(acc, s + j, _mm256_sign_epi8(wv, wv), wv, LOAD256(xq), ones);
        }
    }
    return reduce_avx2(acc);
}

/* Shared body of the 4-row kernels: one weight block, four activation rows. */
#define X4_STEP(ws, wv)                                                     \
    do {                                                                    \
        __m256i aw = _mm256_sign_epi8((wv), (wv));                          \
        ACCUMULATE(a0, (ws) * xd[0], aw, (wv), LOAD256(xq), ones);          \
        ACCUMULATE(a1, (ws) * xd[stride], aw, (wv), LOAD256(xq + qs), ones); \
        ACCUMULATE(a2, (ws) * xd[2 * stride], aw, (wv), LOAD256(xq + 2 * qs), ones); \
        ACCUMULATE(a3, (ws) * xd[3 * stride], aw, (wv), LOAD256(xq + 3 * qs), ones); \
    } while (0)

#define X4_STORE()               \
    out[0] = reduce_avx2(a0);    \
    out[1] = reduce_avx2(a1);    \
    out[2] = reduce_avx2(a2);    \
    out[3] = reduce_avx2(a3)

__attribute__((target("avx2,f16c")))
void dot_q8_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]) {
    const __m256i ones = _mm256_set1_epi16(1);
    const size_t qs = (size_t)stride * QK;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK, ++xd) {
        __m256i wv = LOAD256(w + 4);
        X4_STEP(rd_f32(w), wv);
    }
    X4_STORE();
}

__attribute__((target("avx2,f16c")))
void dot_q5_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]) {
    const __m256i ones = _mm256_set1_epi16(1);
    const size_t qs = (size_t)stride * QK;
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    float ws[Q5_CHUNK], s0[Q5_CHUNK], s1[Q5_CHUNK], s2[Q5_CHUNK], s3[Q5_CHUNK];
    uint32_t b0, j;
    for (b0 = 0; b0 < nb; b0 += Q5_CHUNK) {
        uint32_t n = nb - b0 < Q5_CHUNK ? nb - b0 : Q5_CHUNK;
        q5_scales(w, n, ws);
        scale_products(ws, xd + b0, n, s0);
        scale_products(ws, xd + b0 + stride, n, s1);
        scale_products(ws, xd + b0 + 2 * (size_t)stride, n, s2);
        scale_products(ws, xd + b0 + 3 * (size_t)stride, n, s3);
        for (j = 0; j < n; ++j, w += Q5_BLOCK_BYTES, xq += QK) {
            __m256i wv = unpack_q5_avx2(w + 2), aw = _mm256_sign_epi8(wv, wv);
            ACCUMULATE_AT(a0, s0 + j, aw, wv, LOAD256(xq), ones);
            ACCUMULATE_AT(a1, s1 + j, aw, wv, LOAD256(xq + qs), ones);
            ACCUMULATE_AT(a2, s2 + j, aw, wv, LOAD256(xq + 2 * qs), ones);
            ACCUMULATE_AT(a3, s3 + j, aw, wv, LOAD256(xq + 3 * qs), ones);
        }
    }
    X4_STORE();
}
#endif

const char *kernels_select(int force_scalar) {
    dot_q8 = dot_q8_scalar;
    dot_q8_x4 = dot_q8_scalar_x4;
    dot_q5 = dot_q5_scalar;
    dot_q5_x4 = dot_q5_scalar_x4;
#if defined(SM_HAVE_AVX2)
    /* F16C converts Q5 block scales exactly, matching f16_to_f32(). */
    if (!force_scalar && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        dot_q8 = dot_q8_avx2;
        dot_q8_x4 = dot_q8_avx2_x4;
        dot_q5 = dot_q5_avx2;
        dot_q5_x4 = dot_q5_avx2_x4;
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

void dequantize_q5_row(const unsigned char *w, uint32_t cols, float *out) {
    uint32_t b, j, nb = (cols + QK - 1) / QK;
    int8_t q[QK];
    for (b = 0; b < nb; ++b, w += Q5_BLOCK_BYTES) {
        float scale = f16_to_f32(rd_u16(w));
        uint32_t n = cols - b * QK < QK ? cols - b * QK : QK;
        unpack_q5(w + 2, q);
        for (j = 0; j < n; ++j) out[b * QK + j] = scale * (float)q[j];
    }
}

static int q5_level(float v, float inv) {
    long q = lrintf(v * inv);
    return q < -16 ? -16 : q > 15 ? 15 : (int)q;
}

/* Squared error of a block quantized with the (f16-rounded) scale d. */
static double q5_error(const float *x, uint32_t n, float d) {
    double err = 0;
    float inv = 1.0f / d;
    uint32_t j;
    for (j = 0; j < n; ++j) {
        double e = (double)x[j] - (double)q5_level(x[j], inv) * (double)d;
        err += e * e;
    }
    return err;
}

/* The signed extreme maps near -16 (the side with 16 levels); a small scan of
 * divisors around it trades clipping against resolution. */
void quantize_q5_row(const float *x, uint32_t cols, unsigned char *out) {
    uint32_t nb = (cols + QK - 1) / QK, b, j, k;
    for (b = 0; b < nb; ++b, x += QK, out += Q5_BLOCK_BYTES) {
        uint32_t n = cols - b * QK < QK ? cols - b * QK : QK;
        float vmax = 0, best_d = 0, inv;
        double best = -1;
        uint16_t half = 0;
        for (j = 0; j < n; ++j)
            if (fabsf(x[j]) > fabsf(vmax)) vmax = x[j];
        if (vmax != 0) {
            for (k = 0; k <= 20; ++k) {
                uint16_t h = f32_to_f16(vmax / -(14.5f + 0.1f * (float)k));
                float d = f16_to_f32(h);
                double err;
                if (d == 0 || !isfinite(d)) continue;
                err = q5_error(x, n, d);
                if (best < 0 || err < best) { best = err; best_d = d; half = h; }
            }
        }
        out[0] = (unsigned char)(half & 0xff);
        out[1] = (unsigned char)(half >> 8);
        inv = best_d != 0 ? 1.0f / best_d : 0.0f;
        {
            uint32_t high = 0;
            for (j = 0; j < QK; ++j) {
                uint32_t u = (uint32_t)((j < n && best_d != 0 ? q5_level(x[j], inv) : 0) + 16);
                high |= (u >> 4) << j;
                if (j < QK / 2) out[6 + j] = (unsigned char)(u & 15);
                else out[6 + j - QK / 2] |= (unsigned char)((u & 15) << 4);
            }
            out[2] = (unsigned char)high; out[3] = (unsigned char)(high >> 8);
            out[4] = (unsigned char)(high >> 16); out[5] = (unsigned char)(high >> 24);
        }
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
