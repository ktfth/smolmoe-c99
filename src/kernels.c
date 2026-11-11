#include "kernels.h"
#include "f16.h"
#include "util.h"
#include "vecmath.h"

#include <math.h>

#if defined(SM_HAVE_AVX2)
#  include <immintrin.h>
#endif
#if defined(SM_HAVE_NEON)
#  include <arm_neon.h>
#endif
#if defined(SM_HAVE_WASM_SIMD)
#  include <wasm_simd128.h>
#endif

DotQ8 dot_q8 = dot_q8_scalar, dot_q5 = dot_q5_scalar;
DotQ8x4 dot_q8_x4 = dot_q8_scalar_x4, dot_q5_x4 = dot_q5_scalar_x4;
QuantizeQ8 quantize_q8 = quantize_q8_scalar;
AttendF32 attend_f32 = attend_f32_scalar;
AttendF16 attend_f16 = attend_f16_scalar;
PackF16 pack_f16 = pack_f16_scalar;

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

#if defined(SM_HAVE_NEON)
/* The 8 lanes of the scalar kernel as two vectors: lanes 0-3 and 4-7. The
 * reduction is reduce8: (lo + hi) holds l0+l4 .. l3+l7, then (s0+s2) + (s1+s3). */
static float reduce_neon(float32x4_t lo, float32x4_t hi) {
    float32x4_t s = vaddq_f32(lo, hi);
    float32x2_t t = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
}

/* One block: lane l gets the exact integer sum of products 4l..4l+3. Pairwise
 * adds of the 16-bit products give pairs, then quads, in element order. */
static void block_sums(int8x16_t w0, int8x16_t w1, int8x16_t x0, int8x16_t x1, int32x4_t *lo, int32x4_t *hi) {
    *lo = vpaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(w0), vget_low_s8(x0))), vpaddlq_s16(vmull_high_s8(w0, x0)));
    *hi = vpaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(w1), vget_low_s8(x1))), vpaddlq_s16(vmull_high_s8(w1, x1)));
}

/* acc += d * (float)sum: the scalar kernel's two roundings (no fused multiply-add). */
static float32x4_t accumulate_neon(float32x4_t acc, float d, int32x4_t sums) {
    return vaddq_f32(acc, vmulq_f32(vdupq_n_f32(d), vcvtq_f32_s32(sums)));
}

/* Block body -> elements 0-15 (*e0) and 16-31 (*e1), minus 16. Byte k of the
 * u32 holds the high bits of elements 8k..8k+7; vtst turns each into 0xFF. */
static void unpack_q5_neon(const unsigned char *body, int8x16_t *e0, int8x16_t *e1) {
    static const uint8_t BIT[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    static const uint8_t PICK0[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    static const uint8_t PICK1[16] = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    const uint8x16_t low = vld1q_u8(body + 4), high = vreinterpretq_u8_u32(vdupq_n_u32(rd_u32(body)));
    const uint8x16_t bit = vld1q_u8(BIT), sixteen = vdupq_n_u8(16);
    const uint8x16_t h0 = vandq_u8(vtstq_u8(vqtbl1q_u8(high, vld1q_u8(PICK0)), bit), sixteen);
    const uint8x16_t h1 = vandq_u8(vtstq_u8(vqtbl1q_u8(high, vld1q_u8(PICK1)), bit), sixteen);
    *e0 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vandq_u8(low, vdupq_n_u8(15)), h0), sixteen));
    *e1 = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vshrq_n_u8(low, 4), h1), sixteen));
}

float dot_q8_neon(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    float32x4_t lo = vdupq_n_f32(0.0f), hi = lo;
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK) {
        const int8_t *wq = (const int8_t *)(w + 4);
        const float d = rd_f32(w) * xd[b];
        int32x4_t slo, shi;
        block_sums(vld1q_s8(wq), vld1q_s8(wq + 16), vld1q_s8(xq), vld1q_s8(xq + 16), &slo, &shi);
        lo = accumulate_neon(lo, d, slo);
        hi = accumulate_neon(hi, d, shi);
    }
    return reduce_neon(lo, hi);
}

float dot_q5_neon(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    float32x4_t lo = vdupq_n_f32(0.0f), hi = lo;
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q5_BLOCK_BYTES, xq += QK) {
        const float d = f16_to_f32(rd_u16(w)) * xd[b];
        int8x16_t w0, w1;
        int32x4_t slo, shi;
        unpack_q5_neon(w + 2, &w0, &w1);
        block_sums(w0, w1, vld1q_s8(xq), vld1q_s8(xq + 16), &slo, &shi);
        lo = accumulate_neon(lo, d, slo);
        hi = accumulate_neon(hi, d, shi);
    }
    return reduce_neon(lo, hi);
}

/* Shared body of the 4-row kernels: one weight block (scale ws), four activation rows. */
static void x4_block(float32x4_t lo[4], float32x4_t hi[4], float ws, int8x16_t w0, int8x16_t w1, const float *xd,
                     const int8_t *xq, uint32_t stride) {
    const size_t qs = (size_t)stride * QK;
    uint32_t t;
    for (t = 0; t < 4; ++t) {
        const int8_t *x = xq + t * qs;
        const float d = ws * xd[(size_t)t * stride];
        int32x4_t slo, shi;
        block_sums(w0, w1, vld1q_s8(x), vld1q_s8(x + 16), &slo, &shi);
        lo[t] = accumulate_neon(lo[t], d, slo);
        hi[t] = accumulate_neon(hi[t], d, shi);
    }
}

#define X4_NEON_INIT(lo, hi)                              \
    do {                                                  \
        uint32_t t_;                                      \
        for (t_ = 0; t_ < 4; ++t_) lo[t_] = hi[t_] = vdupq_n_f32(0.0f); \
    } while (0)

void dot_q8_neon_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]) {
    float32x4_t lo[4], hi[4];
    uint32_t b, t;
    X4_NEON_INIT(lo, hi);
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK, ++xd) {
        const int8_t *wq = (const int8_t *)(w + 4);
        x4_block(lo, hi, rd_f32(w), vld1q_s8(wq), vld1q_s8(wq + 16), xd, xq, stride);
    }
    for (t = 0; t < 4; ++t) out[t] = reduce_neon(lo[t], hi[t]);
}

void dot_q5_neon_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]) {
    float32x4_t lo[4], hi[4];
    uint32_t b, t;
    X4_NEON_INIT(lo, hi);
    for (b = 0; b < nb; ++b, w += Q5_BLOCK_BYTES, xq += QK, ++xd) {
        int8x16_t w0, w1;
        unpack_q5_neon(w + 2, &w0, &w1);
        x4_block(lo, hi, f16_to_f32(rd_u16(w)), w0, w1, xd, xq, stride);
    }
    for (t = 0; t < 4; ++t) out[t] = reduce_neon(lo[t], hi[t]);
}
#endif

#if defined(SM_HAVE_WASM_SIMD)
/* Same structure as the NEON kernels: the 8 lanes of the scalar kernel as two
 * vectors, lanes 0-3 and 4-7, reduced as reduce8: (lo + hi) holds l0+l4 ..
 * l3+l7, then (s0+s2) + (s1+s3). */
static float reduce_wasm(v128_t lo, v128_t hi) {
    const v128_t s = wasm_f32x4_add(lo, hi);
    const v128_t t = wasm_f32x4_add(s, wasm_i32x4_shuffle(s, s, 2, 3, 0, 1));
    return wasm_f32x4_extract_lane(t, 0) + wasm_f32x4_extract_lane(t, 1);
}

/* 16 weights (sign-extended to 16 bits: wl elements 0-7, wh 8-15) against 16
 * activations: lane l gets the exact integer sum of products 4l..4l+3. The
 * 16-bit dot gives pairs in element order; even plus odd pairs give quads. */
static v128_t quad_sums(v128_t wl, v128_t wh, v128_t x) {
    const v128_t lo = wasm_i32x4_dot_i16x8(wl, wasm_i16x8_extend_low_i8x16(x));
    const v128_t hi = wasm_i32x4_dot_i16x8(wh, wasm_i16x8_extend_high_i8x16(x));
    return wasm_i32x4_add(wasm_i32x4_shuffle(lo, hi, 0, 2, 4, 6), wasm_i32x4_shuffle(lo, hi, 1, 3, 5, 7));
}

/* acc += d * (float)sum: the scalar kernel's two roundings (SIMD128 has no
 * fused multiply-add). */
static v128_t accumulate_wasm(v128_t acc, float d, v128_t sums) {
    return wasm_f32x4_add(acc, wasm_f32x4_mul(wasm_f32x4_splat(d), wasm_f32x4_convert_i32x4(sums)));
}

/* One weight block, sign-extended once: elements 0-7, 8-15, 16-23, 24-31. */
typedef struct WideBlock {
    v128_t w[4];
} WideBlock;

static WideBlock widen(v128_t w0, v128_t w1) {
    WideBlock b;
    b.w[0] = wasm_i16x8_extend_low_i8x16(w0);
    b.w[1] = wasm_i16x8_extend_high_i8x16(w0);
    b.w[2] = wasm_i16x8_extend_low_i8x16(w1);
    b.w[3] = wasm_i16x8_extend_high_i8x16(w1);
    return b;
}

/* lo/hi += d * the block's lane sums against activations x[0..32). */
static void block_wasm(v128_t *lo, v128_t *hi, float d, const WideBlock *w, const int8_t *x) {
    *lo = accumulate_wasm(*lo, d, quad_sums(w->w[0], w->w[1], wasm_v128_load(x)));
    *hi = accumulate_wasm(*hi, d, quad_sums(w->w[2], w->w[3], wasm_v128_load(x + 16)));
}

/* Block body -> elements 0-15 (*e0) and 16-31 (*e1), minus 16. Byte k of the
 * u32 holds the high bits of elements 8k..8k+7; a byte compare against its
 * bit turns each into 0xFF. */
static void unpack_q5_wasm(const unsigned char *body, v128_t *e0, v128_t *e1) {
    const v128_t low = wasm_v128_load(body + 4), high = wasm_i32x4_splat((int32_t)rd_u32(body));
    const v128_t bit = wasm_u8x16_const(1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
    const v128_t sixteen = wasm_u8x16_splat(16);
    const v128_t p0 = wasm_i8x16_shuffle(high, high, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const v128_t p1 = wasm_i8x16_shuffle(high, high, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const v128_t h0 = wasm_v128_and(wasm_i8x16_eq(wasm_v128_and(p0, bit), bit), sixteen);
    const v128_t h1 = wasm_v128_and(wasm_i8x16_eq(wasm_v128_and(p1, bit), bit), sixteen);
    *e0 = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(low, wasm_u8x16_splat(15)), h0), sixteen);
    *e1 = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(low, 4), h1), sixteen);
}

static WideBlock q8_block(const unsigned char *w) {
    return widen(wasm_v128_load(w + 4), wasm_v128_load(w + 20));
}

static WideBlock q5_block(const unsigned char *w) {
    v128_t e0, e1;
    unpack_q5_wasm(w + 2, &e0, &e1);
    return widen(e0, e1);
}

float dot_q8_wasm(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    v128_t lo = wasm_f32x4_splat(0.0f), hi = lo;
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK) {
        const WideBlock wb = q8_block(w);
        block_wasm(&lo, &hi, rd_f32(w) * xd[b], &wb, xq);
    }
    return reduce_wasm(lo, hi);
}

float dot_q5_wasm(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    v128_t lo = wasm_f32x4_splat(0.0f), hi = lo;
    uint32_t b;
    for (b = 0; b < nb; ++b, w += Q5_BLOCK_BYTES, xq += QK) {
        const WideBlock wb = q5_block(w);
        block_wasm(&lo, &hi, f16_to_f32(rd_u16(w)) * xd[b], &wb, xq);
    }
    return reduce_wasm(lo, hi);
}

/* Shared body of the 4-row kernels: one weight block (scale ws), widened
 * once, against four activation rows. */
static void x4_block_wasm(v128_t lo[4], v128_t hi[4], float ws, const WideBlock *wb, const float *xd,
                          const int8_t *xq, uint32_t stride) {
    const size_t qs = (size_t)stride * QK;
    uint32_t t;
    for (t = 0; t < 4; ++t) block_wasm(&lo[t], &hi[t], ws * xd[(size_t)t * stride], wb, xq + t * qs);
}

void dot_q8_wasm_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]) {
    v128_t lo[4], hi[4];
    uint32_t b, t;
    for (t = 0; t < 4; ++t) lo[t] = hi[t] = wasm_f32x4_splat(0.0f);
    for (b = 0; b < nb; ++b, w += Q8_BLOCK_BYTES, xq += QK, ++xd) {
        const WideBlock wb = q8_block(w);
        x4_block_wasm(lo, hi, rd_f32(w), &wb, xd, xq, stride);
    }
    for (t = 0; t < 4; ++t) out[t] = reduce_wasm(lo[t], hi[t]);
}

void dot_q5_wasm_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]) {
    v128_t lo[4], hi[4];
    uint32_t b, t;
    for (t = 0; t < 4; ++t) lo[t] = hi[t] = wasm_f32x4_splat(0.0f);
    for (b = 0; b < nb; ++b, w += Q5_BLOCK_BYTES, xq += QK, ++xd) {
        const WideBlock wb = q5_block(w);
        x4_block_wasm(lo, hi, f16_to_f32(rd_u16(w)), &wb, xd, xq, stride);
    }
    for (t = 0; t < 4; ++t) out[t] = reduce_wasm(lo[t], hi[t]);
}
#endif

#if defined(SM_HAVE_AVX2)
__attribute__((target("avx2,f16c")))
static void attend_f32_avx2(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out);
__attribute__((target("avx2,f16c")))
static void attend_f16_avx2(const float *q, const uint16_t *keys, const uint16_t *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out);
__attribute__((target("avx2,f16c")))
static void pack_f16_avx2(const float *src, uint16_t *dst, uint32_t n);
#elif defined(SM_HAVE_NEON)
static void attend_f32_neon(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out);
static void attend_f16_neon(const float *q, const uint16_t *keys, const uint16_t *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out);
#elif defined(SM_HAVE_WASM_SIMD)
static void attend_f32_wasm(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out);
#endif

const char *kernels_select(int force_scalar) {
    dot_q8 = dot_q8_scalar;
    dot_q8_x4 = dot_q8_scalar_x4;
    dot_q5 = dot_q5_scalar;
    dot_q5_x4 = dot_q5_scalar_x4;
    quantize_q8 = quantize_q8_scalar;
    attend_f32 = attend_f32_scalar;
    attend_f16 = attend_f16_scalar;
    pack_f16 = pack_f16_scalar;
    vecmath_select(0);
#if defined(SM_HAVE_AVX2)
    /* F16C converts Q5 block scales exactly, matching f16_to_f32(). */
    if (!force_scalar && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        dot_q8 = dot_q8_avx2;
        dot_q8_x4 = dot_q8_avx2_x4;
        dot_q5 = dot_q5_avx2;
        dot_q5_x4 = dot_q5_avx2_x4;
        quantize_q8 = quantize_q8_avx2;
        attend_f32 = attend_f32_avx2;
        attend_f16 = attend_f16_avx2;
        pack_f16 = pack_f16_avx2;
        vecmath_select(1);
        return "avx2";
    }
#elif defined(SM_HAVE_NEON)
    if (!force_scalar) {
        dot_q8 = dot_q8_neon;
        dot_q8_x4 = dot_q8_neon_x4;
        dot_q5 = dot_q5_neon;
        dot_q5_x4 = dot_q5_neon_x4;
        quantize_q8 = quantize_q8_neon;
        attend_f32 = attend_f32_neon;
        attend_f16 = attend_f16_neon;
        vecmath_select(1);
        return "neon";
    }
#elif defined(SM_HAVE_WASM_SIMD)
    if (!force_scalar) {
        dot_q8 = dot_q8_wasm;
        dot_q8_x4 = dot_q8_wasm_x4;
        dot_q5 = dot_q5_wasm;
        dot_q5_x4 = dot_q5_wasm_x4;
        quantize_q8 = quantize_q8_wasm;
        attend_f32 = attend_f32_wasm;
        vecmath_select(1);
        return "simd128";
    }
#else
    (void)force_scalar;
#endif
    return "scalar";
}

/* 1/scale stays finite, so x * inv never leaves [-127, 127] (and both
 * versions round the same values). */
static float inverse_scale(float scale) { return scale >= FLT_MIN ? 1.0f / scale : 0.0f; }

int quantize_q8_scalar(const float *x, uint32_t n, float *d, int8_t *q) {
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
        inv = inverse_scale(scale);
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

#if defined(SM_HAVE_AVX2)
__attribute__((target("avx2,f16c")))
static float max_avx2(__m256 v) {
    __m128 m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    return _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
}

#define TO_INT(v, inv) _mm256_cvtps_epi32(_mm256_mul_ps((v), (inv))) /* rounds to nearest even, as lrintf */
#define FINITE(a, big) _mm256_cmp_ps((a), (big), _CMP_LE_OQ)        /* false for inf and NaN */

/* quantize_q8_scalar() 32 values at a time; a partial last block goes to it. */
__attribute__((target("avx2,f16c")))
int quantize_q8_avx2(const float *x, uint32_t n, float *d, int8_t *q) {
    const __m256 sign = _mm256_set1_ps(-0.0f), big = _mm256_set1_ps(FLT_MAX);
    const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7); /* undoes the lane interleave of packs */
    uint32_t b, full = n / QK;
    for (b = 0; b < full; ++b, x += QK, q += QK) {
        __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
        __m256 v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
        __m256 a0 = _mm256_andnot_ps(sign, v0), a1 = _mm256_andnot_ps(sign, v1);
        __m256 a2 = _mm256_andnot_ps(sign, v2), a3 = _mm256_andnot_ps(sign, v3), inv;
        __m256i i01, i23;
        float scale;
        if (_mm256_movemask_ps(_mm256_and_ps(_mm256_and_ps(FINITE(a0, big), FINITE(a1, big)),
                                             _mm256_and_ps(FINITE(a2, big), FINITE(a3, big)))) != 0xFF)
            return -1;
        scale = max_avx2(_mm256_max_ps(_mm256_max_ps(a0, a1), _mm256_max_ps(a2, a3))) / 127.0f;
        d[b] = scale;
        inv = _mm256_set1_ps(inverse_scale(scale));
        i01 = _mm256_packs_epi32(TO_INT(v0, inv), TO_INT(v1, inv));
        i23 = _mm256_packs_epi32(TO_INT(v2, inv), TO_INT(v3, inv));
        _mm256_storeu_si256((__m256i *)(void *)q, _mm256_permutevar8x32_epi32(_mm256_packs_epi16(i01, i23), order));
    }
    return full * QK < n ? quantize_q8_scalar(x, n - full * QK, d + full, q) : 0;
}
#endif

#if defined(SM_HAVE_NEON)
/* quantize_q8_scalar() 32 values at a time; a partial last block goes to it.
 * vcvtnq rounds to nearest even, as lrintf; |x * inv| stays within 127, so the
 * saturating narrows never clip. */
int quantize_q8_neon(const float *x, uint32_t n, float *d, int8_t *q) {
    const float32x4_t big = vdupq_n_f32(FLT_MAX);
    uint32_t b, k, full = n / QK;
    for (b = 0; b < full; ++b, x += QK, q += QK) {
        float32x4_t v[8], amax = vdupq_n_f32(0.0f), inv;
        uint32x4_t ok_mask = vdupq_n_u32(0xFFFFFFFFu);
        int16x8_t h[4];
        float scale;
        for (k = 0; k < 8; ++k) {
            float32x4_t a;
            v[k] = vld1q_f32(x + 4 * k);
            a = vabsq_f32(v[k]);
            ok_mask = vandq_u32(ok_mask, vcleq_f32(a, big)); /* false for inf and NaN */
            amax = vmaxq_f32(amax, a);
        }
        if (vminvq_u32(ok_mask) == 0) return -1;
        scale = vmaxvq_f32(amax) / 127.0f;
        d[b] = scale;
        inv = vdupq_n_f32(inverse_scale(scale));
        for (k = 0; k < 4; ++k)
            h[k] = vcombine_s16(vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(v[2 * k], inv))),
                                vqmovn_s32(vcvtnq_s32_f32(vmulq_f32(v[2 * k + 1], inv))));
        vst1q_s8(q, vcombine_s8(vqmovn_s16(h[0]), vqmovn_s16(h[1])));
        vst1q_s8(q + 16, vcombine_s8(vqmovn_s16(h[2]), vqmovn_s16(h[3])));
    }
    return full * QK < n ? quantize_q8_scalar(x, n - full * QK, d + full, q) : 0;
}
#endif

#if defined(SM_HAVE_WASM_SIMD)
/* Rounds to nearest even, as lrintf; the value is already integral when truncated. */
static v128_t to_int_wasm(v128_t v, v128_t inv) {
    return wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(wasm_f32x4_mul(v, inv)));
}

/* quantize_q8_scalar() 32 values at a time; a partial last block goes to it.
 * pmax(m, a) is "m < a ? a : m", the scalar update; |x * inv| stays within
 * 127, so the saturating narrows never clip. */
int quantize_q8_wasm(const float *x, uint32_t n, float *d, int8_t *q) {
    const v128_t big = wasm_f32x4_splat(FLT_MAX);
    uint32_t b, k, full = n / QK;
    for (b = 0; b < full; ++b, x += QK, q += QK) {
        v128_t v[8], h[4], amax = wasm_f32x4_splat(0.0f), ok = wasm_i32x4_splat(-1), inv;
        float scale;
        for (k = 0; k < 8; ++k) {
            v128_t a;
            v[k] = wasm_v128_load(x + 4 * k);
            a = wasm_f32x4_abs(v[k]);
            ok = wasm_v128_and(ok, wasm_f32x4_le(a, big)); /* false for inf and NaN */
            amax = wasm_f32x4_pmax(amax, a);
        }
        if (!wasm_i32x4_all_true(ok)) return -1;
        amax = wasm_f32x4_pmax(amax, wasm_i32x4_shuffle(amax, amax, 2, 3, 0, 1));
        amax = wasm_f32x4_pmax(amax, wasm_i32x4_shuffle(amax, amax, 1, 0, 3, 2));
        scale = wasm_f32x4_extract_lane(amax, 0) / 127.0f;
        d[b] = scale;
        inv = wasm_f32x4_splat(inverse_scale(scale));
        for (k = 0; k < 4; ++k) h[k] = wasm_i16x8_narrow_i32x4(to_int_wasm(v[2 * k], inv), to_int_wasm(v[2 * k + 1], inv));
        wasm_v128_store(q, wasm_i8x16_narrow_i16x8(h[0], h[1]));
        wasm_v128_store(q + 16, wasm_i8x16_narrow_i16x8(h[2], h[3]));
    }
    return full * QK < n ? quantize_q8_scalar(x, n - full * QK, d + full, q) : 0;
}
#endif

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

static float dot_f16_scalar(const float *a, const uint16_t *b, uint32_t n) {
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}, sum;
    uint32_t i, lane;
    for (i = 0; i + 8 <= n; i += 8)
        for (lane = 0; lane < 8; ++lane) acc[lane] += a[i + lane] * f16_to_f32(b[i + lane]);
    sum = reduce8(acc);
    for (; i < n; ++i) sum += a[i] * f16_to_f32(b[i]);
    return sum;
}

static void saxpy_f32_scalar(float *out, float w, const float *v, uint32_t n) {
    uint32_t j;
    for (j = 0; j < n; ++j) out[j] += w * v[j];
}

static void saxpy_f16_scalar(float *out, float w, const uint16_t *v, uint32_t n) {
    uint32_t j;
    for (j = 0; j < n; ++j) out[j] += w * f16_to_f32(v[j]);
}

void pack_f16_scalar(const float *src, uint16_t *dst, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}

/* Raw scores are already in scores[start..pos]. Same max, exp and sum as a
 * fused loop: the max does not change the stored values it is taken from. */
static float softmax_inv(float *scores, uint32_t start, uint32_t pos) {
    uint32_t p;
    float high = -FLT_MAX, total = 0;
    for (p = start; p <= pos; ++p)
        if (scores[p] > high) high = scores[p];
    for (p = start; p <= pos; ++p) {
        scores[p] = exp_det(scores[p] - high);
        total += scores[p];
    }
    return 1.0f / total;
}

void attend_f32_scalar(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                       uint32_t start, uint32_t pos, float *scores, float *out) {
    uint32_t p;
    float factor = 1.0f / sqrtf((float)hd), inv;
    for (p = start; p <= pos; ++p) scores[p] = dot_f32(q, keys + (size_t)p * stride, hd) * factor;
    inv = softmax_inv(scores, start, pos);
    memset(out, 0, hd * sizeof(float));
    for (p = start; p <= pos; ++p) saxpy_f32_scalar(out, scores[p] * inv, values + (size_t)p * stride, hd);
}

void attend_f16_scalar(const float *q, const uint16_t *keys, const uint16_t *values, uint32_t stride, uint32_t hd,
                       uint32_t start, uint32_t pos, float *scores, float *out) {
    uint32_t p;
    float factor = 1.0f / sqrtf((float)hd), inv;
    for (p = start; p <= pos; ++p) scores[p] = dot_f16_scalar(q, keys + (size_t)p * stride, hd) * factor;
    inv = softmax_inv(scores, start, pos);
    memset(out, 0, hd * sizeof(float));
    for (p = start; p <= pos; ++p) saxpy_f16_scalar(out, scores[p] * inv, values + (size_t)p * stride, hd);
}

#if defined(SM_HAVE_AVX2)
__attribute__((target("avx2,f16c")))
static float reduce256(__m256 v) {
    float a[8];
    _mm256_storeu_ps(a, v);
    return reduce8(a);
}

__attribute__((target("avx2,f16c")))
static float dot_f32_avx2(const float *a, const float *b, uint32_t n) {
    __m256 acc = _mm256_setzero_ps();
    float sum;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    sum = reduce256(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

__attribute__((target("avx2,f16c")))
static float dot_f16_avx2(const float *a, const uint16_t *b, uint32_t n) {
    __m256 acc = _mm256_setzero_ps();
    float sum;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 bf = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(b + i)));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i), bf));
    }
    sum = reduce256(acc);
    for (; i < n; ++i) sum += a[i] * f16_to_f32(b[i]);
    return sum;
}

__attribute__((target("avx2,f16c")))
static void saxpy_f32_avx2(float *out, float w, const float *v, uint32_t n) {
    __m256 ww = _mm256_set1_ps(w);
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8)
        _mm256_storeu_ps(out + j, _mm256_add_ps(_mm256_loadu_ps(out + j), _mm256_mul_ps(ww, _mm256_loadu_ps(v + j))));
    for (; j < n; ++j) out[j] += w * v[j];
}

__attribute__((target("avx2,f16c")))
static void saxpy_f16_avx2(float *out, float w, const uint16_t *v, uint32_t n) {
    __m256 ww = _mm256_set1_ps(w);
    uint32_t j = 0;
    for (; j + 8 <= n; j += 8) {
        __m256 vf = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(const void *)(v + j)));
        _mm256_storeu_ps(out + j, _mm256_add_ps(_mm256_loadu_ps(out + j), _mm256_mul_ps(ww, vf)));
    }
    for (; j < n; ++j) out[j] += w * f16_to_f32(v[j]);
}

__attribute__((target("avx2,f16c")))
static void pack_f16_avx2(const float *src, uint16_t *dst, uint32_t n) {
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(void *)(dst + i), h);
    }
    for (; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}

__attribute__((target("avx2,f16c")))
static void attend_f32_avx2(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out) {
    uint32_t p;
    float factor = 1.0f / sqrtf((float)hd), inv;
    for (p = start; p <= pos; ++p) scores[p] = dot_f32_avx2(q, keys + (size_t)p * stride, hd) * factor;
    inv = softmax_inv(scores, start, pos);
    memset(out, 0, hd * sizeof(float));
    for (p = start; p <= pos; ++p) saxpy_f32_avx2(out, scores[p] * inv, values + (size_t)p * stride, hd);
}

__attribute__((target("avx2,f16c")))
static void attend_f16_avx2(const float *q, const uint16_t *keys, const uint16_t *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out) {
    uint32_t p;
    float factor = 1.0f / sqrtf((float)hd), inv;
    for (p = start; p <= pos; ++p) scores[p] = dot_f16_avx2(q, keys + (size_t)p * stride, hd) * factor;
    inv = softmax_inv(scores, start, pos);
    memset(out, 0, hd * sizeof(float));
    for (p = start; p <= pos; ++p) saxpy_f16_avx2(out, scores[p] * inv, values + (size_t)p * stride, hd);
}
#elif defined(SM_HAVE_NEON)
/* reduce_neon() above is the same 8-lane order as reduce8. */
static float dot_f32_neon(const float *a, const float *b, uint32_t n) {
    float32x4_t lo = vdupq_n_f32(0), hi = vdupq_n_f32(0);
    float sum;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        lo = vaddq_f32(lo, vmulq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        hi = vaddq_f32(hi, vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4)));
    }
    sum = reduce_neon(lo, hi);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

/* f16 -> f32 is the scalar converter (exact); the eight products then follow
 * the same lanes as dot_f32. NEON fp16 arithmetic is not required. */
static float dot_f16_neon(const float *a, const uint16_t *b, uint32_t n) {
    float32x4_t lo = vdupq_n_f32(0), hi = vdupq_n_f32(0);
    float sum;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float bf[8];
        uint32_t k;
        for (k = 0; k < 8; ++k) bf[k] = f16_to_f32(b[i + k]);
        lo = vaddq_f32(lo, vmulq_f32(vld1q_f32(a + i), vld1q_f32(bf)));
        hi = vaddq_f32(hi, vmulq_f32(vld1q_f32(a + i + 4), vld1q_f32(bf + 4)));
    }
    sum = reduce_neon(lo, hi);
    for (; i < n; ++i) sum += a[i] * f16_to_f32(b[i]);
    return sum;
}

static void saxpy_f32_neon(float *out, float w, const float *v, uint32_t n) {
    float32x4_t ww = vdupq_n_f32(w);
    uint32_t j = 0;
    for (; j + 4 <= n; j += 4)
        vst1q_f32(out + j, vaddq_f32(vld1q_f32(out + j), vmulq_f32(ww, vld1q_f32(v + j))));
    for (; j < n; ++j) out[j] += w * v[j];
}

static void saxpy_f16_neon(float *out, float w, const uint16_t *v, uint32_t n) {
    float32x4_t ww = vdupq_n_f32(w);
    uint32_t j = 0;
    for (; j + 4 <= n; j += 4) {
        float bf[4];
        bf[0] = f16_to_f32(v[j]);
        bf[1] = f16_to_f32(v[j + 1]);
        bf[2] = f16_to_f32(v[j + 2]);
        bf[3] = f16_to_f32(v[j + 3]);
        vst1q_f32(out + j, vaddq_f32(vld1q_f32(out + j), vmulq_f32(ww, vld1q_f32(bf))));
    }
    for (; j < n; ++j) out[j] += w * f16_to_f32(v[j]);
}

static void attend_f32_neon(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out) {
    uint32_t p;
    float factor = 1.0f / sqrtf((float)hd), inv;
    for (p = start; p <= pos; ++p) scores[p] = dot_f32_neon(q, keys + (size_t)p * stride, hd) * factor;
    inv = softmax_inv(scores, start, pos);
    memset(out, 0, hd * sizeof(float));
    for (p = start; p <= pos; ++p) saxpy_f32_neon(out, scores[p] * inv, values + (size_t)p * stride, hd);
}

static void attend_f16_neon(const float *q, const uint16_t *keys, const uint16_t *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out) {
    uint32_t p;
    float factor = 1.0f / sqrtf((float)hd), inv;
    for (p = start; p <= pos; ++p) scores[p] = dot_f16_neon(q, keys + (size_t)p * stride, hd) * factor;
    inv = softmax_inv(scores, start, pos);
    memset(out, 0, hd * sizeof(float));
    for (p = start; p <= pos; ++p) saxpy_f16_neon(out, scores[p] * inv, values + (size_t)p * stride, hd);
}
#elif defined(SM_HAVE_WASM_SIMD)
/* reduce_wasm() above is the same 8-lane order as reduce8. */
static float dot_f32_wasm(const float *a, const float *b, uint32_t n) {
    v128_t lo = wasm_f32x4_splat(0), hi = wasm_f32x4_splat(0);
    float sum;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        lo = wasm_f32x4_add(lo, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
        hi = wasm_f32x4_add(hi, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
    }
    sum = reduce_wasm(lo, hi);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

static void saxpy_f32_wasm(float *out, float w, const float *v, uint32_t n) {
    v128_t ww = wasm_f32x4_splat(w);
    uint32_t j = 0;
    for (; j + 4 <= n; j += 4)
        wasm_v128_store(out + j, wasm_f32x4_add(wasm_v128_load(out + j), wasm_f32x4_mul(ww, wasm_v128_load(v + j))));
    for (; j < n; ++j) out[j] += w * v[j];
}

static void attend_f32_wasm(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                            uint32_t start, uint32_t pos, float *scores, float *out) {
    uint32_t p;
    float factor = 1.0f / sqrtf((float)hd), inv;
    for (p = start; p <= pos; ++p) scores[p] = dot_f32_wasm(q, keys + (size_t)p * stride, hd) * factor;
    inv = softmax_inv(scores, start, pos);
    memset(out, 0, hd * sizeof(float));
    for (p = start; p <= pos; ++p) saxpy_f32_wasm(out, scores[p] * inv, values + (size_t)p * stride, hd);
}
#endif
