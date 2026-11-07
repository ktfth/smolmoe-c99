#include "vecmath.h"

#include <stdint.h>
#include <string.h>

#if defined(SM_HAVE_AVX2)
#  include <immintrin.h>
#endif
#if defined(SM_HAVE_NEON)
#  include <arm_neon.h>
#endif

SiluMul silu_mul = silu_mul_scalar;

/* e^x = 2^(k/32) * 2^(r/32) with z = 32x/ln2, k = round(z), r = z - k in
 * [-0.5, 0.5]. Everything runs in binary64 and is rounded to binary32 once:
 * the polynomial's error (about 1e-12) only matters for the rare x whose e^x
 * lies that close to a rounding boundary. Same method as ARM's
 * optimized-routines expf; the table comes from scripts/gen-exp-table.ts. */

/* Inputs are clamped so that k stays in range; the clamps already saturate the
 * result to +inf and 0. */
#define EXP_HI 89.0f
#define EXP_LO -104.0f
#define INV_LN2_N 0x1.71547652b82fep+5 /* 32 / ln 2 */
#define SHIFT 0x1.8p+52                /* z + SHIFT rounds z to an integer held in the low bits */
#define EXP_A (0x1.62e42fefa39efp-1 / 32) /* ln 2 / 32: 2^(r/32) = e^(EXP_A r) */

/* Taylor coefficients of e^(EXP_A r) up to degree 4. */
static const double C1 = EXP_A, C2 = EXP_A * EXP_A / 2, C3 = EXP_A * EXP_A * EXP_A / 6,
                    C4 = EXP_A * EXP_A * EXP_A * EXP_A / 24;

/* Bits of 2^(j/32) minus j << 47: adding k << 47 gives the bits of 2^(k/32). */
static const uint64_t EXP2_TABLE[32] = {
    0x3ff0000000000000u, 0x3fefd9b0d3158574u, 0x3fefb5586cf9890fu, 0x3fef9301d0125b51u,
    0x3fef72b83c7d517bu, 0x3fef54873168b9aau, 0x3fef387a6e756238u, 0x3fef1e9df51fdee1u,
    0x3fef06fe0a31b715u, 0x3feef1a7373aa9cbu, 0x3feedea64c123422u, 0x3feece086061892du,
    0x3feebfdad5362a27u, 0x3feeb42b569d4f82u, 0x3feeab07dd485429u, 0x3feea47eb03a5585u,
    0x3feea09e667f3bcdu, 0x3fee9f75e8ec5f74u, 0x3feea11473eb0187u, 0x3feea589994cce13u,
    0x3feeace5422aa0dbu, 0x3feeb737b0cdc5e5u, 0x3feec49182a3f090u, 0x3feed503b23e255du,
    0x3feee89f995ad3adu, 0x3feeff76f2fb5e47u, 0x3fef199bdd85529cu, 0x3fef3720dcef9069u,
    0x3fef5818dcfba487u, 0x3fef7c97337b9b5fu, 0x3fefa4afa2a490dau, 0x3fefd0765b6e4540u,
};

static uint64_t bits_of(double d) {
    uint64_t u;
    memcpy(&u, &d, sizeof(u));
    return u;
}

static double double_of(uint64_t u) {
    double d;
    memcpy(&d, &u, sizeof(d));
    return d;
}

float exp_det(float x) {
    double z, kd, r, p;
    uint64_t ki;
    if (x != x) return x;
    x = x > EXP_HI ? EXP_HI : x < EXP_LO ? EXP_LO : x;
    z = INV_LN2_N * (double)x;
    kd = z + SHIFT;
    ki = bits_of(kd);
    kd = kd - SHIFT;
    r = z - kd;
    p = C4 * r + C3;
    p = p * r + C2;
    p = p * r + C1;
    p = p * r + 1.0;
    return (float)(p * double_of(EXP2_TABLE[ki & 31u] + (ki << 47)));
}

void silu_mul_scalar(float *up, const float *gate, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) up[i] *= gate[i] / (1.0f + exp_det(-gate[i]));
}

#if defined(SM_HAVE_AVX2)
/* exp_det() on 4 clamped, finite lanes, operation for operation, before the
 * final rounding to binary32. */
__attribute__((target("avx2,f16c")))
static __m256d exp_pd(__m256d x) {
    const __m256d shift = _mm256_set1_pd(SHIFT);
    __m256d z = _mm256_mul_pd(_mm256_set1_pd(INV_LN2_N), x), kd = _mm256_add_pd(z, shift), r, p;
    __m256i ki = _mm256_castpd_si256(kd), t;
    kd = _mm256_sub_pd(kd, shift);
    r = _mm256_sub_pd(z, kd);
    p = _mm256_add_pd(_mm256_mul_pd(_mm256_set1_pd(C4), r), _mm256_set1_pd(C3));
    p = _mm256_add_pd(_mm256_mul_pd(p, r), _mm256_set1_pd(C2));
    p = _mm256_add_pd(_mm256_mul_pd(p, r), _mm256_set1_pd(C1));
    p = _mm256_add_pd(_mm256_mul_pd(p, r), _mm256_set1_pd(1.0));
    t = _mm256_i64gather_epi64((const long long *)(const void *)EXP2_TABLE,
                               _mm256_and_si256(ki, _mm256_set1_epi64x(31)), 8);
    return _mm256_mul_pd(p, _mm256_castsi256_pd(_mm256_add_epi64(t, _mm256_slli_epi64(ki, 47))));
}

__attribute__((target("avx2,f16c")))
static __m256 exp_avx2(__m256 x) {
    const __m256 nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    const __m256 c = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(EXP_LO)), _mm256_set1_ps(EXP_HI));
    const __m128 lo = _mm256_cvtpd_ps(exp_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(c))));
    const __m128 hi = _mm256_cvtpd_ps(exp_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(c, 1))));
    return _mm256_blendv_ps(_mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1), x, nan);
}

__attribute__((target("avx2,f16c")))
void silu_mul_avx2(float *up, const float *gate, size_t n) {
    const __m256 sign = _mm256_set1_ps(-0.0f), one = _mm256_set1_ps(1.0f);
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 s = _mm256_div_ps(g, _mm256_add_ps(one, exp_avx2(_mm256_xor_ps(g, sign))));
        _mm256_storeu_ps(up + i, _mm256_mul_ps(_mm256_loadu_ps(up + i), s));
    }
    silu_mul_scalar(up + i, gate + i, n - i);
}
#endif

#if defined(SM_HAVE_NEON)
/* exp_det() on 2 clamped, finite lanes, operation for operation, before the
 * final rounding to binary32. NEON has no gather: the table is read per lane. */
static float64x2_t exp_f64x2(float64x2_t x) {
    const float64x2_t shift = vdupq_n_f64(SHIFT);
    float64x2_t z = vmulq_f64(vdupq_n_f64(INV_LN2_N), x), kd = vaddq_f64(z, shift), r, p;
    const uint64x2_t ki = vreinterpretq_u64_f64(kd);
    const uint64_t k0 = vgetq_lane_u64(ki, 0), k1 = vgetq_lane_u64(ki, 1);
    kd = vsubq_f64(kd, shift);
    r = vsubq_f64(z, kd);
    p = vaddq_f64(vmulq_f64(vdupq_n_f64(C4), r), vdupq_n_f64(C3));
    p = vaddq_f64(vmulq_f64(p, r), vdupq_n_f64(C2));
    p = vaddq_f64(vmulq_f64(p, r), vdupq_n_f64(C1));
    p = vaddq_f64(vmulq_f64(p, r), vdupq_n_f64(1.0));
    return vmulq_f64(p, vreinterpretq_f64_u64(vcombine_u64(vcreate_u64(EXP2_TABLE[k0 & 31u] + (k0 << 47)),
                                                             vcreate_u64(EXP2_TABLE[k1 & 31u] + (k1 << 47)))));
}

static float32x4_t exp_neon(float32x4_t x) {
    const float32x4_t c = vminq_f32(vmaxq_f32(x, vdupq_n_f32(EXP_LO)), vdupq_n_f32(EXP_HI));
    const float32x2_t lo = vcvt_f32_f64(exp_f64x2(vcvt_f64_f32(vget_low_f32(c))));
    const float32x4_t e = vcvt_high_f32_f64(lo, exp_f64x2(vcvt_high_f64_f32(c)));
    return vbslq_f32(vceqq_f32(x, x), e, x); /* a NaN lane stays NaN, as in exp_det */
}

void silu_mul_neon(float *up, const float *gate, size_t n) {
    const float32x4_t one = vdupq_n_f32(1.0f);
    size_t i;
    for (i = 0; i + 4 <= n; i += 4) {
        const float32x4_t g = vld1q_f32(gate + i);
        const float32x4_t s = vdivq_f32(g, vaddq_f32(one, exp_neon(vnegq_f32(g))));
        vst1q_f32(up + i, vmulq_f32(vld1q_f32(up + i), s));
    }
    silu_mul_scalar(up + i, gate + i, n - i);
}
#endif

void vecmath_select(int simd) {
#if defined(SM_HAVE_AVX2)
    silu_mul = simd ? silu_mul_avx2 : silu_mul_scalar;
#elif defined(SM_HAVE_NEON)
    silu_mul = simd ? silu_mul_neon : silu_mul_scalar;
#else
    (void)simd;
    silu_mul = silu_mul_scalar;
#endif
}
