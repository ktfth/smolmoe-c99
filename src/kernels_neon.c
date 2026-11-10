/* NEON, SDOT (FEAT_DotProd) and i8mm USDOT kernels. Each variant is compiled
 * with a function target attribute and selected at runtime. Integer lanes match
 * the scalar kernel, including a weight byte of -128. */
#include "kernels.h"

#if defined(SM_HAVE_NEON)
#  include "f16.h"
#  include "util.h"

#  include <arm_neon.h>
#  include <math.h>
#  include <string.h>

typedef struct Sum8 {
    int32x4_t lo, hi;
} Sum8;

static uint16_t sm_u16n(const unsigned char *p) { return (uint16_t)(p[0] | (unsigned)p[1] << 8); }

static unsigned char *sm_align64n(unsigned char *p) {
    uintptr_t u = (uintptr_t)p;
    return (unsigned char *)((u + 63u) & ~(uintptr_t)63u);
}

static void sm_store4n(float *out, size_t stride, uint32_t t, uint32_t row, const float y[4]) {
    uint32_t i;
    for (i = 0; i < 4; ++i) out[(size_t)(t + i) * stride + row] = y[i];
}

static void sm_interleave2n(const unsigned char *a, const unsigned char *b, unsigned char *dst, uint32_t nb) {
    uint32_t i;
    for (i = 0; i < nb; ++i) {
        memcpy(dst, a, Q8_BLOCK_BYTES);
        memcpy(dst + Q8_BLOCK_BYTES, b, Q8_BLOCK_BYTES);
        a += Q8_BLOCK_BYTES;
        b += Q8_BLOCK_BYTES;
        dst += 2 * Q8_BLOCK_BYTES;
    }
}

/* Same Q5 expansion as the baseline NEON dot. */
static void sm_unpack_neon(const unsigned char *body, int8x16_t *e0, int8x16_t *e1) {
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

static void sm_q5_interleave2n(const unsigned char *a, const unsigned char *b, unsigned char *dst, uint32_t nb) {
    uint32_t i;
    for (i = 0; i < nb; ++i, a += Q5_BLOCK_BYTES, b += Q5_BLOCK_BYTES, dst += 2 * Q8_BLOCK_BYTES) {
        int8x16_t a0, a1, b0, b1;
        float s0 = f16_to_f32(sm_u16n(a)), s1 = f16_to_f32(sm_u16n(b));
        sm_unpack_neon(a + 2, &a0, &a1);
        sm_unpack_neon(b + 2, &b0, &b1);
        memcpy(dst, &s0, 4);
        vst1q_s8((int8_t *)(dst + 4), a0);
        vst1q_s8((int8_t *)(dst + 20), a1);
        memcpy(dst + Q8_BLOCK_BYTES, &s1, 4);
        vst1q_s8((int8_t *)(dst + Q8_BLOCK_BYTES + 4), b0);
        vst1q_s8((int8_t *)(dst + Q8_BLOCK_BYTES + 20), b1);
    }
}

static float sm_reduce(float32x4_t lo, float32x4_t hi) {
    float32x4_t s = vaddq_f32(lo, hi);
    float32x2_t t = vadd_f32(vget_low_f32(s), vget_high_f32(s));
    return vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
}

/* |w| as a u8 (wrapping -128 -> 128) times sign(w)*x, the USDOT form of w*x. */
__attribute__((target("+i8mm")))
static int32x4_t sm_usdot(int8x16_t w, int8x16_t x) {
    uint8x16_t neg = vcltq_s8(w, vdupq_n_s8(0));
    uint8x16_t aw = vbslq_u8(neg, vreinterpretq_u8_s8(vnegq_s8(w)), vreinterpretq_u8_s8(w));
    int8x16_t sx = vbslq_s8(neg, vnegq_s8(x), x);
    return vusdotq_s32(vdupq_n_s32(0), aw, sx);
}

#  define SM_SFX neon
#  define SM_KIND 0
#  define SM_FAST 0
#  define SM_SKIP_DOT 1
#  include "kernels_neon.inc"

#  define SM_SFX neon_fast
#  define SM_KIND 0
#  define SM_FAST 1
#  include "kernels_neon.inc"

#  define SM_SFX neon_dotprod
#  define SM_KIND 1
#  define SM_FAST 0
#  define SM_TGT __attribute__((target("+dotprod")))
#  include "kernels_neon.inc"

#  define SM_SFX neon_dotprod_fast
#  define SM_KIND 1
#  define SM_FAST 1
#  define SM_TGT __attribute__((target("+dotprod")))
#  include "kernels_neon.inc"

#  define SM_SFX neon_i8mm
#  define SM_KIND 2
#  define SM_FAST 0
#  define SM_TGT __attribute__((target("+i8mm")))
#  include "kernels_neon.inc"

#  define SM_SFX neon_i8mm_fast
#  define SM_KIND 2
#  define SM_FAST 1
#  define SM_TGT __attribute__((target("+i8mm")))
#  include "kernels_neon.inc"

#endif
