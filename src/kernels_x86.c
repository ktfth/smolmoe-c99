/* AVX2 / AVX-VNNI / AVX-512-VNNI dots and the prefill tile.
 * Compiled into every x86 build; each function opts into its ISA, and
 * kernels_select() picks one at runtime. The integer lanes match the scalar
 * kernel, including a weight of -128 (abs as an unsigned byte, activation
 * flipped). */
#include "kernels.h"
#include "f16.h"
#include "util.h"

#include <string.h>

#if defined(SM_HAVE_AVX2)
#  include <immintrin.h>

#  define SM_LOAD(p) _mm256_loadu_si256((const __m256i *)(const void *)(p))

__attribute__((always_inline, target("avx2,f16c")))
static inline __m256 sm_zero(void) { return _mm256_setzero_ps(); }

__attribute__((always_inline, target("avx2,f16c")))
static inline __m256 sm_acc(__m256 acc, float scale, __m256i prod) {
    return _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(prod)));
}

__attribute__((always_inline, target("avx2,f16c")))
static inline float sm_reduce8(__m256 acc) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    __m128 t = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(t, _mm_shuffle_ps(t, t, 1)));
}

__attribute__((always_inline, target("avx2,f16c,fma")))
static inline __m256 sm_fma(__m256 acc, float scale, __m256i prod) {
    return _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(prod), acc);
}

static uint16_t sm_u16(const unsigned char *p) { return (uint16_t)(p[0] | (unsigned)p[1] << 8); }

static unsigned char *sm_align64(unsigned char *p) {
    uintptr_t u = (uintptr_t)p;
    return (unsigned char *)((u + 63u) & ~(uintptr_t)63u);
}

static void sm_store4(float *out, size_t stride, uint32_t t, uint32_t row, const float y[4]) {
    uint32_t i;
    for (i = 0; i < 4; ++i) out[(size_t)(t + i) * stride + row] = y[i];
}

/* Two Q8 rows, blocks interleaved so the tile streams one contiguous buffer. */
static void sm_interleave2(const unsigned char *a, const unsigned char *b, unsigned char *dst, uint32_t nb) {
    uint32_t i;
    for (i = 0; i < nb; ++i) {
        memcpy(dst, a, Q8_BLOCK_BYTES);
        memcpy(dst + Q8_BLOCK_BYTES, b, Q8_BLOCK_BYTES);
        a += Q8_BLOCK_BYTES;
        b += Q8_BLOCK_BYTES;
        dst += 2 * Q8_BLOCK_BYTES;
    }
}

/* Same high-bit expansion as the AVX2 Q5 dot. */
__attribute__((target("avx2,f16c")))
static __m256i sm_unpack_q5(const unsigned char *body) {
    const __m256i v = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(const void *)(body + 4)));
    const __m256i low = _mm256_and_si256(_mm256_blend_epi32(v, _mm256_srli_epi16(v, 4), 0xF0), _mm256_set1_epi8(15));
    const __m256i spread = _mm256_shuffle_epi8(
        _mm256_set1_epi32((int)rd_u32(body)),
        _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0));
    const __m256i bits = _mm256_cmpeq_epi8(_mm256_or_si256(spread, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfeLL)),
                                           _mm256_set1_epi8(-1));
    return _mm256_sub_epi8(_mm256_or_si256(low, _mm256_and_si256(bits, _mm256_set1_epi8(16))), _mm256_set1_epi8(16));
}

__attribute__((target("avx2,f16c")))
static void sm_q5_interleave2(const unsigned char *a, const unsigned char *b, unsigned char *dst, uint32_t nb) {
    uint32_t i;
    for (i = 0; i < nb; ++i, a += Q5_BLOCK_BYTES, b += Q5_BLOCK_BYTES, dst += 2 * Q8_BLOCK_BYTES) {
        float s0 = f16_to_f32(sm_u16(a)), s1 = f16_to_f32(sm_u16(b));
        memcpy(dst, &s0, 4);
        _mm256_storeu_si256((__m256i *)(void *)(dst + 4), sm_unpack_q5(a + 2));
        memcpy(dst + Q8_BLOCK_BYTES, &s1, 4);
        _mm256_storeu_si256((__m256i *)(void *)(dst + Q8_BLOCK_BYTES + 4), sm_unpack_q5(b + 2));
    }
}

#  define SM_PROD_AVX2(wv, xv)                                                                                      \
      _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8((wv), (wv)), _mm256_sign_epi8((xv), (wv))),           \
                        _mm256_set1_epi16(1))
#  define SM_PROD_VNNI(wv, xv)                                                                                      \
      _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_sign_epi8((wv), (wv)), _mm256_sign_epi8((xv), (wv)))
#  define SM_PROD_VNNI512(wv, xv)                                                                                   \
      _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_sign_epi8((wv), (wv)), _mm256_sign_epi8((xv), (wv)))

#  define SM_SFX avx2
#  define SM_TARGET "avx2,f16c"
#  define SM_PROD(wv, xv) SM_PROD_AVX2(wv, xv)
#  define SM_FAST 0
#  define SM_SKIP_DOT 1
#  include "kernels_x86.inc"

#  define SM_SFX avx2_fast
#  define SM_TARGET "avx2,f16c,fma"
#  define SM_PROD(wv, xv) SM_PROD_AVX2(wv, xv)
#  define SM_FAST 1
#  include "kernels_x86.inc"

#  define SM_SFX avxvnni
#  define SM_TARGET "avx2,f16c,avxvnni"
#  define SM_PROD(wv, xv) SM_PROD_VNNI(wv, xv)
#  define SM_FAST 0
#  include "kernels_x86.inc"

#  define SM_SFX avxvnni_fast
#  define SM_TARGET "avx2,f16c,fma,avxvnni"
#  define SM_PROD(wv, xv) SM_PROD_VNNI(wv, xv)
#  define SM_FAST 1
#  include "kernels_x86.inc"

#  define SM_SFX avx512vnni
#  define SM_TARGET "avx512f,avx512vl,avx512vnni,avx2,f16c"
#  define SM_PROD(wv, xv) SM_PROD_VNNI512(wv, xv)
#  define SM_FAST 0
#  include "kernels_x86.inc"

#  define SM_SFX avx512vnni_fast
#  define SM_TARGET "avx512f,avx512vl,avx512vnni,avx2,f16c,fma"
#  define SM_PROD(wv, xv) SM_PROD_VNNI512(wv, xv)
#  define SM_FAST 1
#  include "kernels_x86.inc"

#endif
