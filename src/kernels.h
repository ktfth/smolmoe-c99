/* Q8 compute kernels.
 *
 * Weights: rows of 36-byte blocks {f32 scale, int8 q[32]} (docs/FORMAT.md).
 * Activations are quantized per 32-value block to {f32 scale, int8 q[32]}.
 * A block's integer dot product is exact, and the float accumulation follows
 * a fixed 8-lane order that the scalar and SIMD kernels reproduce bit for bit.
 * Results therefore do not depend on the backend, thread count or batch size. */
#ifndef SMOLMOE_KERNELS_H
#define SMOLMOE_KERNELS_H

#include <stdint.h>

/* AVX2 is compiled per function and selected at runtime: no -mavx2 needed. */
#if !defined(SM_NO_SIMD) && (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#  define SM_HAVE_AVX2 1
#endif

#define QK 32u
#define Q8_BLOCK_BYTES 36u

typedef float (*DotQ8)(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
/* One weight row against 4 activation rows spaced `stride` blocks apart.
 * Each output equals dot_q8() for that row exactly; sharing the weight load
 * and interleaving 4 independent accumulators hides FP add latency. */
typedef void (*DotQ8x4)(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                        uint32_t stride, float out[4]);

extern DotQ8 dot_q8;
extern DotQ8x4 dot_q8_x4;

/* Selects the fastest supported backend; returns its name. */
const char *kernels_select(int force_scalar);

float dot_q8_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]);
#if defined(SM_HAVE_AVX2)
float dot_q8_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
#endif

/* Quantizes n floats into ceil(n/32) blocks; padding is zero.
 * Returns 0, or -1 if the input is not finite. */
int quantize_q8(const float *x, uint32_t n, float *d, int8_t *q);

/* Expands one Q8 row of `cols` values. */
void dequantize_q8_row(const unsigned char *w, uint32_t cols, float *out);

/* Sum of a[i]*b[i] in a fixed 8-lane order (vectorizable, deterministic). */
float dot_f32(const float *a, const float *b, uint32_t n);

#endif
