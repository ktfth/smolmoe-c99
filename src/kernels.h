/* Q8 compute kernels.
 *
 * Weights: rows of 36-byte blocks {f32 scale, int8 q[32]} (docs/FORMAT.md).
 * Activations are quantized per 32-value block to {f32 scale, int8 q[32]}.
 * A block's integer dot product is exact, and the float accumulation follows
 * a fixed 8-lane order that the scalar and SIMD kernels reproduce bit for bit.
 * Results therefore do not depend on the backend, thread count or batch size. */
#ifndef SMOLMOE_KERNELS_H
#define SMOLMOE_KERNELS_H

#include <float.h>
#include <stdint.h>

/* The bit-identity contract needs plain binary32 evaluation (x87 extended
 * precision would make scalar and SIMD results differ). */
#if defined(FLT_EVAL_METHOD) && FLT_EVAL_METHOD != 0
#  error "smolmoe requires FLT_EVAL_METHOD == 0; on 32-bit x86 build with -msse2 -mfpmath=sse"
#endif

/* AVX2 (+F16C) is compiled per function and selected at runtime: no -mavx2 needed. */
#if !defined(SM_NO_SIMD) && (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
#  define SM_HAVE_AVX2 1
#endif

/* NEON (Advanced SIMD) is part of every AArch64 CPU: no runtime check. The Q5
 * unpacking assumes little-endian lanes, as every AArch64 OS we target uses. */
#if !defined(SM_NO_SIMD) && defined(__aarch64__) && defined(__ARM_NEON) && \
    (!defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#  define SM_HAVE_NEON 1
#endif

/* WebAssembly SIMD128 is chosen at compile time (-msimd128): a module built
 * with it loads only where SIMD128 exists, which is every current browser.
 * Relaxed SIMD is never used: its results vary between machines. */
#if !defined(SM_NO_SIMD) && defined(__wasm_simd128__)
#  define SM_HAVE_WASM_SIMD 1
#endif

#define QK 32u
#define Q8_BLOCK_BYTES 36u /* f32 scale + 32 x int8 */
#define Q5_BLOCK_BYTES 22u /* f16 scale, u32 high bits (bit j = element j), 16 bytes of low nibbles (j low, j+16 high); value = q - 16 */

typedef float (*DotQ8)(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
/* One weight row against 4 activation rows spaced `stride` blocks apart.
 * Each output equals dot_q8() for that row exactly; sharing the weight load
 * and interleaving 4 independent accumulators hides FP add latency. */
typedef void (*DotQ8x4)(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                        uint32_t stride, float out[4]);

extern DotQ8 dot_q8, dot_q5;
extern DotQ8x4 dot_q8_x4, dot_q5_x4;

/* Selects the fastest supported backend; returns its name. */
const char *kernels_select(int force_scalar);

float dot_q8_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]);
float dot_q5_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]);
#if defined(SM_HAVE_AVX2)
float dot_q8_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
float dot_q5_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
#endif
#if defined(SM_HAVE_NEON)
float dot_q8_neon(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_neon_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
float dot_q5_neon(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_neon_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
#endif
#if defined(SM_HAVE_WASM_SIMD)
float dot_q8_wasm(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_wasm_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
float dot_q5_wasm(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_wasm_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
#endif

/* Quantizes n floats into ceil(n/32) blocks; padding is zero. A block whose
 * scale would be subnormal quantizes to zeros. Returns 0, or -1 if the input
 * is not finite. Both versions are bit-identical. */
typedef int (*QuantizeQ8)(const float *x, uint32_t n, float *d, int8_t *q);
extern QuantizeQ8 quantize_q8;
int quantize_q8_scalar(const float *x, uint32_t n, float *d, int8_t *q);
#if defined(SM_HAVE_AVX2)
int quantize_q8_avx2(const float *x, uint32_t n, float *d, int8_t *q);
#endif
#if defined(SM_HAVE_NEON)
int quantize_q8_neon(const float *x, uint32_t n, float *d, int8_t *q);
#endif
#if defined(SM_HAVE_WASM_SIMD)
int quantize_q8_wasm(const float *x, uint32_t n, float *d, int8_t *q);
#endif

/* Expands one Q8 row of `cols` values. */
void dequantize_q8_row(const unsigned char *w, uint32_t cols, float *out);
void dequantize_q5_row(const unsigned char *w, uint32_t cols, float *out);

/* Quantizes one row of `cols` finite floats to Q5, choosing each block scale
 * from a small candidate set to minimize squared error. Used by the converter. */
void quantize_q5_row(const float *x, uint32_t cols, unsigned char *out);

/* Sum of a[i]*b[i] in a fixed 8-lane order. The scalar reference for attention. */
float dot_f32(const float *a, const float *b, uint32_t n);

/* One query head over KV positions [start, pos]. keys/values at position p
 * begin at p * stride (a head packed inside a token). The f32 path is
 * bit-identical on every backend; f16 rounds each stored element with
 * f32_to_f16 and then does the same 8-lane dot. */
typedef void (*AttendF32)(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                          uint32_t start, uint32_t pos, float *scores, float *out);
typedef void (*AttendF16)(const float *q, const uint16_t *keys, const uint16_t *values, uint32_t stride, uint32_t hd,
                          uint32_t start, uint32_t pos, float *scores, float *out);
void attend_f32_scalar(const float *q, const float *keys, const float *values, uint32_t stride, uint32_t hd,
                       uint32_t start, uint32_t pos, float *scores, float *out);
void attend_f16_scalar(const float *q, const uint16_t *keys, const uint16_t *values, uint32_t stride, uint32_t hd,
                       uint32_t start, uint32_t pos, float *scores, float *out);
extern AttendF32 attend_f32;
extern AttendF16 attend_f16;

/* f32_to_f16() over n values. The SIMD path matches the scalar one bit for bit. */
typedef void (*PackF16)(const float *src, uint16_t *dst, uint32_t n);
void pack_f16_scalar(const float *src, uint16_t *dst, uint32_t n);
extern PackF16 pack_f16;

#endif
