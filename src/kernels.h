/* Q8 compute kernels.
 *
 * Weights: rows of 36-byte blocks {f32 scale, int8 q[32]} (docs/FORMAT.md).
 * Activations are quantized per 32-value block to {f32 scale, int8 q[32]}.
 * A block's integer dot product is exact, and the float accumulation follows
 * a fixed 8-lane order that the scalar and SIMD kernels reproduce bit for bit.
 * Results therefore do not depend on the backend, thread count or batch size.
 *
 * kernels_set_fast() keeps those integer lanes but updates each one with a
 * single FMA. The final reduction is unchanged, so fast backends still match
 * each other; they do not match the default mul+add kernels. */
#ifndef SMOLMOE_KERNELS_H
#define SMOLMOE_KERNELS_H

#include <float.h>
#include <stddef.h>
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

/* Baseline NEON is part of every AArch64 CPU. SDOT and i8mm are picked at
 * runtime. Q5 unpacking assumes little-endian lanes, as every target OS does. */
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
/* Four weight rows spaced `row_bytes` apart, one activation. out[i] equals
 * dot_q8() of row i. Four independent accumulators hide the lane-add latency
 * of a single-token decode. */
typedef void (*DotQ8R4)(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq,
                        uint32_t nb, float out[4]);
/* nrows weight rows by ntok activation rows. out[token * stride + row] equals
 * the dot of that pair. `scratch` is gemm_scratch_bytes(nb) bytes, or NULL to
 * skip the repack (still exact, one streaming pass). */
typedef void (*GemmQ8)(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq,
                       uint32_t nrows, uint32_t ntok, uint32_t nb, float *out, size_t stride,
                       unsigned char *scratch);

extern DotQ8 dot_q8, dot_q5;
extern DotQ8x4 dot_q8_x4, dot_q5_x4;
extern DotQ8R4 dot_q8_r4, dot_q5_r4;
extern GemmQ8 gemm_q8, gemm_q5;

/* Bytes of scratch one matrix tile needs for `nb` blocks (two repacked rows). */
size_t gemm_scratch_bytes(uint32_t nb);

/* Default is the bit-exact mul+add contract. --fast (call before
 * kernels_select, or again afterwards) uses a lane-wise FMA instead. */
void kernels_set_fast(int fast);
int kernels_fast(void);

/* Nonzero when this process can run the matching kernel. */
int kernels_cpu_avxvnni(void);
int kernels_cpu_avx512vnni(void);
int kernels_cpu_dotprod(void);
int kernels_cpu_i8mm(void);

/* Selects the fastest supported backend; returns its name. */
const char *kernels_select(int force_scalar);

float dot_q8_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]);
float dot_q5_scalar(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_scalar_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                      uint32_t stride, float out[4]);
void dot_q8_scalar_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                      float out[4]);
void dot_q5_scalar_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                      float out[4]);
void gemm_q8_scalar(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                    uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
void gemm_q5_scalar(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                    uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
float dot_q8_scalar_fast(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_scalar_fast_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                           float out[4]);
void dot_q8_scalar_fast_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                           float out[4]);
float dot_q5_scalar_fast(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_scalar_fast_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                           float out[4]);
void dot_q5_scalar_fast_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                           float out[4]);
void gemm_q8_scalar_fast(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                         uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
void gemm_q5_scalar_fast(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                         uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
#define SM_DECLARE_KERNEL(sfx)                                                                                      \
    float dot_q8_##sfx(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);                      \
    void dot_q8_##sfx##_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,  \
                           float out[4]);                                                                            \
    void dot_q8_##sfx##_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb, \
                           float out[4]);                                                                            \
    float dot_q5_##sfx(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);                      \
    void dot_q5_##sfx##_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,  \
                           float out[4]);                                                                            \
    void dot_q5_##sfx##_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb, \
                           float out[4]);                                                                            \
    void gemm_q8_##sfx(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,  \
                       uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);               \
    void gemm_q5_##sfx(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,  \
                       uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
#if defined(SM_HAVE_AVX2)
float dot_q8_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
float dot_q5_avx2(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_avx2_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
void dot_q8_avx2_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                    float out[4]);
void dot_q5_avx2_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                    float out[4]);
void gemm_q8_avx2(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                  uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
void gemm_q5_avx2(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                  uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
SM_DECLARE_KERNEL(avx2_fast)
SM_DECLARE_KERNEL(avxvnni)
SM_DECLARE_KERNEL(avxvnni_fast)
SM_DECLARE_KERNEL(avx512vnni)
SM_DECLARE_KERNEL(avx512vnni_fast)
#endif
#if defined(SM_HAVE_NEON)
float dot_q8_neon(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q8_neon_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
float dot_q5_neon(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb);
void dot_q5_neon_x4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb,
                    uint32_t stride, float out[4]);
void dot_q8_neon_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                    float out[4]);
void dot_q5_neon_r4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                    float out[4]);
void gemm_q8_neon(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                  uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
void gemm_q5_neon(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
                  uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch);
SM_DECLARE_KERNEL(neon_fast)
SM_DECLARE_KERNEL(neon_dotprod)
SM_DECLARE_KERNEL(neon_dotprod_fast)
SM_DECLARE_KERNEL(neon_i8mm)
SM_DECLARE_KERNEL(neon_i8mm_fast)
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

/* Sum of a[i]*b[i] in a fixed 8-lane order (vectorizable, deterministic). */
float dot_f32(const float *a, const float *b, uint32_t n);

#endif
