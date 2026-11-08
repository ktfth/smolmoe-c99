/* Deterministic elementwise math for the forward pass.
 *
 * The scalar and SIMD (AVX2, NEON, WebAssembly SIMD128) paths execute the
 * same operations in the same order (no FMA, no libm), so their results are
 * bit-identical, and unlike the C library's expf they do not change between
 * operating systems. */
#ifndef SMOLMOE_VECMATH_H
#define SMOLMOE_VECMATH_H

#include <stddef.h>

#include "kernels.h"

/* e^x computed in binary64 (32-entry table and a degree-4 polynomial) and
 * rounded to binary32 once: correctly rounded except for about 1 input in
 * 10^5, which is 1 ulp off. Gradual underflow, 0 below about -103.9, +inf
 * above about 88.72, NaN for NaN. */
float exp_det(float x);

/* up[i] *= gate[i] / (1 + e^-gate[i]): the SwiGLU activation. */
typedef void (*SiluMul)(float *up, const float *gate, size_t n);
extern SiluMul silu_mul;

void silu_mul_scalar(float *up, const float *gate, size_t n);
/* up[i] *= gelu_tanh(gate[i]): the GeGLU activation (Gemma 3). */
void gelu_mul_scalar(float *up, const float *gate, size_t n);
#if defined(SM_HAVE_AVX2)
void silu_mul_avx2(float *up, const float *gate, size_t n);
#endif
#if defined(SM_HAVE_NEON)
void silu_mul_neon(float *up, const float *gate, size_t n);
#endif
#if defined(SM_HAVE_WASM_SIMD)
void silu_mul_wasm(float *up, const float *gate, size_t n);
#endif

/* Points silu_mul at the SIMD (AVX2, NEON or SIMD128) or the scalar version (called by kernels_select). */
void vecmath_select(int simd);

#endif
