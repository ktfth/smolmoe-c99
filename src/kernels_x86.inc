/* One ISA variant of the int8 dot and the 2x4 prefill tile.
 * The includer defines SM_SFX, SM_TARGET, SM_PROD, SM_FAST and maybe SM_SKIP_DOT.
 * SM_PROD(wv, xv) is the eight exact 4-product lanes (evaluated more than once). */
#define SM_CAT(a, b) a##b
#define SM_X(a, b) SM_CAT(a, b)
#define SM_Q8 SM_X(dot_q8_, SM_SFX)
#define SM_Q8X4 SM_X(SM_Q8, _x4)
#define SM_Q8R4 SM_X(SM_Q8, _r4)
#define SM_Q5 SM_X(dot_q5_, SM_SFX)
#define SM_Q5X4 SM_X(SM_Q5, _x4)
#define SM_Q5R4 SM_X(SM_Q5, _r4)
#define SM_G8 SM_X(gemm_q8_, SM_SFX)
#define SM_G5 SM_X(gemm_q5_, SM_SFX)
#define SM_DOT SM_X(sm_dot_, SM_SFX)
#define SM_X4F SM_X(sm_x4_, SM_SFX)
#define SM_R4F SM_X(sm_r4_, SM_SFX)
#define SM_TILE SM_X(sm_tile_, SM_SFX)
#define SM_Q5DOT SM_X(sm_q5dot_, SM_SFX)
#define SM_Q5X4F SM_X(sm_q5x4_, SM_SFX)
#define SM_Q5R4F SM_X(sm_q5r4_, SM_SFX)

#if SM_FAST
#  define SM_ACC __m256
#  define SM_ZERO sm_zero()
#  define SM_UPD(acc, scale, prod) sm_fma((acc), (scale), (prod))
#  define SM_OUT(acc) sm_reduce8(acc)
#else
#  define SM_ACC __m256
#  define SM_ZERO sm_zero()
#  define SM_UPD(acc, scale, prod) sm_acc((acc), (scale), (prod))
#  define SM_OUT(acc) sm_reduce8(acc)
#endif

__attribute__((target(SM_TARGET)))
static float SM_DOT(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    SM_ACC acc = SM_ZERO;
    uint32_t blk;
    for (blk = 0; blk < nb; ++blk, w += Q8_BLOCK_BYTES, xq += QK) {
        __m256i wv = SM_LOAD(w + 4), xv = SM_LOAD(xq);
        acc = SM_UPD(acc, rd_f32(w) * xd[blk], SM_PROD(wv, xv));
    }
    return SM_OUT(acc);
}

__attribute__((target(SM_TARGET)))
static void SM_X4F(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                   float out[4]) {
    const size_t qs = (size_t)stride * QK;
    const int8_t *x1 = xq + qs, *x2 = xq + 2 * qs, *x3 = xq + 3 * qs;
    const float *d1 = xd + stride, *d2 = xd + 2 * (size_t)stride, *d3 = xd + 3 * (size_t)stride;
    SM_ACC a0 = SM_ZERO, a1 = SM_ZERO, a2 = SM_ZERO, a3 = SM_ZERO;
    uint32_t blk;
    for (blk = 0; blk < nb; ++blk, w += Q8_BLOCK_BYTES, xq += QK, x1 += QK, x2 += QK, x3 += QK) {
        __m256i wv = SM_LOAD(w + 4);
        float ws = rd_f32(w);
        __m256i xv = SM_LOAD(xq);
        a0 = SM_UPD(a0, ws * xd[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x1);
        a1 = SM_UPD(a1, ws * d1[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x2);
        a2 = SM_UPD(a2, ws * d2[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x3);
        a3 = SM_UPD(a3, ws * d3[blk], SM_PROD(wv, xv));
    }
    out[0] = SM_OUT(a0);
    out[1] = SM_OUT(a1);
    out[2] = SM_OUT(a2);
    out[3] = SM_OUT(a3);
}

/* Four weight rows, one activation: the decode microkernel. */
__attribute__((target(SM_TARGET)))
static void SM_R4F(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                   float out[4]) {
    const unsigned char *r1 = w + row_bytes, *r2 = r1 + row_bytes, *r3 = r2 + row_bytes;
    SM_ACC a0 = SM_ZERO, a1 = SM_ZERO, a2 = SM_ZERO, a3 = SM_ZERO;
    uint32_t blk;
    for (blk = 0; blk < nb; ++blk, w += Q8_BLOCK_BYTES, r1 += Q8_BLOCK_BYTES, r2 += Q8_BLOCK_BYTES,
        r3 += Q8_BLOCK_BYTES, xq += QK) {
        __m256i xv = SM_LOAD(xq), wv = SM_LOAD(w + 4);
        a0 = SM_UPD(a0, rd_f32(w) * xd[blk], SM_PROD(wv, xv));
        wv = SM_LOAD(r1 + 4);
        a1 = SM_UPD(a1, rd_f32(r1) * xd[blk], SM_PROD(wv, xv));
        wv = SM_LOAD(r2 + 4);
        a2 = SM_UPD(a2, rd_f32(r2) * xd[blk], SM_PROD(wv, xv));
        wv = SM_LOAD(r3 + 4);
        a3 = SM_UPD(a3, rd_f32(r3) * xd[blk], SM_PROD(wv, xv));
    }
    out[0] = SM_OUT(a0);
    out[1] = SM_OUT(a1);
    out[2] = SM_OUT(a2);
    out[3] = SM_OUT(a3);
}

/* Two weight rows against four activations. pair_stride/block_stride describe
 * either the original matrix (row_bytes, 36) or an interleaved repack (36, 72). */
__attribute__((target(SM_TARGET)))
static void SM_TILE(const unsigned char *w, size_t pair_stride, size_t block_stride, const float *xd,
                    const int8_t *xq, uint32_t nb, float *out, size_t stride) {
    const unsigned char *w1 = w + pair_stride;
    const size_t qs = (size_t)nb * QK;
    const int8_t *x0 = xq, *x1 = xq + qs, *x2 = xq + 2 * qs, *x3 = xq + 3 * qs;
    const float *d0 = xd, *d1 = xd + nb, *d2 = xd + 2 * (size_t)nb, *d3 = xd + 3 * (size_t)nb;
    SM_ACC a0 = SM_ZERO, a1 = SM_ZERO, a2 = SM_ZERO, a3 = SM_ZERO;
    SM_ACC b0 = SM_ZERO, b1 = SM_ZERO, b2 = SM_ZERO, b3 = SM_ZERO;
    uint32_t blk;
    for (blk = 0; blk < nb; ++blk) {
        __m256i wv = SM_LOAD(w + 4), xv;
        float ws = rd_f32(w);
        xv = SM_LOAD(x0);
        a0 = SM_UPD(a0, ws * d0[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x1);
        a1 = SM_UPD(a1, ws * d1[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x2);
        a2 = SM_UPD(a2, ws * d2[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x3);
        a3 = SM_UPD(a3, ws * d3[blk], SM_PROD(wv, xv));
        wv = SM_LOAD(w1 + 4);
        ws = rd_f32(w1);
        xv = SM_LOAD(x0);
        b0 = SM_UPD(b0, ws * d0[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x1);
        b1 = SM_UPD(b1, ws * d1[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x2);
        b2 = SM_UPD(b2, ws * d2[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x3);
        b3 = SM_UPD(b3, ws * d3[blk], SM_PROD(wv, xv));
        w += block_stride;
        w1 += block_stride;
        x0 += QK;
        x1 += QK;
        x2 += QK;
        x3 += QK;
    }
    out[0] = SM_OUT(a0);
    out[stride] = SM_OUT(a1);
    out[2 * stride] = SM_OUT(a2);
    out[3 * stride] = SM_OUT(a3);
    out[1] = SM_OUT(b0);
    out[stride + 1] = SM_OUT(b1);
    out[2 * stride + 1] = SM_OUT(b2);
    out[3 * stride + 1] = SM_OUT(b3);
}

__attribute__((target(SM_TARGET)))
static float SM_Q5DOT(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    SM_ACC acc = SM_ZERO;
    uint32_t blk;
    for (blk = 0; blk < nb; ++blk, w += Q5_BLOCK_BYTES, xq += QK) {
        __m256i wv = sm_unpack_q5(w + 2), xv = SM_LOAD(xq);
        acc = SM_UPD(acc, f16_to_f32(sm_u16(w)) * xd[blk], SM_PROD(wv, xv));
    }
    return SM_OUT(acc);
}

__attribute__((target(SM_TARGET)))
static void SM_Q5X4F(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                     float out[4]) {
    const size_t qs = (size_t)stride * QK;
    const int8_t *x1 = xq + qs, *x2 = xq + 2 * qs, *x3 = xq + 3 * qs;
    const float *d1 = xd + stride, *d2 = xd + 2 * (size_t)stride, *d3 = xd + 3 * (size_t)stride;
    SM_ACC a0 = SM_ZERO, a1 = SM_ZERO, a2 = SM_ZERO, a3 = SM_ZERO;
    uint32_t blk;
    for (blk = 0; blk < nb; ++blk, w += Q5_BLOCK_BYTES, xq += QK, x1 += QK, x2 += QK, x3 += QK) {
        __m256i wv = sm_unpack_q5(w + 2), xv;
        float ws = f16_to_f32(sm_u16(w));
        xv = SM_LOAD(xq);
        a0 = SM_UPD(a0, ws * xd[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x1);
        a1 = SM_UPD(a1, ws * d1[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x2);
        a2 = SM_UPD(a2, ws * d2[blk], SM_PROD(wv, xv));
        xv = SM_LOAD(x3);
        a3 = SM_UPD(a3, ws * d3[blk], SM_PROD(wv, xv));
    }
    out[0] = SM_OUT(a0);
    out[1] = SM_OUT(a1);
    out[2] = SM_OUT(a2);
    out[3] = SM_OUT(a3);
}

__attribute__((target(SM_TARGET)))
static void SM_Q5R4F(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                     float out[4]) {
    const unsigned char *r1 = w + row_bytes, *r2 = r1 + row_bytes, *r3 = r2 + row_bytes;
    SM_ACC a0 = SM_ZERO, a1 = SM_ZERO, a2 = SM_ZERO, a3 = SM_ZERO;
    uint32_t blk;
    for (blk = 0; blk < nb; ++blk, w += Q5_BLOCK_BYTES, r1 += Q5_BLOCK_BYTES, r2 += Q5_BLOCK_BYTES,
        r3 += Q5_BLOCK_BYTES, xq += QK) {
        __m256i xv = SM_LOAD(xq), wv = sm_unpack_q5(w + 2);
        a0 = SM_UPD(a0, f16_to_f32(sm_u16(w)) * xd[blk], SM_PROD(wv, xv));
        wv = sm_unpack_q5(r1 + 2);
        a1 = SM_UPD(a1, f16_to_f32(sm_u16(r1)) * xd[blk], SM_PROD(wv, xv));
        wv = sm_unpack_q5(r2 + 2);
        a2 = SM_UPD(a2, f16_to_f32(sm_u16(r2)) * xd[blk], SM_PROD(wv, xv));
        wv = sm_unpack_q5(r3 + 2);
        a3 = SM_UPD(a3, f16_to_f32(sm_u16(r3)) * xd[blk], SM_PROD(wv, xv));
    }
    out[0] = SM_OUT(a0);
    out[1] = SM_OUT(a1);
    out[2] = SM_OUT(a2);
    out[3] = SM_OUT(a3);
}

__attribute__((target(SM_TARGET)))
void SM_G8(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
           uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch) {
    unsigned char *pack = scratch ? sm_align64(scratch) : NULL;
    uint32_t r, t, i;
    if (!nrows || !ntok) return;
    if (ntok == 1) {
        for (r = 0; r + 4 <= nrows; r += 4) {
            float y[4];
            SM_R4F(w + (size_t)r * row_bytes, row_bytes, xd, xq, nb, y);
            for (i = 0; i < 4; ++i) out[r + i] = y[i];
        }
        for (; r < nrows; ++r) out[r] = SM_DOT(w + (size_t)r * row_bytes, xd, xq, nb);
        return;
    }
    for (r = 0; r + 2 <= nrows; r += 2) {
        const unsigned char *a = w + (size_t)r * row_bytes, *b = a + row_bytes;
        const unsigned char *src = a;
        size_t pair = row_bytes, step = Q8_BLOCK_BYTES;
        /* Repack only when the pair will not stay resident in a small L1. */
        if (pack && ntok >= 8 && row_bytes > 16u * 1024u) {
            sm_interleave2(a, b, pack, nb);
            src = pack;
            pair = Q8_BLOCK_BYTES;
            step = 2 * Q8_BLOCK_BYTES;
        }
        for (t = 0; t + 4 <= ntok; t += 4)
            SM_TILE(src, pair, step, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, out + (size_t)t * stride + r,
                    stride);
        for (; t < ntok; ++t) {
            out[(size_t)t * stride + r] = SM_DOT(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
            out[(size_t)t * stride + r + 1] = SM_DOT(b, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
        }
    }
    if (r < nrows) {
        const unsigned char *a = w + (size_t)r * row_bytes;
        for (t = 0; t + 4 <= ntok; t += 4) {
            float y[4];
            SM_X4F(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, nb, y);
            sm_store4(out, stride, t, r, y);
        }
        for (; t < ntok; ++t)
            out[(size_t)t * stride + r] = SM_DOT(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
    }
}

__attribute__((target(SM_TARGET)))
void SM_G5(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows,
           uint32_t ntok, uint32_t nb, float *out, size_t stride, unsigned char *scratch) {
    unsigned char *pack = scratch ? sm_align64(scratch) : NULL;
    uint32_t r, t, i;
    if (!nrows || !ntok) return;
    if (ntok == 1) {
        for (r = 0; r + 4 <= nrows; r += 4) {
            float y[4];
            SM_Q5R4F(w + (size_t)r * row_bytes, row_bytes, xd, xq, nb, y);
            for (i = 0; i < 4; ++i) out[r + i] = y[i];
        }
        for (; r < nrows; ++r) out[r] = SM_Q5DOT(w + (size_t)r * row_bytes, xd, xq, nb);
        return;
    }
    for (r = 0; r + 2 <= nrows; r += 2) {
        const unsigned char *a = w + (size_t)r * row_bytes, *b = a + row_bytes;
        if (pack && ntok >= 8 && (size_t)nb * Q8_BLOCK_BYTES > 16u * 1024u) {
            sm_q5_interleave2(a, b, pack, nb);
            for (t = 0; t + 4 <= ntok; t += 4)
                SM_TILE(pack, Q8_BLOCK_BYTES, 2 * Q8_BLOCK_BYTES, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb,
                        out + (size_t)t * stride + r, stride);
        } else {
            for (t = 0; t + 4 <= ntok; t += 4) {
                float y[4];
                SM_Q5X4F(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, nb, y);
                sm_store4(out, stride, t, r, y);
                SM_Q5X4F(b, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, nb, y);
                sm_store4(out, stride, t, r + 1, y);
            }
        }
        for (t = ntok - (ntok % 4); t < ntok; ++t) {
            out[(size_t)t * stride + r] = SM_Q5DOT(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
            out[(size_t)t * stride + r + 1] = SM_Q5DOT(b, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
        }
    }
    if (r < nrows) {
        const unsigned char *a = w + (size_t)r * row_bytes;
        for (t = 0; t + 4 <= ntok; t += 4) {
            float y[4];
            SM_Q5X4F(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, nb, y);
            sm_store4(out, stride, t, r, y);
        }
        for (; t < ntok; ++t)
            out[(size_t)t * stride + r] = SM_Q5DOT(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
    }
}

#if !defined(SM_SKIP_DOT)
__attribute__((target(SM_TARGET)))
float SM_Q8(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    return SM_DOT(w, xd, xq, nb);
}
__attribute__((target(SM_TARGET)))
void SM_Q8X4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride, float out[4]) {
    SM_X4F(w, xd, xq, nb, stride, out);
}
__attribute__((target(SM_TARGET)))
float SM_Q5(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    return SM_Q5DOT(w, xd, xq, nb);
}
__attribute__((target(SM_TARGET)))
void SM_Q5X4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride, float out[4]) {
    SM_Q5X4F(w, xd, xq, nb, stride, out);
}
#endif

__attribute__((target(SM_TARGET)))
void SM_Q8R4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb, float out[4]) {
    SM_R4F(w, row_bytes, xd, xq, nb, out);
}
__attribute__((target(SM_TARGET)))
void SM_Q5R4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb, float out[4]) {
    SM_Q5R4F(w, row_bytes, xd, xq, nb, out);
}

#undef SM_ACC
#undef SM_ZERO
#undef SM_UPD
#undef SM_OUT
#undef SM_Q8
#undef SM_Q8X4
#undef SM_Q8R4
#undef SM_Q5
#undef SM_Q5X4
#undef SM_Q5R4
#undef SM_G8
#undef SM_G5
#undef SM_DOT
#undef SM_X4F
#undef SM_R4F
#undef SM_TILE
#undef SM_Q5DOT
#undef SM_Q5X4F
#undef SM_Q5R4F
#undef SM_CAT
#undef SM_X
#undef SM_SFX
#undef SM_TARGET
#undef SM_PROD
#undef SM_FAST
#undef SM_SKIP_DOT
