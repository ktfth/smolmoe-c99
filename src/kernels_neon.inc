/* One NEON variant. SM_KIND 0 = vmull, 1 = SDOT, 2 = i8mm USDOT.
 * SM_FAST 0 keeps the 8-lane mul+add; 1 is one fmaf of the 32-product sum. */
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
#define SM_DOT SM_X(smn_dot_, SM_SFX)
#define SM_X4F SM_X(smn_x4_, SM_SFX)
#define SM_R4F SM_X(smn_r4_, SM_SFX)
#define SM_TILE SM_X(smn_tile_, SM_SFX)
#define SM_Q5DOT SM_X(smn_q5dot_, SM_SFX)
#define SM_Q5X4F SM_X(smn_q5x4_, SM_SFX)
#define SM_Q5R4F SM_X(smn_q5r4_, SM_SFX)
#define SM_PRODFN SM_X(smn_prod_, SM_SFX)
#ifndef SM_TGT
#  define SM_TGT
#  define SM_TGT_LOCAL 1
#endif

#if SM_FAST
#  define SM_NACC(n) float32x4_t lo[n], hi[n]
#  define SM_ZACC(n)                                                                                                 \
      do {                                                                                                           \
          uint32_t z_;                                                                                               \
          for (z_ = 0; z_ < (uint32_t)(n); ++z_) lo[z_] = hi[z_] = vdupq_n_f32(0.0f);                                \
      } while (0)
#  define SM_AUPD(i, scale, s)                                                                                       \
      do {                                                                                                           \
          float32x4_t sc_ = vdupq_n_f32(scale);                                                                      \
          lo[i] = vfmaq_f32(lo[i], sc_, vcvtq_f32_s32((s).lo));                                                      \
          hi[i] = vfmaq_f32(hi[i], sc_, vcvtq_f32_s32((s).hi));                                                      \
      } while (0)
#  define SM_AGET(i) sm_reduce(lo[i], hi[i])
#else
#  define SM_NACC(n) float32x4_t lo[n], hi[n]
#  define SM_ZACC(n)                                                                                                 \
      do {                                                                                                           \
          uint32_t z_;                                                                                               \
          for (z_ = 0; z_ < (uint32_t)(n); ++z_) lo[z_] = hi[z_] = vdupq_n_f32(0.0f);                                \
      } while (0)
#  define SM_AUPD(i, scale, s)                                                                                       \
      do {                                                                                                           \
          float32x4_t sc_ = vdupq_n_f32(scale);                                                                      \
          lo[i] = vaddq_f32(lo[i], vmulq_f32(sc_, vcvtq_f32_s32((s).lo)));                                           \
          hi[i] = vaddq_f32(hi[i], vmulq_f32(sc_, vcvtq_f32_s32((s).hi)));                                           \
      } while (0)
#  define SM_AGET(i) sm_reduce(lo[i], hi[i])
#endif

SM_TGT
static inline Sum8 SM_PRODFN(int8x16_t w0, int8x16_t x0, int8x16_t w1, int8x16_t x1) {
    Sum8 s;
#if SM_KIND == 1
    s.lo = vdotq_s32(vdupq_n_s32(0), w0, x0);
    s.hi = vdotq_s32(vdupq_n_s32(0), w1, x1);
#elif SM_KIND == 2
    s.lo = sm_usdot(w0, x0);
    s.hi = sm_usdot(w1, x1);
#else
    s.lo = vpaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(w0), vget_low_s8(x0))), vpaddlq_s16(vmull_high_s8(w0, x0)));
    s.hi = vpaddq_s32(vpaddlq_s16(vmull_s8(vget_low_s8(w1), vget_low_s8(x1))), vpaddlq_s16(vmull_high_s8(w1, x1)));
#endif
    return s;
}

SM_TGT
static inline Sum8 SM_X(smn_q8_, SM_SFX)(const unsigned char *w, const int8_t *x) {
    return SM_PRODFN(vld1q_s8((const int8_t *)(w + 4)), vld1q_s8(x), vld1q_s8((const int8_t *)(w + 20)),
                     vld1q_s8(x + 16));
}

SM_TGT
static inline Sum8 SM_X(smn_q5b_, SM_SFX)(const unsigned char *w, const int8_t *x) {
    int8x16_t e0, e1;
    sm_unpack_neon(w + 2, &e0, &e1);
    return SM_PRODFN(e0, vld1q_s8(x), e1, vld1q_s8(x + 16));
}

#define SM_Q8B SM_X(smn_q8_, SM_SFX)
#define SM_Q5B SM_X(smn_q5b_, SM_SFX)

SM_TGT
static float SM_DOT(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    SM_NACC(1);
    uint32_t blk;
    SM_ZACC(1);
    for (blk = 0; blk < nb; ++blk, w += Q8_BLOCK_BYTES, xq += QK)
        SM_AUPD(0, rd_f32(w) * xd[blk], SM_Q8B(w, xq));
    return SM_AGET(0);
}

SM_TGT
static void SM_X4F(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                   float out[4]) {
    const size_t qs = (size_t)stride * QK;
    const int8_t *x1 = xq + qs, *x2 = xq + 2 * qs, *x3 = xq + 3 * qs;
    const float *d1 = xd + stride, *d2 = xd + 2 * (size_t)stride, *d3 = xd + 3 * (size_t)stride;
    SM_NACC(4);
    uint32_t blk;
    SM_ZACC(4);
    for (blk = 0; blk < nb; ++blk, w += Q8_BLOCK_BYTES, xq += QK, x1 += QK, x2 += QK, x3 += QK) {
        float ws = rd_f32(w);
        int8x16_t w0 = vld1q_s8((const int8_t *)(w + 4)), w1 = vld1q_s8((const int8_t *)(w + 20));
        SM_AUPD(0, ws * xd[blk], SM_PRODFN(w0, vld1q_s8(xq), w1, vld1q_s8(xq + 16)));
        SM_AUPD(1, ws * d1[blk], SM_PRODFN(w0, vld1q_s8(x1), w1, vld1q_s8(x1 + 16)));
        SM_AUPD(2, ws * d2[blk], SM_PRODFN(w0, vld1q_s8(x2), w1, vld1q_s8(x2 + 16)));
        SM_AUPD(3, ws * d3[blk], SM_PRODFN(w0, vld1q_s8(x3), w1, vld1q_s8(x3 + 16)));
    }
    out[0] = SM_AGET(0);
    out[1] = SM_AGET(1);
    out[2] = SM_AGET(2);
    out[3] = SM_AGET(3);
}

SM_TGT
static void SM_R4F(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                   float out[4]) {
    const unsigned char *r1 = w + row_bytes, *r2 = r1 + row_bytes, *r3 = r2 + row_bytes;
    SM_NACC(4);
    uint32_t blk;
    SM_ZACC(4);
    for (blk = 0; blk < nb; ++blk, w += Q8_BLOCK_BYTES, r1 += Q8_BLOCK_BYTES, r2 += Q8_BLOCK_BYTES,
        r3 += Q8_BLOCK_BYTES, xq += QK) {
        SM_AUPD(0, rd_f32(w) * xd[blk], SM_Q8B(w, xq));
        SM_AUPD(1, rd_f32(r1) * xd[blk], SM_Q8B(r1, xq));
        SM_AUPD(2, rd_f32(r2) * xd[blk], SM_Q8B(r2, xq));
        SM_AUPD(3, rd_f32(r3) * xd[blk], SM_Q8B(r3, xq));
    }
    out[0] = SM_AGET(0);
    out[1] = SM_AGET(1);
    out[2] = SM_AGET(2);
    out[3] = SM_AGET(3);
}

SM_TGT
static void SM_TILE(const unsigned char *w, size_t pair_stride, size_t block_stride, const float *xd, const int8_t *xq,
                    uint32_t nb, float *out, size_t stride) {
    const unsigned char *w1 = w + pair_stride;
    const size_t qs = (size_t)nb * QK;
    const int8_t *x0 = xq, *x1 = xq + qs, *x2 = xq + 2 * qs, *x3 = xq + 3 * qs;
    const float *d0 = xd, *d1 = xd + nb, *d2 = xd + 2 * (size_t)nb, *d3 = xd + 3 * (size_t)nb;
    SM_NACC(8);
    uint32_t blk;
    SM_ZACC(8);
    for (blk = 0; blk < nb; ++blk) {
        float ws = rd_f32(w);
        int8x16_t a0 = vld1q_s8((const int8_t *)(w + 4)), a1 = vld1q_s8((const int8_t *)(w + 20));
        SM_AUPD(0, ws * d0[blk], SM_PRODFN(a0, vld1q_s8(x0), a1, vld1q_s8(x0 + 16)));
        SM_AUPD(1, ws * d1[blk], SM_PRODFN(a0, vld1q_s8(x1), a1, vld1q_s8(x1 + 16)));
        SM_AUPD(2, ws * d2[blk], SM_PRODFN(a0, vld1q_s8(x2), a1, vld1q_s8(x2 + 16)));
        SM_AUPD(3, ws * d3[blk], SM_PRODFN(a0, vld1q_s8(x3), a1, vld1q_s8(x3 + 16)));
        ws = rd_f32(w1);
        a0 = vld1q_s8((const int8_t *)(w1 + 4));
        a1 = vld1q_s8((const int8_t *)(w1 + 20));
        SM_AUPD(4, ws * d0[blk], SM_PRODFN(a0, vld1q_s8(x0), a1, vld1q_s8(x0 + 16)));
        SM_AUPD(5, ws * d1[blk], SM_PRODFN(a0, vld1q_s8(x1), a1, vld1q_s8(x1 + 16)));
        SM_AUPD(6, ws * d2[blk], SM_PRODFN(a0, vld1q_s8(x2), a1, vld1q_s8(x2 + 16)));
        SM_AUPD(7, ws * d3[blk], SM_PRODFN(a0, vld1q_s8(x3), a1, vld1q_s8(x3 + 16)));
        w += block_stride;
        w1 += block_stride;
        x0 += QK;
        x1 += QK;
        x2 += QK;
        x3 += QK;
    }
    out[0] = SM_AGET(0);
    out[stride] = SM_AGET(1);
    out[2 * stride] = SM_AGET(2);
    out[3 * stride] = SM_AGET(3);
    out[1] = SM_AGET(4);
    out[stride + 1] = SM_AGET(5);
    out[2 * stride + 1] = SM_AGET(6);
    out[3 * stride + 1] = SM_AGET(7);
}

SM_TGT
static float SM_Q5DOT(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    SM_NACC(1);
    uint32_t blk;
    SM_ZACC(1);
    for (blk = 0; blk < nb; ++blk, w += Q5_BLOCK_BYTES, xq += QK)
        SM_AUPD(0, f16_to_f32(sm_u16n(w)) * xd[blk], SM_Q5B(w, xq));
    return SM_AGET(0);
}

SM_TGT
static void SM_Q5X4F(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                     float out[4]) {
    const size_t qs = (size_t)stride * QK;
    const int8_t *x1 = xq + qs, *x2 = xq + 2 * qs, *x3 = xq + 3 * qs;
    const float *d1 = xd + stride, *d2 = xd + 2 * (size_t)stride, *d3 = xd + 3 * (size_t)stride;
    SM_NACC(4);
    uint32_t blk;
    SM_ZACC(4);
    for (blk = 0; blk < nb; ++blk, w += Q5_BLOCK_BYTES, xq += QK, x1 += QK, x2 += QK, x3 += QK) {
        int8x16_t e0, e1;
        float ws = f16_to_f32(sm_u16n(w));
        sm_unpack_neon(w + 2, &e0, &e1);
        SM_AUPD(0, ws * xd[blk], SM_PRODFN(e0, vld1q_s8(xq), e1, vld1q_s8(xq + 16)));
        SM_AUPD(1, ws * d1[blk], SM_PRODFN(e0, vld1q_s8(x1), e1, vld1q_s8(x1 + 16)));
        SM_AUPD(2, ws * d2[blk], SM_PRODFN(e0, vld1q_s8(x2), e1, vld1q_s8(x2 + 16)));
        SM_AUPD(3, ws * d3[blk], SM_PRODFN(e0, vld1q_s8(x3), e1, vld1q_s8(x3 + 16)));
    }
    out[0] = SM_AGET(0);
    out[1] = SM_AGET(1);
    out[2] = SM_AGET(2);
    out[3] = SM_AGET(3);
}

SM_TGT
static void SM_Q5R4F(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                     float out[4]) {
    const unsigned char *r1 = w + row_bytes, *r2 = r1 + row_bytes, *r3 = r2 + row_bytes;
    SM_NACC(4);
    uint32_t blk;
    SM_ZACC(4);
    for (blk = 0; blk < nb; ++blk, w += Q5_BLOCK_BYTES, r1 += Q5_BLOCK_BYTES, r2 += Q5_BLOCK_BYTES,
        r3 += Q5_BLOCK_BYTES, xq += QK) {
        SM_AUPD(0, f16_to_f32(sm_u16n(w)) * xd[blk], SM_Q5B(w, xq));
        SM_AUPD(1, f16_to_f32(sm_u16n(r1)) * xd[blk], SM_Q5B(r1, xq));
        SM_AUPD(2, f16_to_f32(sm_u16n(r2)) * xd[blk], SM_Q5B(r2, xq));
        SM_AUPD(3, f16_to_f32(sm_u16n(r3)) * xd[blk], SM_Q5B(r3, xq));
    }
    out[0] = SM_AGET(0);
    out[1] = SM_AGET(1);
    out[2] = SM_AGET(2);
    out[3] = SM_AGET(3);
}

SM_TGT
void SM_G8(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows, uint32_t ntok,
           uint32_t nb, float *out, size_t stride, unsigned char *scratch) {
    unsigned char *pack = scratch ? sm_align64n(scratch) : NULL;
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
        if (pack && ntok >= 8 && row_bytes > 16u * 1024u) {
            sm_interleave2n(a, b, pack, nb);
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
            sm_store4n(out, stride, t, r, y);
        }
        for (; t < ntok; ++t)
            out[(size_t)t * stride + r] = SM_DOT(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
    }
}

SM_TGT
void SM_G5(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nrows, uint32_t ntok,
           uint32_t nb, float *out, size_t stride, unsigned char *scratch) {
    unsigned char *pack = scratch ? sm_align64n(scratch) : NULL;
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
        const unsigned char *a = w + (size_t)r * row_bytes, *brow = a + row_bytes;
        if (pack && ntok >= 8 && (size_t)nb * Q8_BLOCK_BYTES > 16u * 1024u) {
            sm_q5_interleave2n(a, brow, pack, nb);
            for (t = 0; t + 4 <= ntok; t += 4)
                SM_TILE(pack, Q8_BLOCK_BYTES, 2 * Q8_BLOCK_BYTES, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb,
                        out + (size_t)t * stride + r, stride);
        } else {
            for (t = 0; t + 4 <= ntok; t += 4) {
                float y[4];
                SM_Q5X4F(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, nb, y);
                sm_store4n(out, stride, t, r, y);
                SM_Q5X4F(brow, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, nb, y);
                sm_store4n(out, stride, t, r + 1, y);
            }
        }
        for (t = ntok - (ntok % 4); t < ntok; ++t) {
            out[(size_t)t * stride + r] = SM_Q5DOT(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
            out[(size_t)t * stride + r + 1] = SM_Q5DOT(brow, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
        }
    }
    if (r < nrows) {
        const unsigned char *a = w + (size_t)r * row_bytes;
        for (t = 0; t + 4 <= ntok; t += 4) {
            float y[4];
            SM_Q5X4F(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb, nb, y);
            sm_store4n(out, stride, t, r, y);
        }
        for (; t < ntok; ++t)
            out[(size_t)t * stride + r] = SM_Q5DOT(a, xd + (size_t)t * nb, xq + (size_t)t * nb * QK, nb);
    }
}

#if !defined(SM_SKIP_DOT)
SM_TGT float SM_Q8(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    return SM_DOT(w, xd, xq, nb);
}
SM_TGT void SM_Q8X4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                    float out[4]) {
    SM_X4F(w, xd, xq, nb, stride, out);
}
SM_TGT float SM_Q5(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb) {
    return SM_Q5DOT(w, xd, xq, nb);
}
SM_TGT void SM_Q5X4(const unsigned char *w, const float *xd, const int8_t *xq, uint32_t nb, uint32_t stride,
                    float out[4]) {
    SM_Q5X4F(w, xd, xq, nb, stride, out);
}
#endif

SM_TGT void SM_Q8R4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                    float out[4]) {
    SM_R4F(w, row_bytes, xd, xq, nb, out);
}
SM_TGT void SM_Q5R4(const unsigned char *w, size_t row_bytes, const float *xd, const int8_t *xq, uint32_t nb,
                    float out[4]) {
    SM_Q5R4F(w, row_bytes, xd, xq, nb, out);
}

#undef SM_NACC
#undef SM_ZACC
#undef SM_AUPD
#undef SM_AGET
#undef SM_Q8B
#undef SM_Q5B
#undef SM_PRODFN
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
#undef SM_KIND
#undef SM_FAST
#undef SM_SKIP_DOT
#undef SM_TARGET
#ifdef SM_TGT_LOCAL
#  undef SM_TGT_LOCAL
#  undef SM_TGT
#else
#  undef SM_TGT
#endif
