/* IEEE binary16 conversions in portable integer code, so every backend and
 * platform decodes Q4 block scales to exactly the same float. */
#ifndef SMOLMOE_F16_H
#define SMOLMOE_F16_H

#include <stdint.h>
#include <string.h>

/* Exact: every binary16 value is representable in binary32. */
static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, mant = h & 0x3ffu, bits;
    float f;
    if (exp == 0x1f) {
        bits = sign | 0x7f800000u | (mant << 13); /* inf / NaN */
    } else if (exp) {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    } else if (mant) {
        /* Subnormal: normalize. */
        exp = 113;
        while (!(mant & 0x400u)) { mant <<= 1; --exp; }
        bits = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
    } else {
        bits = sign;
    }
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Round to nearest, ties to even; overflow saturates to infinity. */
static inline uint16_t f32_to_f16(float f) {
    uint32_t bits, sign, mant;
    int32_t exp;
    memcpy(&bits, &f, sizeof(bits));
    sign = (bits >> 16) & 0x8000u;
    exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    mant = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        uint32_t shift, half, rest, value;
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        shift = (uint32_t)(14 - exp);
        value = mant >> shift;
        rest = mant & ((1u << shift) - 1);
        half = 1u << (shift - 1);
        if (rest > half || (rest == half && (value & 1u))) ++value;
        return (uint16_t)(sign | value);
    }
    {
        uint32_t value = ((uint32_t)exp << 10) | (mant >> 13), rest = mant & 0x1fffu;
        if (rest > 0x1000u || (rest == 0x1000u && (value & 1u))) ++value; /* may carry into inf: correct */
        return (uint16_t)(sign | value);
    }
}

#endif
