/**
 * Block quantization, bit-identical to tools/q8pack.c (Q8) and src/f16.h.
 *
 * Q8 block of 32: f32 scale = f32(max|v| / 127); q = clamp(rint(double(v) /
 * double(scale)), -127, 127), 0 when the scale is 0; padding q = 0. JS has no
 * FMA and evaluates in double, matching C99 built with -ffp-contract=off.
 */
import { Q8_BLOCK_BYTES, QK } from "./format";

const F32_MAX = 3.4028234663852886e38;
const ROUND_MAGIC = 6755399441055744; /* 1.5 * 2^52: x + M - M rounds to nearest even for |x| < 2^51 */
const ROUND_LIMIT = 2 ** 51;

const scratchF32 = new Float32Array(1);
const scratchU32 = new Uint32Array(scratchF32.buffer);

/** nearbyint/lrint under the default rounding mode: round half to even. */
export function rint(x: number): number {
  if (!(Math.abs(x) < ROUND_LIMIT)) return x; /* already an integer (or non-finite) */
  return x + ROUND_MAGIC - ROUND_MAGIC;
}

export function bf16ToF32(u: number): number {
  scratchU32[0] = (u & 0xffff) << 16;
  return scratchF32[0]!;
}

/** Port of f16_to_f32 (src/f16.h). */
export function f16ToF32(h: number): number {
  const sign = (h & 0x8000) << 16;
  const exp = (h >> 10) & 0x1f;
  let mant = h & 0x3ff;
  let bits: number;
  if (exp === 0x1f) bits = sign | 0x7f800000 | (mant << 13);
  else if (exp) bits = sign | ((exp + 112) << 23) | (mant << 13);
  else if (mant) {
    let e = 113;
    while (!(mant & 0x400)) { mant <<= 1; --e; }
    bits = sign | (e << 23) | ((mant & 0x3ff) << 13);
  } else bits = sign;
  scratchU32[0] = bits >>> 0;
  return scratchF32[0]!;
}

/** Port of f32_to_f16 (src/f16.h): round to nearest even, overflow to inf. */
export function f32ToF16(x: number): number {
  scratchF32[0] = x;
  const bits = scratchU32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  const biased = (bits >>> 23) & 0xff;
  const exp = biased - 127 + 15;
  let mant = bits & 0x7fffff;
  if (biased === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  if (exp >= 0x1f) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - exp;
    let value = mant >>> shift;
    const rest = mant & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rest > half || (rest === half && (value & 1))) ++value;
    return sign | value;
  }
  let value = (exp << 10) | (mant >>> 13);
  const rest = mant & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && (value & 1))) ++value;
  return sign | value;
}

export const q8RowBytes = (cols: number): number => Math.ceil(cols / QK) * Q8_BLOCK_BYTES;

/** Quantizes x[xOff .. xOff+cols) into out[oOff ..), q8RowBytes(cols) bytes. Rejects non-finite input. */
export function quantizeQ8Row(x: Float32Array, xOff: number, cols: number, out: Uint8Array, view: DataView,
  oOff: number): void {
  for (let i = 0, p = oOff; i < cols; i += QK, p += Q8_BLOCK_BYTES) {
    const n = cols - i < QK ? cols - i : QK;
    const base = xOff + i;
    let high = 0;
    for (let j = 0; j < n; j++) {
      const a = Math.abs(x[base + j]!);
      if (!(a <= F32_MAX)) throw new Error("non-finite source value");
      if (a > high) high = a;
    }
    const scale = Math.fround(high / 127);
    view.setFloat32(p, scale, true);
    const q = p + 4;
    if (scale === 0) {
      out.fill(0, q, q + QK);
      continue;
    }
    for (let j = 0; j < n; j++) {
      const r = x[base + j]! / scale + ROUND_MAGIC - ROUND_MAGIC;
      out[q + j] = (r > 127 ? 127 : r < -127 ? -127 : r) & 0xff;
    }
    if (n < QK) out.fill(0, q + n, q + QK);
  }
}

/** Quantizes a row-major rows x cols matrix into a fresh buffer. */
export function quantizeQ8(x: Float32Array, rows: number, cols: number): Uint8Array {
  if (x.length < rows * cols) throw new Error("quantizeQ8: input shorter than rows * cols");
  const rowBytes = q8RowBytes(cols);
  const out = new Uint8Array(rows * rowBytes);
  const view = new DataView(out.buffer);
  for (let r = 0; r < rows; r++) quantizeQ8Row(x, r * cols, cols, out, view, r * rowBytes);
  return out;
}

/** weight_j = f32(scale * q_j), as the runtime reads it. */
export function dequantizeQ8Row(bytes: Uint8Array, offset: number, cols: number): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(cols);
  for (let i = 0, p = offset; i < cols; i += QK, p += Q8_BLOCK_BYTES) {
    const scale = view.getFloat32(p, true);
    for (let j = 0; j < QK && i + j < cols; j++) out[i + j] = Math.fround(scale * view.getInt8(p + 4 + j));
  }
  return out;
}

/** Little-endian F32 payload of a vector. */
export function encodeF32(x: Float32Array): Uint8Array {
  const out = new Uint8Array(x.length * 4);
  const view = new DataView(out.buffer);
  for (let i = 0; i < x.length; i++) {
    if (!Number.isFinite(x[i]!)) throw new Error("non-finite source value");
    view.setFloat32(i * 4, x[i]!, true);
  }
  return out;
}
