/**
 * Source element decoding to float32: BF16 (bits << 16, exact), F16 (src/f16.h
 * rule, exact), F32 (as is). Non-finite values are rejected on the raw bits,
 * like q8pack.c's "non-finite source" check.
 */
import { f16ToF32 } from "../smoe/quant";
import { DTYPE_BYTES, type SourceDtype } from "./safetensors";

function bitsView(dst: Float32Array): Uint32Array {
  return new Uint32Array(dst.buffer, dst.byteOffset, dst.length);
}

/** Decodes `cols` elements starting at byte `at` of `src` into dst[0 .. cols). */
export function decodeRow(dtype: SourceDtype, src: Uint8Array, at: number, cols: number, dst: Float32Array): void {
  if (dst.length < cols) throw new Error("decodeRow: destination too short");
  const start = src.byteOffset + at;
  const aligned = start % DTYPE_BYTES[dtype] === 0;
  if (dtype === "BF16") {
    const out = bitsView(dst);
    if (aligned) {
      const u = new Uint16Array(src.buffer, start, cols);
      for (let j = 0; j < cols; j++) {
        const b = u[j]!;
        if ((b & 0x7f80) === 0x7f80) throw new Error("non-finite source value");
        out[j] = b << 16;
      }
    } else {
      for (let j = 0; j < cols; j++) {
        const b = src[at + 2 * j]! | (src[at + 2 * j + 1]! << 8);
        if ((b & 0x7f80) === 0x7f80) throw new Error("non-finite source value");
        out[j] = b << 16;
      }
    }
    return;
  }
  const view = new DataView(src.buffer, start, cols * DTYPE_BYTES[dtype]);
  if (dtype === "F16") {
    for (let j = 0; j < cols; j++) {
      const h = view.getUint16(2 * j, true);
      if ((h & 0x7c00) === 0x7c00) throw new Error("non-finite source value");
      dst[j] = f16ToF32(h);
    }
    return;
  }
  const out = bitsView(dst);
  for (let j = 0; j < cols; j++) {
    const u = view.getUint32(4 * j, true);
    if ((u & 0x7f800000) === 0x7f800000) throw new Error("non-finite source value");
    out[j] = u;
  }
}

/** Decodes a whole 1-D tensor's bytes. */
export function decodeVector(dtype: SourceDtype, bytes: Uint8Array): Float32Array {
  const n = bytes.length / DTYPE_BYTES[dtype];
  const out = new Float32Array(n);
  decodeRow(dtype, bytes, 0, n, out);
  return out;
}
