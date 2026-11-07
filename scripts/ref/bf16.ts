/**
 * bfloat16 decoding for the independent reference. A bf16 value is the upper
 * 16 bits of an IEEE-754 float32, so decoding is exact: shift the 16 bits into
 * the high half of a float32 and read it. Every bf16 value is representable in
 * float32 and therefore in float64.
 */

const scratchU32 = new Uint32Array(1);
const scratchF32 = new Float32Array(scratchU32.buffer);

/** Decodes one bf16 bit pattern (0..0xffff) to a number. */
export function bf16ToNumber(bits: number): number {
  if (!Number.isInteger(bits) || bits < 0 || bits > 0xffff) {
    throw new RangeError(`bf16 bits out of range: ${bits}`);
  }
  scratchU32[0] = bits << 16;
  return scratchF32[0]!;
}

/** Decodes a run of bf16 bit patterns into a new Float32Array (exact). */
export function decodeBf16(src: Uint16Array): Float32Array {
  const out = new Float32Array(src.length);
  decodeBf16Into(src, out);
  return out;
}

/**
 * Same as decodeBf16, but writes into a caller-owned scratch buffer (used by the
 * weight streamer to avoid allocating a fresh buffer per chunk).
 */
export function decodeBf16Into(src: Uint16Array, dst: Float32Array): void {
  if (dst.length < src.length) throw new RangeError(`bf16 decode: ${src.length} values into ${dst.length}`);
  const dstBits = new Uint32Array(dst.buffer, dst.byteOffset, src.length);
  for (let i = 0; i < src.length; i++) dstBits[i] = src[i]! << 16;
}
