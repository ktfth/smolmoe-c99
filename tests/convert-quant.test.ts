/**
 * Q8 quantizer of the TypeScript converter: bit-identical to tools/q8pack.c.
 * Goldens in goldens/convert-q8.json were produced by compiling the exact
 * q8pack.c inner loop (gcc -O3 -ffp-contract=off) over the patterns below.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  bf16ToF32, dequantizeQ8Row, f16ToF32, f32ToF16, quantizeQ8, quantizeQ8Row, q8RowBytes, rint,
} from "../scripts/smoe/quant";

interface GoldenCase {
  readonly rows: number;
  readonly cols: number;
  readonly seed: number;
  readonly bytes: number;
  readonly sha256: string;
  readonly firstRowHex?: string;
}

const golden = JSON.parse(readFileSync(join(import.meta.dir, "goldens", "convert-q8.json"), "utf8")) as {
  readonly cases: readonly GoldenCase[];
};

/** Same generator that fed the C reference: finite BF16 bit patterns of six kinds. */
function bf16Patterns(rows: number, cols: number, seed0: number): Uint16Array {
  let seed = seed0 >>> 0;
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed; };
  const tie = new Float32Array(1);
  const tieBits = new Uint32Array(tie.buffer);
  const sample = (row: number, j: number): number => {
    const mode = row % 6;
    if (mode === 0) { for (;;) { const u = rnd() & 0xffff; if (((u >> 7) & 0xff) !== 0xff) return u; } }
    if (mode === 1) { const exp = 0x78 + (rnd() % 6); return ((rnd() & 1) << 15) | (exp << 7) | (rnd() & 0x7f); }
    if (mode === 2) return ((rnd() & 1) << 15) | ((rnd() % 3) << 7) | (rnd() & 0x7f);
    if (mode === 3) {
      if (j % 32 === 0) return 0x42fe;
      tie[0] = ((rnd() % 254) - 127) + 0.5;
      return tieBits[0]! >>> 16;
    }
    if (mode === 4) return rnd() & 1 ? 0x8000 : 0;
    const exp = 0xf0 + (rnd() % 0xe); return ((rnd() & 1) << 15) | (exp << 7) | (rnd() & 0x7f);
  };
  const out = new Uint16Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let j = 0; j < cols; j++) out[r * cols + j] = sample(r, j);
  return out;
}

const sha256 = (b: Uint8Array) => new Bun.CryptoHasher("sha256").update(b).digest("hex");

describe("rint (round half to even, like nearbyint)", () => {
  test.each([
    [0.5, 0], [1.5, 2], [2.5, 2], [3.5, 4], [-0.5, 0], [-1.5, -2], [-2.5, -2], [126.5, 126], [127.5, 128],
    [-127.5, -128], [3.4999999, 3], [3.5000001, 4], [0, 0], [-7, -7], [1e300, 1e300], [2 ** 52 + 1, 2 ** 52 + 1],
  ])("rint(%p) = %p", (x, want) => {
    expect(rint(x) === want || (want === 0 && rint(x) === 0)).toBe(true);
  });
});

describe("Q8 quantizer against the q8pack.c reference", () => {
  for (const c of golden.cases) {
    test(`${c.rows} x ${c.cols} (seed ${c.seed}) is byte-identical`, () => {
      const src = bf16Patterns(c.rows, c.cols, c.seed);
      const x = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) x[i] = bf16ToF32(src[i]!);
      const out = quantizeQ8(x, c.rows, c.cols);
      expect(out.length).toBe(c.bytes);
      if (c.firstRowHex) expect(Buffer.from(out.subarray(0, q8RowBytes(c.cols))).toString("hex")).toBe(c.firstRowHex);
      expect(sha256(out)).toBe(c.sha256);
    });
  }

  test("an all-zero row (synthesized router) is all zero bytes", () => {
    const out = quantizeQ8(new Float32Array(896), 1, 896);
    expect(out.length).toBe(28 * 36);
    expect(out.every((b) => b === 0)).toBe(true);
  });

  test("negative zero never produces a negative scale", () => {
    const out = quantizeQ8(new Float32Array([-0, -0, -0]), 1, 3);
    expect(new DataView(out.buffer).getUint32(0, true)).toBe(0);
  });

  test("padding of a partial block is zero and |q| peaks at 127", () => {
    const x = new Float32Array([1, -2, 0.5, 4, -4]);
    const out = quantizeQ8(x, 1, 5);
    const view = new DataView(out.buffer);
    expect(view.getFloat32(0, true)).toBe(Math.fround(4 / 127));
    expect([...new Int8Array(out.buffer, 4, 5)]).toEqual([32, -64, 16, 127, -127]);
    expect(out.subarray(9).every((b) => b === 0)).toBe(true);
  });

  test("dequantization recovers scale * q", () => {
    const x = new Float32Array([0.25, -0.75, 1]);
    const out = quantizeQ8(x, 1, 3);
    const back = dequantizeQ8Row(out, 0, 3);
    expect(back[2]).toBe(Math.fround(Math.fround(1 / 127) * 127));
  });

  test.each([[Number.NaN], [Number.POSITIVE_INFINITY], [Number.NEGATIVE_INFINITY]])("rejects %p", (bad) => {
    const x = new Float32Array([1, 2, bad, 3]);
    const out = new Uint8Array(36);
    expect(() => quantizeQ8Row(x, 0, 4, out, new DataView(out.buffer), 0)).toThrow(/non-finite/);
  });

  test("row offsets let one buffer hold many rows", () => {
    const x = new Float32Array([1, 2, 3, 4, 5, 6]);
    const whole = quantizeQ8(x, 2, 3);
    const second = quantizeQ8(x.subarray(3), 1, 3);
    expect(Buffer.from(whole.subarray(36)).equals(Buffer.from(second))).toBe(true);
  });
});

describe("half precision helpers (ports of src/f16.h)", () => {
  test.each([
    [1, 0x3c00], [-2, 0xc000], [65504, 0x7bff], [65520, 0x7c00], [1e-8, 0x0000], [5.960464477539063e-8, 0x0001],
    [1 + 2 ** -11, 0x3c00], [1 + 3 * 2 ** -11, 0x3c02], [Number.POSITIVE_INFINITY, 0x7c00], [-0, 0x8000],
  ])("f32ToF16(%p) = %p", (x, h) => {
    expect(f32ToF16(x)).toBe(h);
  });

  test("f16ToF32 inverts exact values, subnormals and specials", () => {
    expect(f16ToF32(0x3c00)).toBe(1);
    expect(f16ToF32(0x0001)).toBe(2 ** -24);
    expect(f16ToF32(0x03ff)).toBe(1023 * 2 ** -24);
    expect(f16ToF32(0x7c00)).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(f16ToF32(0x7e00))).toBe(true);
    expect(Object.is(f16ToF32(0x8000), -0)).toBe(true);
    for (let h = 0; h < 0x7c00; h += 97) expect(f32ToF16(f16ToF32(h))).toBe(h);
  });

  test("bf16ToF32 widens by a 16-bit shift", () => {
    expect(bf16ToF32(0x3f80)).toBe(1);
    expect(bf16ToF32(0xc040)).toBe(-3);
    expect(bf16ToF32(0x42fe)).toBe(127);
  });
});
