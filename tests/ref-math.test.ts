/**
 * Unit tests for the math pieces of the independent Phi-3/Phi-4-mini reference
 * (scripts/ref). Every expected value is computed by hand in the comments, not
 * by calling the code under test.
 */
import { describe, expect, test } from "bun:test";

import { bf16ToNumber, decodeBf16 } from "../scripts/ref/bf16";
import {
  applyRope,
  attentionFactor,
  invFreq,
  kvHeadFor,
  rmsNorm,
  ropeTable,
  selectRopeFactors,
  silu,
  softmax,
} from "../scripts/ref/math";

const close = (got: ArrayLike<number>, want: readonly number[], digits = 12): void => {
  expect(got.length).toBe(want.length);
  want.forEach((w, i) => expect(got[i]!).toBeCloseTo(w, digits));
};

describe("bf16 decoding (bits << 16 = float32 bits)", () => {
  test("hand-decoded values", () => {
    expect(bf16ToNumber(0x3f80)).toBe(1); // sign 0, exp 127, mantissa 0
    expect(bf16ToNumber(0xc000)).toBe(-2); // sign 1, exp 128
    expect(bf16ToNumber(0x3f81)).toBe(1 + 2 ** -7); // 7-bit mantissa, lowest bit set
    expect(bf16ToNumber(0x4049)).toBe(3.140625); // 1.5703125 * 2
    expect(bf16ToNumber(0x0000)).toBe(0);
    expect(Object.is(bf16ToNumber(0x8000), -0)).toBe(true);
    expect(bf16ToNumber(0x7f80)).toBe(Infinity);
    expect(bf16ToNumber(0xff80)).toBe(-Infinity);
    expect(Number.isNaN(bf16ToNumber(0x7fc0))).toBe(true);
    expect(bf16ToNumber(0x0001)).toBe(2 ** -133); // smallest subnormal: 2^-149 * 2^16
  });

  test("bulk decode matches scalar decode and returns a new array", () => {
    const src = new Uint16Array([0x3f80, 0xc000, 0x3f81, 0x8000, 0x0001, 0x4049]);
    const out = decodeBf16(src);
    expect(out).toBeInstanceOf(Float32Array);
    expect(Array.from(out)).toEqual(Array.from(src, bf16ToNumber));
    expect(src[0]).toBe(0x3f80);
  });

  test("rejects out-of-range bits", () => {
    expect(() => bf16ToNumber(0x10000)).toThrow();
    expect(() => bf16ToNumber(-1)).toThrow();
  });
});

describe("RMSNorm: w * x / sqrt(mean(x^2) + eps), no mean subtraction", () => {
  test("x = [3, 4], w = [1, 2], eps = 0", () => {
    // mean(x^2) = (9 + 16) / 2 = 12.5; 1 / sqrt(12.5) = 0.282842712474619
    close(rmsNorm(new Float64Array([3, 4]), [1, 2], 0), [0.848528137423857, 2.262741699796952]);
  });

  test("eps goes inside the square root", () => {
    // mean(x^2) = 1, sqrt(1 + 3) = 2, so y = 1 / 2 (outside would give 1 / (1 + 3) or 1 / 1 + 3)
    close(rmsNorm(new Float64Array([1, -1]), [1, 1], 3), [0.5, -0.5]);
  });

  test("gain is w itself (not 1 + w) and the input is not modified", () => {
    const x = new Float64Array([2, 2, 2, 2]); // rms = 2
    close(rmsNorm(x, [0, 1, -1, 0.5], 0), [0, 1, -1, 0.5]);
    expect(Array.from(x)).toEqual([2, 2, 2, 2]);
  });

  test("rejects length mismatch", () => {
    expect(() => rmsNorm(new Float64Array(3), [1, 1], 0)).toThrow();
  });
});

describe("softmax", () => {
  test("exp values 1, 2, 3 normalise to 1/6, 2/6, 3/6", () => {
    close(softmax(new Float64Array([0, Math.log(2), Math.log(3)])), [1 / 6, 2 / 6, 3 / 6]);
  });

  test("is shift-invariant and stable for large inputs", () => {
    close(softmax(new Float64Array([1000, 1000])), [0.5, 0.5]);
    close(softmax(new Float64Array([-1000, -1000 + Math.log(3)])), [0.25, 0.75]);
  });

  test("a single score gives probability 1", () => {
    close(softmax(new Float64Array([-7.5])), [1]);
  });
});

describe("SiLU", () => {
  test("x * sigmoid(x)", () => {
    expect(silu(0)).toBe(0);
    expect(silu(Math.log(3))).toBeCloseTo(Math.log(3) * 0.75, 14); // sigmoid(ln 3) = 3 / 4
    expect(silu(-Math.log(3))).toBeCloseTo(-Math.log(3) * 0.25, 14);
  });
});

describe("GQA mapping (repeat_kv: expanded index g * n_rep + r)", () => {
  test("24 query heads over 8 KV heads: h -> floor(h / 3)", () => {
    const got = Array.from({ length: 24 }, (_, h) => kvHeadFor(h, 24, 8));
    expect(got).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7]);
  });

  test("grouping is contiguous, not interleaved (head 1 uses KV 0, not KV 1)", () => {
    expect(kvHeadFor(1, 24, 8)).toBe(0);
    expect(kvHeadFor(8, 24, 8)).toBe(2);
  });

  test("rejects bad shapes", () => {
    expect(() => kvHeadFor(0, 24, 7)).toThrow();
    expect(() => kvHeadFor(24, 24, 8)).toThrow();
  });
});

describe("LongRoPE parameters", () => {
  test("attention factor = sqrt(1 + ln(131072/4096) / ln 4096) = sqrt(17/12)", () => {
    // ln 32 / ln 4096 = 5 ln 2 / 12 ln 2 = 5/12
    expect(attentionFactor(131072, 4096)).toBeCloseTo(Math.sqrt(17 / 12), 15);
    expect(attentionFactor(131072, 4096)).toBeCloseTo(1.1902380714238083, 15);
  });

  test("attention factor is 1 when there is no extension", () => {
    expect(attentionFactor(4096, 4096)).toBe(1);
  });

  test("inv_freq over 96 rotary dims with unit factors: 10000^(-2i/96)", () => {
    const f = invFreq(10000, 96, new Array(48).fill(1));
    expect(f.length).toBe(48);
    expect(f[0]).toBe(1);
    expect(f[12]).toBeCloseTo(0.1, 15); // 10000^(-24/96) = 10000^(-1/4)
    expect(f[24]).toBeCloseTo(0.01, 15); // 10000^(-1/2)
    expect(f[36]).toBeCloseTo(0.001, 15); // 10000^(-3/4)
    expect(f[1]).toBeCloseTo(0.8254041853, 9); // 10000^(-1/48), spec appendix A
  });

  test("inv_freq divides by the extension factor", () => {
    const f = invFreq(10000, 4, [2, 4]);
    close(f, [0.5, 0.01 / 4]); // dim 4: exponents 0/4 and 2/4 -> 1 and 1/100
  });

  test("inv_freq rejects a factor list of the wrong length", () => {
    expect(() => invFreq(10000, 96, [1, 1])).toThrow();
  });

  test("factor choice: short while max position + 1 <= original max, else long", () => {
    const short = [1, 1];
    const long = [2, 3];
    expect(selectRopeFactors(15, 4096, short, long)).toBe(short);
    expect(selectRopeFactors(4095, 4096, short, long)).toBe(short); // seq_len 4096 is not > 4096
    expect(selectRopeFactors(4096, 4096, short, long)).toBe(long); // seq_len 4097
  });
});

describe("partial rotary RoPE, split-half rotate_half, attention factor on cos and sin", () => {
  // head_dim 6, rotary_dim 4: pairs (x0, x2) with theta0 and (x1, x3) with theta1; x4, x5 pass through.
  const x = new Float64Array([1, 2, 3, 4, 5, 6]);

  test("hand-computed rotation at position 1", () => {
    // theta0 = pi/2 -> cos 0, sin 1; theta1 = pi -> cos -1, sin 0; A = 2.
    // y0 = A(x0 c0 - x2 s0) = -6, y2 = A(x2 c0 + x0 s0) = 2
    // y1 = A(x1 c1 - x3 s1) = -4, y3 = A(x3 c1 + x1 s1) = -8
    const table = ropeTable([1], new Float64Array([Math.PI / 2, Math.PI]), 2);
    close(applyRope(x, table.cos.subarray(0, 2), table.sin.subarray(0, 2)), [-6, -4, 2, -8, 5, 6]);
  });

  test("position 0: rotary dims are scaled by A, pass-through dims untouched", () => {
    const table = ropeTable([0], new Float64Array([0.3, 0.7]), 1.5);
    close(applyRope(x, table.cos.subarray(0, 2), table.sin.subarray(0, 2)), [1.5, 3, 4.5, 6, 5, 6]);
  });

  test("table layout is [position][frequency] with A folded into cos and sin", () => {
    const table = ropeTable([0, 2], new Float64Array([0.25, 1]), 3);
    expect(table.half).toBe(2);
    close(table.cos, [3, 3, 3 * Math.cos(0.5), 3 * Math.cos(2)]);
    close(table.sin, [0, 0, 3 * Math.sin(0.5), 3 * Math.sin(2)]);
  });

  test("returns a new array and leaves the input untouched", () => {
    const table = ropeTable([5], new Float64Array([0.1, 0.2]), 1);
    const y = applyRope(x, table.cos, table.sin);
    expect(y).not.toBe(x);
    expect(Array.from(x)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  test("rejects a rotary span longer than the head", () => {
    expect(() => applyRope(new Float64Array(4), new Float64Array(3), new Float64Array(3))).toThrow();
  });
});
