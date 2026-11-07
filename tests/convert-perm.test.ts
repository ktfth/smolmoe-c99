/**
 * Q/K row permutations: HF split-half rotary (rotate_half) to the runtime's
 * adjacent pairs, over the whole head (Qwen2) or only the first rotary_dim
 * coordinates (Phi-3 partial rotary), with pass-through dims untouched.
 */
import { describe, expect, test } from "bun:test";

import { permuteHeads, qkRows, rangeRows, ropePairOrder } from "../scripts/hf/permute";

/** HF: rotate pairs (i, i + rot/2) of the first `rot` dims; the rest passes through. */
function hfRotate(x: readonly number[], rot: number, angle: (i: number) => number): number[] {
  const y = [...x];
  const half = rot / 2;
  for (let i = 0; i < half; i++) {
    const c = Math.cos(angle(i)), s = Math.sin(angle(i));
    y[i] = x[i]! * c - x[i + half]! * s;
    y[i + half] = x[i + half]! * c + x[i]! * s;
  }
  return y;
}

/** Runtime: rotate adjacent pairs (2i, 2i+1) of the first `rot` dims. */
function adjacentRotate(z: readonly number[], rot: number, angle: (i: number) => number): number[] {
  const y = [...z];
  for (let i = 0; i < rot / 2; i++) {
    const c = Math.cos(angle(i)), s = Math.sin(angle(i));
    y[2 * i] = z[2 * i]! * c - z[2 * i + 1]! * s;
    y[2 * i + 1] = z[2 * i]! * s + z[2 * i + 1]! * c;
  }
  return y;
}

describe("ropePairOrder", () => {
  test("full rotary equals q8pack.c's formula (Qwen2)", () => {
    const hd = 64;
    const order = ropePairOrder(hd, hd);
    for (let local = 0; local < hd; local++) expect(order[local]).toBe((local % 2) * (hd / 2) + Math.floor(local / 2));
  });

  test("partial rotary (Phi-4-mini: 96 of 128) permutes only the rotary dims", () => {
    const order = ropePairOrder(128, 96);
    for (let n = 0; n < 96; n++) expect(order[n]).toBe((n >> 1) + 48 * (n & 1));
    for (let n = 96; n < 128; n++) expect(order[n]).toBe(n);
    expect([...order.subarray(0, 6)]).toEqual([0, 48, 1, 49, 2, 50]);
    expect(new Set(order).size).toBe(128);
  });

  test("adjacent-pair rotation of the permuted head equals HF's rotation, permuted", () => {
    const order = ropePairOrder(128, 96);
    const x = Array.from({ length: 128 }, (_, i) => Math.sin(i * 1.7) + i / 100);
    const angle = (i: number) => 37 * 10000 ** (-2 * i / 96);
    const reference = hfRotate(x, 96, angle);
    const permuted = adjacentRotate([...order].map((o) => x[o]!), 96, angle);
    permuted.forEach((v, n) => expect(v).toBeCloseTo(reference[order[n]!]!, 12));
  });

  test("q.k is unchanged when Q and K get the same permutation", () => {
    const order = ropePairOrder(16, 12);
    const q = Array.from({ length: 16 }, (_, i) => i - 3), k = Array.from({ length: 16 }, (_, i) => 2 * i + 1);
    const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, v, i) => s + v * b[i]!, 0);
    expect(dot([...order].map((o) => q[o]!), [...order].map((o) => k[o]!))).toBe(dot(q, k));
  });

  test.each([[128, 97], [128, 130], [127, 96], [128, 0]])("rejects head %p with rotary %p", (hd, rot) => {
    expect(() => ropePairOrder(hd, rot)).toThrow(/rotary/);
  });
});

describe("row maps", () => {
  test("qkRows applies the per-head order from a start row (K block of the fused qkv)", () => {
    const rows = qkRows(3072, 8, 128, 96);
    expect(rows.length).toBe(1024);
    expect([...rows.subarray(0, 4)]).toEqual([3072, 3120, 3073, 3121]);
    expect([...rows.subarray(96, 98)]).toEqual([3168, 3169]);
    expect([...rows.subarray(128, 130)]).toEqual([3200, 3248]);
    expect(rows[1023]).toBe(4095);
  });

  test("rangeRows is the identity over a slice (V block, gate/up halves)", () => {
    expect([...rangeRows(8192, 3)]).toEqual([8192, 8193, 8194]);
  });

  test("permuteHeads reorders bias vectors the same way (Qwen q/k bias)", () => {
    const x = Float32Array.from({ length: 8 }, (_, i) => i);
    expect([...permuteHeads(x, 2, 4, 4)]).toEqual([0, 2, 1, 3, 4, 6, 5, 7]);
    expect(() => permuteHeads(x, 3, 4, 4)).toThrow(/heads/);
  });
});
