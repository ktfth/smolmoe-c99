/**
 * Unit tests for the layer pieces of the independent Phi-3 reference: causal GQA
 * attention over a fused qkv row, SwiGLU over a fused gate_up row, the
 * position-batched linear kernel, top-k, and config parsing. Expected values are
 * worked out by hand in the comments.
 */
import { describe, expect, test } from "bun:test";

import { causalAttention, qkvLayout, swiGlu } from "../scripts/ref/layers";
import { matmulRows, topK } from "../scripts/ref/linear";
import { parsePhi3Config } from "../scripts/ref/config";
import { ropeTable } from "../scripts/ref/math";

const close = (got: ArrayLike<number>, want: readonly number[], digits = 12): void => {
  expect(got.length).toBe(want.length);
  want.forEach((w, i) => expect(got[i]!).toBeCloseTo(w, digits));
};

describe("qkv layout (HF slices: q | k | v, head-major, 128 dims each)", () => {
  test("Phi-4-mini shape: Q [0,3072), K [3072,4096), V [4096,5120)", () => {
    expect(qkvLayout({ numHeads: 24, numKvHeads: 8, headDim: 128 })).toEqual({
      qStart: 0,
      kStart: 3072,
      vStart: 4096,
      width: 5120,
    });
  });
});

describe("causal attention with GQA and partial rotary", () => {
  // head_dim 4, rotary_dim 2 (one frequency); a zero frequency makes the rotation the identity.
  const identity = (t: number, scale = 1) =>
    ropeTable(Array.from({ length: t }, (_, i) => i), new Float64Array([0]), scale);

  test("position 0 only sees itself; heads 0,1 share KV 0 and heads 2,3 share KV 1", () => {
    const shape = { numHeads: 4, numKvHeads: 2, headDim: 4 };
    const row = new Float64Array(qkvLayout(shape).width); // 4*4 q + 2*4 k + 2*4 v = 32
    row.fill(0.3, 0, 16); // arbitrary queries
    row.fill(1, 24, 28); // v of KV head 0
    row.fill(2, 28, 32); // v of KV head 1
    const out = causalAttention(row, 1, shape, identity(1));
    close(out, [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]);
  });

  test("hand-computed weights at position 1 (scores scaled by 1/sqrt(head_dim))", () => {
    const shape = { numHeads: 2, numKvHeads: 1, headDim: 4 };
    // row width 2*4 + 4 + 4 = 16: q0 [0,4) q1 [4,8) k [8,12) v [12,16)
    const qkv = new Float64Array(32);
    qkv.set([4, 0, 0, 0], 12); // t=0: k = 0, v = [4,0,0,0]
    qkv.set([1, 0, 0, 0], 16); // t=1: q0 = e0, q1 = 0
    qkv.set([2 * Math.log(3), 0, 0, 0], 24); // t=1: k
    qkv.set([0, 8, 0, 0], 28); // t=1: v
    // head 0: scores [0, 2 ln3 / 2] = [0, ln3] -> [1/4, 3/4] -> o = [1, 6, 0, 0]
    // head 1: scores [0, 0] -> [1/2, 1/2] -> o = [2, 4, 0, 0]
    const out = causalAttention(qkv, 2, shape, identity(2));
    close(out.subarray(8, 16), [1, 6, 0, 0, 2, 4, 0, 0]);
    close(out.subarray(0, 8), [4, 0, 0, 0, 4, 0, 0, 0]);
  });

  test("attention factor scales only the rotary dims of q and k", () => {
    const shape = { numHeads: 1, numKvHeads: 1, headDim: 4 };
    const a = Math.log(3) / 4;
    const b = Math.log(3);
    const qkv = new Float64Array(24); // width 12
    qkv.set([0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], 0); // t=0: q 0, k 0, v [4,0,0,0]
    qkv.set([1, 0, 1, 0, a, 0, b, 0, 0, 8, 0, 0], 12); // t=1: q [1,0|1,0], k [a,0|b,0]
    // with A = 2: q.k = (2*1)(2a) + 1*b = 4a + b = 2 ln3; / sqrt(4) = ln3 -> weights [1/4, 3/4]
    const out = causalAttention(qkv, 2, shape, identity(2, 2));
    close(out.subarray(4, 8), [1, 6, 0, 0]);
  });

  test("rejects a buffer that does not match T * width", () => {
    const shape = { numHeads: 2, numKvHeads: 1, headDim: 4 };
    expect(() => causalAttention(new Float64Array(15), 1, shape, identity(1))).toThrow();
  });
});

describe("SwiGLU over the fused gate_up output (gate = first half, up = second half)", () => {
  test("silu(gate) * up per position", () => {
    // row = [g0, g1, u0, u1]; silu(0) = 0, silu(ln3) = 0.75 ln3
    const gateUp = new Float64Array([0, Math.log(3), 5, 4, 1, 1, 0, 2]);
    const out = swiGlu(gateUp, 2, 2);
    const s1 = 1 / (1 + Math.exp(-1));
    close(out, [0, 3 * Math.log(3), 0, 2 * s1]);
  });
});

describe("position-batched linear: out[t][r] = sum_c W[r][c] x[t][c]", () => {
  test("2x3 weight, 2 positions", () => {
    const w = new Float32Array([1, 2, 3, 4, 5, 6]);
    const x = new Float64Array([1, 0, -1, 0.5, 0.5, 0.5]);
    close(matmulRows(w, 2, 3, x, 2), [-2, -2, 3, 7.5]);
  });

  test("5 positions exercise the grouped and remainder paths", () => {
    const w = new Float32Array([1, 2, 3, 4, 5, 6]);
    const x = new Float64Array(Array.from({ length: 5 }, (_, t) => [t, 1, 0]).flat());
    // r0 = t + 2, r1 = 4t + 5
    close(matmulRows(w, 2, 3, x, 5), [2, 5, 3, 9, 4, 13, 5, 17, 6, 21]);
  });

  test("rejects mismatched sizes", () => {
    expect(() => matmulRows(new Float32Array(5), 2, 3, new Float64Array(3), 1)).toThrow();
  });
});

describe("top-k", () => {
  test("descending by logit, ties broken by lower id", () => {
    const got = topK(new Float64Array([0.5, 3, -1, 3, 2]), 0, 5, 3);
    expect(got).toEqual([
      { id: 1, logit: 3 },
      { id: 3, logit: 3 },
      { id: 4, logit: 2 },
    ]);
  });

  test("reads one position out of a flat buffer", () => {
    const got = topK(new Float64Array([9, 9, 1, 7, 2]), 2, 3, 2);
    expect(got).toEqual([
      { id: 1, logit: 7 },
      { id: 2, logit: 2 },
    ]);
  });
});

describe("config parsing", () => {
  const raw = {
    hidden_size: 3072,
    intermediate_size: 8192,
    num_hidden_layers: 32,
    num_attention_heads: 24,
    num_key_value_heads: 8,
    vocab_size: 200064,
    rms_norm_eps: 1e-5,
    rope_theta: 10000,
    partial_rotary_factor: 0.75,
    max_position_embeddings: 131072,
    original_max_position_embeddings: 4096,
    tie_word_embeddings: true,
    hidden_act: "silu",
    rope_scaling: { type: "longrope", short_factor: new Array(48).fill(1), long_factor: new Array(48).fill(2) },
  };

  test("derives head_dim 128, rotary_dim int(128 * 0.75) = 96", () => {
    const cfg = parsePhi3Config(raw);
    expect(cfg.headDim).toBe(128);
    expect(cfg.rotaryDim).toBe(96);
    expect(cfg.numKvHeads).toBe(8);
    expect(cfg.shortFactor.length).toBe(48);
  });

  test("rejects factor lists that do not have rotary_dim / 2 entries", () => {
    const bad = { ...raw, rope_scaling: { ...raw.rope_scaling, short_factor: [1, 1] } };
    expect(() => parsePhi3Config(bad)).toThrow();
  });

  test("rejects untied embeddings and unknown activations (not handled by this reference)", () => {
    expect(() => parsePhi3Config({ ...raw, tie_word_embeddings: false })).toThrow();
    expect(() => parsePhi3Config({ ...raw, hidden_act: "gelu" })).toThrow();
  });
});
