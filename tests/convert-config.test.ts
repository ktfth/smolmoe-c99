/**
 * config.ini writer/parser of the TypeScript converter: Python-identical text
 * for the shipped Qwen config, and a port of src/config.c's validation.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  type ModelConfig, expectedTensorCount, formatConfig, parseConfig, pyFloatRepr, tensorSpecs, validateConfig,
} from "../scripts/smoe/config";

const ROOT = join(import.meta.dir, "..");

const QWEN: ModelConfig = {
  architecture: "smolmoe_qwen2", dim: 896, hidden_dim: 4864, layers: 24, heads: 14, kv_heads: 2, experts: 1,
  top_k: 1, vocab_size: 151936, max_context: 32768, shared_experts: 0, bos_id: 256, eos_id: 257, rms_eps: 1e-6,
  rope_theta: 1000000,
};

const PHI: ModelConfig = {
  architecture: "smolmoe_v1", dim: 3072, hidden_dim: 8192, layers: 32, heads: 24, kv_heads: 8, experts: 1,
  top_k: 1, vocab_size: 200064, max_context: 4096, shared_experts: 0, bos_id: 256, eos_id: 257, rms_eps: 1e-5,
  rope_theta: 10000, rotary_dim: 96, rope_attention_factor: Math.sqrt(1 + Math.log(32) / Math.log(4096)),
};

const GEMMA: ModelConfig = {
  architecture: "smolmoe_gemma3", dim: 1152, hidden_dim: 6912, layers: 26, heads: 4, kv_heads: 1, experts: 1,
  top_k: 1, vocab_size: 262144, max_context: 32768, shared_experts: 0, bos_id: 256, eos_id: 257, rms_eps: 1e-6,
  rope_theta: 1000000, head_dim: 256, sliding_window: 512, rope_local_theta: 10000,
};

describe("pyFloatRepr (Python repr of a float)", () => {
  test.each([
    [1e-6, "1e-06"], [1e-5, "1e-05"], [1000000, "1000000.0"], [10000, "10000.0"], [1.5e-7, "1.5e-07"],
    [0.0001, "0.0001"], [1e16, "1e+16"], [123.456, "123.456"], [5e-324, "5e-324"], [1e22, "1e+22"], [0.1, "0.1"],
    [1.25e-5, "1.25e-05"], [9999999999999998, "9999999999999998.0"], [500000, "500000.0"], [-2.5, "-2.5"],
    [0, "0.0"], [-0, "-0.0"], [1.1902380714238083, "1.1902380714238083"],
  ])("%p -> %p", (x, want) => {
    expect(pyFloatRepr(x)).toBe(want);
  });

  test("rejects non-finite values", () => {
    expect(() => pyFloatRepr(Number.NaN)).toThrow(/non-finite/);
  });

  test("the shortest repr reads back to the identical double", () => {
    for (const x of [1.1902380714238083, Math.PI, 1 / 3, 1e-5, 0.1 + 0.2]) expect(Number(pyFloatRepr(x))).toBe(x);
  });
});

describe("formatConfig", () => {
  test("reproduces the shipped Qwen config.ini byte for byte", () => {
    const shipped = readFileSync(join(ROOT, "models", "qwen2.5-0.5b-instruct", "config.ini"), "utf8");
    expect(formatConfig(QWEN)).toBe(shipped);
  });

  test("appends the optional RoPE keys after the 15 required ones", () => {
    const text = formatConfig(PHI);
    expect(text.split("\n").slice(-3)).toEqual(["rotary_dim=96", "rope_attention_factor=1.1902380714238083", ""]);
    expect(text).toContain("rms_eps=1e-05\nrope_theta=10000.0\n");
  });

  test("refuses an invalid config before writing", () => {
    expect(() => formatConfig({ ...QWEN, heads: 13 })).toThrow(/invalid config/);
  });
});

describe("parseConfig (port of src/config.c)", () => {
  test("round-trips Qwen, Phi, and Gemma 3 configs", () => {
    expect(parseConfig(formatConfig(QWEN))).toEqual(QWEN);
    expect(parseConfig(formatConfig(PHI))).toEqual(PHI);
    expect(parseConfig(formatConfig(GEMMA))).toEqual(GEMMA);
  });

  test("accepts comments, blank lines, CRLF and any key order", () => {
    const lines = formatConfig(QWEN).trim().split("\n").reverse();
    const text = `# header\n\n${lines.map((l) => `  ${l.replace("=", " = ")}  # note\r`).join("\n")}\n`;
    expect(parseConfig(text)).toEqual(QWEN);
  });

  test.each([
    ["unknown key", "extra=1\n", /unknown config key: extra/],
    ["duplicate key", "dim=896\n", /duplicate config key: dim/],
  ])("rejects an %s", (_what, line, error) => {
    expect(() => parseConfig(formatConfig(QWEN) + line)).toThrow(error);
  });

  test("rejects a missing key, a line without '=', NUL bytes and long lines", () => {
    expect(() => parseConfig(formatConfig(QWEN).replace("top_k=1\n", ""))).toThrow(/missing required fields/);
    expect(() => parseConfig(`${formatConfig(QWEN)}oops\n`)).toThrow(/key=value/);
    expect(() => parseConfig(`${formatConfig(QWEN)}#\0\n`)).toThrow(/NUL/);
    expect(() => parseConfig(`${formatConfig(QWEN)}#${"x".repeat(510)}\n`)).toThrow(/line too long/);
    expect(() => parseConfig(`${formatConfig(QWEN)}#${"x".repeat(509)}\n`)).not.toThrow();
  });

  test.each([
    ["dim=-1", /invalid dim/], ["dim=+5", /invalid dim/], ["dim=0x10", /invalid dim/], ["dim=1048577", /invalid dim/],
    ["rms_eps=abc", /invalid rms_eps/], ["rms_eps=inf", /invalid rms_eps/], ["rope_theta=1e999", /invalid rope_theta/],
    ["architecture=llama", /unsupported architecture: llama/],
  ])("rejects %p", (line, error) => {
    const key = line.split("=")[0]!;
    const text = formatConfig(QWEN).split("\n").map((l) => (l.startsWith(`${key}=`) ? line : l)).join("\n");
    expect(() => parseConfig(text)).toThrow(error);
  });

  test.each([
    ["heads", 13], ["kv_heads", 3], ["experts", 0], ["top_k", 2], ["vocab_size", 257], ["bos_id", 1],
    ["max_context", 0], ["layers", 1025], ["shared_experts", 17],
  ])("rejects %s=%p as invalid dimensions", (key, value) => {
    expect(() => validateConfig({ ...QWEN, [key]: value })).toThrow(/invalid config/);
  });

  test("rms_eps and rope_theta are checked after the cast to float", () => {
    expect(() => validateConfig({ ...QWEN, rms_eps: 1e-50 })).toThrow(/invalid config/);
    expect(() => validateConfig({ ...QWEN, rope_theta: 1e39 })).toThrow(/invalid config/);
  });

  test("optional RoPE keys: rotary_dim even and within the head, factor finite and positive", () => {
    expect(() => validateConfig({ ...PHI, rotary_dim: 95 })).toThrow(/rotary_dim/);
    expect(() => validateConfig({ ...PHI, rotary_dim: 130 })).toThrow(/rotary_dim/);
    expect(() => validateConfig({ ...PHI, rotary_dim: 0 })).toThrow(/rotary_dim/);
    expect(() => validateConfig({ ...PHI, rope_attention_factor: 0 })).toThrow(/rope_attention_factor/);
    expect(() => validateConfig({ ...PHI, rope_attention_factor: Number.NaN })).toThrow(/rope_attention_factor/);
    expect(() => validateConfig({ ...PHI, rope_attention_factor: 1000.5 })).toThrow(/rope_attention_factor/);
    expect(validateConfig({ ...PHI, rope_attention_factor: 1000 }).rope_attention_factor).toBe(1000);
    expect(validateConfig({ ...PHI, rotary_dim: 128 }).rotary_dim).toBe(128);
  });
});

describe("tensor set", () => {
  test("Qwen has 315 sorted tensors with biases; Phi 323 without", () => {
    expect(expectedTensorCount(QWEN)).toBe(315);
    expect(expectedTensorCount(PHI)).toBe(323);
    const q = tensorSpecs(QWEN);
    expect(q.length).toBe(315);
    expect(q.slice(0, 4).map((t) => t.name)).toEqual(["embed", "final_norm", "l.0.attn_norm", "l.0.e.0.down"]);
    expect(q.at(-1)!.name).toBe("lm_head");
    for (let i = 1; i < q.length; i++) expect(q[i - 1]!.name < q[i]!.name).toBe(true);
    expect(q.find((t) => t.name === "l.3.k_bias")).toEqual({ name: "l.3.k_bias", rows: 1, cols: 128, kind: "vector" });
    expect(tensorSpecs(PHI).some((t) => t.name.endsWith("_bias"))).toBe(false);
    expect(tensorSpecs(PHI).find((t) => t.name === "l.0.k")).toEqual({ name: "l.0.k", rows: 1024, cols: 3072, kind: "matrix" });
    expect(expectedTensorCount(GEMMA)).toBe(3 + 26 * 14);
    expect(tensorSpecs(GEMMA).find((t) => t.name === "l.0.q")).toEqual({ name: "l.0.q", rows: 1024, cols: 1152, kind: "matrix" });
    expect(tensorSpecs(GEMMA).find((t) => t.name === "l.0.o")).toEqual({ name: "l.0.o", rows: 1152, cols: 1024, kind: "matrix" });
    expect(tensorSpecs(GEMMA).find((t) => t.name === "l.0.q_norm")).toEqual({ name: "l.0.q_norm", rows: 1, cols: 256, kind: "vector" });
  });

  test("routed and shared experts appear for MoE configs", () => {
    const moe: ModelConfig = { ...QWEN, architecture: "smolmoe_v1", experts: 4, top_k: 2, shared_experts: 1 };
    const names = new Set(tensorSpecs(moe).map((t) => t.name));
    for (const n of ["l.0.e.3.up", "l.0.s.0.down", "l.23.router"]) expect(names.has(n)).toBe(true);
    expect(tensorSpecs(moe).find((t) => t.name === "l.0.router")).toMatchObject({ rows: 4, cols: 896 });
  });
});
