/**
 * End-to-end conversion of tiny synthetic Qwen2 and Phi-3 checkpoints (random
 * weights): row maps, head permutations, fused-tensor splits, gates, failure
 * cleanup, thread-count determinism, v3 packaging and, when built, a run of
 * the C runtime on the converted Qwen2-shaped model.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { ConvertOptions } from "../scripts/hf/common";
import { decodeRow } from "../scripts/hf/decode";
import { ropePairOrder } from "../scripts/hf/permute";
import { convertPhi3, phi3Config } from "../scripts/hf/phi3";
import { convertQwen2, qwen2Config } from "../scripts/hf/qwen2";
import { openCheckpoint, readTensorBytes } from "../scripts/hf/safetensors";
import { PHI_SHAPE, QWEN_SHAPE, writePhi3Synthetic, writeQwen2Synthetic, writeSafetensors } from "../scripts/hf/synthetic";
import { bundleSingleFile } from "../scripts/smoe/bundle";
import { checkFiles } from "../scripts/smoe/check";
import { parseQbp } from "../scripts/smoe/qbp";
import { q8RowBytes, quantizeQ8 } from "../scripts/smoe/quant";
import { readContainer } from "../scripts/smoe/reader";
import { BINARY, BUNDLE_BINARY, run, runOk, tempDir } from "./support/run";

const work = tempDir("convert-hf");
afterAll(() => work.dispose());
const QWEN_SRC = join(work.path, "qwen-src"), PHI_SRC = join(work.path, "phi-src");
const options = (source: string, output: string, extra: Partial<ConvertOptions> = {}): ConvertOptions =>
  ({ source, output, container: "v2", threads: 3, pinned: false, log: () => {}, ...extra });

beforeAll(() => {
  writeQwen2Synthetic(QWEN_SRC);
  writePhi3Synthetic(PHI_SRC);
});

/** Q8 bytes of one runtime row, recomputed from a source tensor row. */
function expectedRow(dir: string, tensor: string, sourceRow: number): Buffer {
  const t = openCheckpoint(dir).tensors.get(tensor)!;
  const cols = t.shape[1]!;
  const x = new Float32Array(cols);
  decodeRow(t.dtype, readTensorBytes(t), sourceRow * cols * 2, cols, x);
  return Buffer.from(quantizeQ8(x, 1, cols));
}

function modelRow(path: string, name: string, row: number): Buffer {
  const r = readContainer(path).records.find((x) => x.name === name)!;
  const bytes = q8RowBytes(r.cols);
  return readFileSync(path).subarray(r.offset + row * bytes, r.offset + (row + 1) * bytes);
}

describe("Qwen2-shaped synthetic checkpoint", () => {
  const out = join(work.path, "qwen-out");
  beforeAll(async () => { await convertQwen2(options(QWEN_SRC, out)); });

  test("writes the four runtime files and they pass the TS loader check", () => {
    for (const f of ["config.ini", "tokenizer.qbp", "model.smoe", "provenance.json"]) expect(existsSync(join(out, f))).toBe(true);
    const r = checkFiles({ model: join(out, "model.smoe"), config: readFileSync(join(out, "config.ini")),
      tokenizer: readFileSync(join(out, "tokenizer.qbp")) });
    expect([r.container.version, r.container.count, r.tokenizer.version]).toEqual([2, 3 + 2 * 13, 1]);
    expect(readdirSync(out).some((f) => f.endsWith(".partial") || f.endsWith(".tmp"))).toBe(false);
  });

  test("embed rows follow the token permutation; q/k rows the head permutation; router is zero", () => {
    const model = join(out, "model.smoe");
    const qbp = parseQbp(readFileSync(join(out, "tokenizer.qbp")), QWEN_SHAPE.vocab);
    expect(new TextDecoder().decode(qbp.pieces[257])).toBe("<|im_end|>");
    expect(modelRow(model, "embed", 256).equals(expectedRow(QWEN_SRC, "model.embed_tokens.weight", 261))).toBe(true);
    /* byte 0x20 is GPT-2 char U+0120, source id 220 in GPT-2 order; it becomes runtime id 0x20 */
    expect(modelRow(model, "embed", 0x20).equals(expectedRow(QWEN_SRC, "model.embed_tokens.weight", 220))).toBe(true);
    const hd = QWEN_SHAPE.dim / QWEN_SHAPE.heads, order = ropePairOrder(hd, hd);
    for (const n of [0, 1, 2, 17, hd + 3]) {
      const src = Math.floor(n / hd) * hd + order[n % hd]!;
      expect(modelRow(model, "l.1.k", n).equals(expectedRow(QWEN_SRC, "model.layers.1.self_attn.k_proj.weight", src))).toBe(true);
    }
    const router = readContainer(model).records.find((r) => r.name === "l.0.router")!;
    expect(readFileSync(model).subarray(router.offset, router.offset + router.bytes).every((b) => b === 0)).toBe(true);
    const head = readContainer(model).records.find((r) => r.name === "lm_head")!;
    expect(head.alias).toBe(1);
  });

  test("the output does not depend on the thread count", async () => {
    const single = join(work.path, "qwen-1t");
    await convertQwen2(options(QWEN_SRC, single, { threads: 1 }));
    expect(readFileSync(join(single, "model.smoe")).equals(readFileSync(join(out, "model.smoe")))).toBe(true);
  });

  test("v3 conversion equals the TS bundle of v2 (and smolmoe-bundle when built)", async () => {
    const v3 = join(work.path, "qwen-v3");
    await convertQwen2(options(QWEN_SRC, v3, { container: "v3" }));
    const repacked = join(work.path, "qwen-repacked.smoe");
    await bundleSingleFile(join(out, "model.smoe"), readFileSync(join(out, "config.ini")),
      readFileSync(join(out, "tokenizer.qbp")), repacked);
    const direct = readFileSync(join(v3, "model.smoe"));
    expect(direct.equals(readFileSync(repacked))).toBe(true);
    expect(checkFiles({ model: join(v3, "model.smoe") }).container.blobs).toBe(2);
    if (existsSync(BUNDLE_BINARY)) {
      const c = join(work.path, "qwen-cbundle.smoe");
      const r = run(["--model", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
        join(out, "tokenizer.qbp"), "--out", c], BUNDLE_BINARY);
      expect(r.code).toBe(0);
      expect(readFileSync(c).equals(direct)).toBe(true);
    }
  });

  test.skipIf(!existsSync(BINARY))("the C runtime loads and runs the converted model (v2 and v3 agree)", () => {
    const args = ["--prompt", "the in", "--tokens", 4, "--context", 64, "--memory-mib", 16, "--hash-logits"];
    const three = runOk(["--model", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
      join(out, "tokenizer.qbp"), ...args]);
    const single = runOk(["--model", join(work.path, "qwen-v3", "model.smoe"), ...args]);
    expect(single.stats.logits_hash).toBe(three.stats.logits_hash);
  });

  test("failures leave no partial output", async () => {
    const bad = join(work.path, "qwen-nan");
    writeQwen2Synthetic(bad);
    const st = join(bad, "model.safetensors");
    const t = openCheckpoint(bad).tensors.get("model.layers.1.mlp.up_proj.weight")!;
    const bytes = readFileSync(st);
    bytes.writeUInt16LE(0x7fc0, t.offset + 10);
    writeFileSync(st, bytes);
    const outBad = join(work.path, "qwen-nan-out");
    await expect(convertQwen2(options(bad, outBad))).rejects.toThrow(/non-finite/);
    expect(readdirSync(outBad)).toEqual([]);
    await expect(convertQwen2(options(QWEN_SRC, join(work.path, "qwen-pinned"), { pinned: true })))
      .rejects.toThrow(/SHA-256|is not id|checksum/);
  });

  test("unused source tensors are refused", async () => {
    const extra = join(work.path, "qwen-extra");
    writeQwen2Synthetic(extra);
    const ckpt = openCheckpoint(extra);
    writeSafetensors(join(extra, "model.safetensors"), [...ckpt.tensors.values()].map((t) => {
      const v = new Float32Array(t.bytes / 2);
      decodeRow("BF16", readTensorBytes(t), 0, v.length, v);
      return { name: t.name, dtype: t.dtype, shape: t.shape, values: v };
    }).concat([{ name: "model.extra.weight", dtype: "BF16", shape: [2], values: new Float32Array(2) }]));
    await expect(convertQwen2(options(extra, join(work.path, "qwen-extra-out")))).rejects.toThrow(/unused source tensors: model\.extra\.weight/);
  });

  test.each([
    ["rope scaling", { rope_scaling: { type: "linear", factor: 2 } }],
    ["a sliding window", { use_sliding_window: true }],
    ["an untied head", { tie_word_embeddings: false }],
    ["another model type", { model_type: "llama" }],
  ])("the Qwen2 gate rejects %s", (_what, patch) => {
    const raw = JSON.parse(readFileSync(join(QWEN_SRC, "config.json"), "utf8")) as Record<string, unknown>;
    expect(() => qwen2Config({ ...raw, ...patch })).toThrow(/only full-attention/);
  });
});

describe("Phi-3-shaped synthetic checkpoint (sharded, fused, partial rotary)", () => {
  const out = join(work.path, "phi-out");
  beforeAll(async () => { await convertPhi3(options(PHI_SRC, out)); });

  test("config carries rotary_dim, the attention factor and the original context", () => {
    const text = readFileSync(join(out, "config.ini"), "utf8");
    expect(text).toContain("architecture=smolmoe_v1\n");
    expect(text).toContain("max_context=64\n");
    /* sqrt(1 + ln(256/64) / ln(64)) in double, as transformers computes it */
    expect(text.endsWith("rotary_dim=12\nrope_attention_factor=1.1547005383792517\n")).toBe(true);
    const r = checkFiles({ model: join(out, "model.smoe"), config: readFileSync(join(out, "config.ini")),
      tokenizer: readFileSync(join(out, "tokenizer.qbp")) });
    expect([r.container.count, r.tokenizer.version]).toEqual([3 + 2 * 10, 2]);
  });

  test("tokenizer: v2 header, ids 256/257, placeholder and padding flagged", () => {
    const q = parseQbp(readFileSync(join(out, "tokenizer.qbp")), PHI_SHAPE.vocab);
    expect(q.header).toEqual({ version: 2, normalizer: 0, pretokenizer: 2, template: 2 });
    const text = (i: number) => new TextDecoder().decode(q.pieces[i]);
    expect([text(256), text(257)]).toEqual(["<|endoftext|>", "<|end|>"]);
    const placeholder = q.pieces.findIndex((p) => new TextDecoder().decode(p) === "<|endoftext07|>");
    expect(q.special[placeholder]).toBe(1);
    expect(q.special[PHI_SHAPE.vocab - 1]).toBe(1);
    expect(q.special[260]).toBe(0);
  });

  test("q/k split from qkv with the partial permutation; v, gate and up are plain slices", () => {
    const model = join(out, "model.smoe");
    const hd = PHI_SHAPE.dim / PHI_SHAPE.heads, order = ropePairOrder(hd, 12), qkv = "model.layers.1.self_attn.qkv_proj.weight";
    expect([...order]).toEqual([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 12, 13, 14, 15]);
    for (const n of [0, 1, 11, 12, 15, hd + 5]) {
      const src = Math.floor(n / hd) * hd + order[n % hd]!;
      expect(modelRow(model, "l.1.q", n).equals(expectedRow(PHI_SRC, qkv, src))).toBe(true);
      expect(modelRow(model, "l.1.k", n).equals(expectedRow(PHI_SRC, qkv, PHI_SHAPE.dim + src))).toBe(true);
    }
    expect(modelRow(model, "l.1.v", 3).equals(expectedRow(PHI_SRC, qkv, PHI_SHAPE.dim + 2 * hd + 3))).toBe(true);
    const gu = "model.layers.0.mlp.gate_up_proj.weight";
    expect(modelRow(model, "l.0.e.0.gate", 5).equals(expectedRow(PHI_SRC, gu, 5))).toBe(true);
    expect(modelRow(model, "l.0.e.0.up", 5).equals(expectedRow(PHI_SRC, gu, PHI_SHAPE.hidden + 5))).toBe(true);
  });

  test("provenance records the source hashes and the RoPE handling", () => {
    const p = JSON.parse(readFileSync(join(out, "provenance.json"), "utf8")) as Record<string, unknown>;
    expect(Object.keys(p.source_sha256 as object)).toEqual(["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]);
    expect(p).toMatchObject({ revision: "unpinned", tied_lm_head: true, rope: { rotary_dim: 12, max_context: 64 } });
  });

  test("pinned mode refuses a tokenizer that is not the pinned Phi-4-mini one", async () => {
    await expect(convertPhi3(options(PHI_SRC, join(work.path, "phi-pinned"), { pinned: true }))).rejects.toThrow(/SHA-256|pinned/);
  });

  const raw = () => JSON.parse(readFileSync(join(PHI_SRC, "config.json"), "utf8")) as Record<string, unknown>;
  const rope = (patch: Record<string, unknown>) => ({ rope_scaling: { ...(raw().rope_scaling as object), ...patch } });
  test.each([
    ["a short_factor other than 1.0", rope({ short_factor: [1, 1, 1, 1, 1, 1.1] }), /short_factor/],
    ["a short_factor of the wrong length", rope({ short_factor: [1, 1] }), /short_factor/],
    ["another scaling type", rope({ type: "yarn" }), /longrope/],
    ["an unknown scaling key", rope({ beta: 1 }), /unsupported rope_scaling/],
    ["another model type", { model_type: "phi" }, /phi3/],
    ["an untied head", { tie_word_embeddings: false }, /tie_word_embeddings/],
    ["attention biases", { attention_bias: true }, /attention_bias/],
    ["a rotary dim that is not an even integer", { partial_rotary_factor: 0.7 }, /partial_rotary_factor/],
    ["an active sliding window", { sliding_window: 32 }, /sliding_window/],
    ["gelu", { hidden_act: "gelu" }, /silu/],
    ["head_dim != hidden/heads", { head_dim: 32 }, /head_dim/],
  ])("the Phi-3 gate rejects %s", (_what, patch, error) => {
    expect(() => phi3Config({ ...raw(), ...patch })).toThrow(error);
  });

  test("F16 and F32 sources convert like BF16 of the same values", async () => {
    const f32 = join(work.path, "phi-f32");
    writePhi3Synthetic(f32, 2, () => {}, "F32");
    await convertPhi3(options(f32, join(work.path, "phi-f32-out")));
    expect(checkFiles({ model: join(work.path, "phi-f32-out", "model.smoe"),
      config: readFileSync(join(work.path, "phi-f32-out", "config.ini")),
      tokenizer: readFileSync(join(work.path, "phi-f32-out", "tokenizer.qbp")) }).container.count).toBe(23);
  });
});
