/**
 * Llama 3 support: conversion of tiny synthetic Llama-shaped checkpoints
 * (random weights, tied and untied heads), the converter's gates, and, when
 * built, the C runtime on the converted model: llama3 RoPE keys, the llama3
 * pre-tokenizer (3-digit runs), ignore_merges and the ChatML-BOS template.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { ConvertOptions } from "../scripts/hf/common";
import { decodeRow } from "../scripts/hf/decode";
import { convertLlama, llamaConfig, llamaTokenizer } from "../scripts/hf/llama";
import { ropePairOrder } from "../scripts/hf/permute";
import { openCheckpoint, readTensorBytes } from "../scripts/hf/safetensors";
import { LLAMA_SHAPE, LLAMA_WHOLE_PIECES, writeLlamaSynthetic } from "../scripts/hf/synthetic";
import { checkFiles } from "../scripts/smoe/check";
import { parseConfig } from "../scripts/smoe/config";
import { PRETOKENIZER, TEMPLATE, parseQbp } from "../scripts/smoe/qbp";
import { q8RowBytes, quantizeQ8 } from "../scripts/smoe/quant";
import { readContainer } from "../scripts/smoe/reader";
import { BINARY, run, runOk, tempDir } from "./support/run";

const work = tempDir("convert-llama");
afterAll(() => work.dispose());
const UNTIED = join(work.path, "untied-src"), TIED = join(work.path, "tied-src");
const options = (source: string, output: string): ConvertOptions =>
  ({ source, output, container: "v2", threads: 2, pinned: false, log: () => {} });

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

const rawConfig = (dir: string) => JSON.parse(readFileSync(join(dir, "config.json"), "utf8")) as Record<string, unknown>;
const rawTokenizer = (dir: string) => JSON.parse(readFileSync(join(dir, "tokenizer.json"), "utf8")) as Record<string, unknown>;

beforeAll(() => {
  writeLlamaSynthetic(UNTIED);
  writeLlamaSynthetic(TIED, 4, true);
});

describe("Llama-shaped synthetic checkpoint", () => {
  const out = join(work.path, "untied-out"), tiedOut = join(work.path, "tied-out");
  beforeAll(async () => {
    await convertLlama(options(UNTIED, out));
    await convertLlama(options(TIED, tiedOut));
  });

  test("config: smolmoe_v1 with the four llama3 RoPE keys", () => {
    const text = readFileSync(join(out, "config.ini"), "utf8");
    expect(text).toContain("architecture=smolmoe_v1\n");
    expect(text).toContain("rope_theta=500000.0\nrope_llama3_factor=8.0\nrope_llama3_low_freq_factor=1.0\n" +
      "rope_llama3_high_freq_factor=4.0\nrope_llama3_original_context=64\n");
    expect(parseConfig(text).rope_llama3_original_context).toBe(64);
  });

  test("tokenizer: v2 header with the llama3 pre-tokenizer, ChatML-BOS, BOS 256 and EOS <|im_end|> 257", () => {
    const qbp = parseQbp(readFileSync(join(out, "tokenizer.qbp")), LLAMA_SHAPE.vocab);
    expect(qbp.header).toEqual({ version: 2, normalizer: 0, pretokenizer: PRETOKENIZER.llama3, template: TEMPLATE.chatmlBos });
    const text = (id: number) => new TextDecoder().decode(qbp.pieces[id]);
    expect([text(256), text(257)]).toEqual(["<|begin_of_text|>", "<|im_end|>"]);
    /* pieces no merge produces stay ordinary tokens (ignore_merges reaches them) */
    const whole = qbp.pieces.findIndex((p) => new TextDecoder().decode(p) === " zq");
    expect(whole).toBeGreaterThan(257);
    expect(qbp.special[whole]).toBe(0);
  });

  test("files pass the loader check; the untied head is stored, the tied one aliased", () => {
    for (const dir of [out, tiedOut]) {
      const r = checkFiles({ model: join(dir, "model.smoe"), config: readFileSync(join(dir, "config.ini")),
        tokenizer: readFileSync(join(dir, "tokenizer.qbp")) });
      expect(r.container.count).toBe(3 + 2 * 10);
    }
    expect(readContainer(join(out, "model.smoe")).records.find((r) => r.name === "lm_head")!.alias).toBe(0);
    expect(readContainer(join(tiedOut, "model.smoe")).records.find((r) => r.name === "lm_head")!.alias).toBe(1);
  });

  test("embed and lm_head rows follow the token permutation; q/k rows the head permutation", () => {
    const model = join(out, "model.smoe");
    /* <|begin_of_text|> is source id 263 (256 bytes + 5 merges + 2 whole pieces) and runtime id 256 */
    expect(modelRow(model, "embed", 256).equals(expectedRow(UNTIED, "model.embed_tokens.weight", 263))).toBe(true);
    expect(modelRow(model, "lm_head", 256).equals(expectedRow(UNTIED, "lm_head.weight", 263))).toBe(true);
    expect(modelRow(model, "lm_head", 257).equals(expectedRow(UNTIED, "lm_head.weight", 266))).toBe(true);
    expect(modelRow(model, "lm_head", 0x20).equals(expectedRow(UNTIED, "lm_head.weight", 220))).toBe(true);
    const hd = LLAMA_SHAPE.dim / LLAMA_SHAPE.heads, order = ropePairOrder(hd, hd);
    for (const n of [0, 1, 5, hd + 2]) {
      const src = Math.floor(n / hd) * hd + order[n % hd]!;
      expect(modelRow(model, "l.1.q", n).equals(expectedRow(UNTIED, "model.layers.1.self_attn.q_proj.weight", src))).toBe(true);
      expect(modelRow(model, "l.0.k", n).equals(expectedRow(UNTIED, "model.layers.0.self_attn.k_proj.weight", src))).toBe(true);
    }
    expect(modelRow(model, "l.0.v", 3).equals(expectedRow(UNTIED, "model.layers.0.self_attn.v_proj.weight", 3))).toBe(true);
  });

  test("provenance records the llama3 RoPE scaling and the head layout", () => {
    const p = JSON.parse(readFileSync(join(out, "provenance.json"), "utf8"));
    expect(p.revision).toBe("unpinned");
    expect(p.rope).toEqual({ theta: 500000, scaling: "llama3", factor: 8, low_freq_factor: 1, high_freq_factor: 4,
      original_max_position_embeddings: 64 });
    expect(p.tied_lm_head).toBe(false);
  });

  describe.skipIf(!existsSync(BINARY))("C runtime", () => {
    const args = (dir: string) => ["--model", join(dir, "model.smoe"), "--config", join(dir, "config.ini"), "--tokenizer",
      join(dir, "tokenizer.qbp"), "--context", 64, "--memory-mib", 16];
    const tokenize = (dir: string, text: string, extra: readonly string[] = []) => {
      const ids = join(work.path, "ids.txt");
      const r = run([...args(dir), "--prompt", text, "--tokenize-only", "--token-ids", ids, ...extra]);
      expect(r.code).toBe(0);
      return readFileSync(ids, "utf8").trim().split("\n").map(Number);
    };
    const pieceId = (dir: string, text: string) =>
      parseQbp(readFileSync(join(dir, "tokenizer.qbp")), LLAMA_SHAPE.vocab).pieces.findIndex((p) => new TextDecoder().decode(p) === text);

    test("plain text opens with BOS; digits group by three; whole pieces skip the merges", () => {
      const [zq, d123] = LLAMA_WHOLE_PIECES.map((p) => pieceId(out, p.replace("Ġ", " ")));
      /* pre-tokens "123" (a whole piece), "456" (no merge: bytes) and "7" */
      expect(tokenize(out, "1234567")).toEqual([256, d123!, 0x34, 0x35, 0x36, 0x37]);
      expect(tokenize(out, "a zq")).toEqual([256, 0x61, zq!]);
      /* " zqx" is one pre-token that is not a whole piece: the merges run (no merge covers it: bytes) */
      expect(tokenize(out, " zqx")).toEqual([256, 0x20, 0x7a, 0x71, 0x78]);
    });

    test("the chat opens with BOS before the system message, never before later turns", () => {
      const ids = tokenize(out, "hi", ["--chat", "--system", "S"]);
      const imStart = pieceId(out, "<|im_start|>");
      expect(ids.slice(0, 3)).toEqual([256, imStart, 0x73]);
      expect(ids.filter((x) => x === 256)).toHaveLength(1);
      expect(ids.filter((x) => x === 257)).toHaveLength(2);
    });

    test("the converted model runs, deterministically across threads and batches", () => {
      const go = (extra: readonly (string | number)[]) =>
        runOk([...args(out), "--prompt", "the in 123", "--tokens", 6, "--hash-logits", ...extra]).stats.logits_hash;
      const a = go(["--threads", 1, "--batch", 1]);
      expect(go(["--threads", 3, "--batch", 4])).toBe(a);
      expect(go(["--scalar"])).toBe(a);
      runOk([...args(tiedOut), "--prompt", "zq", "--tokens", 3]);
    });

    test("RoPE scaling changes the logits, and a partial set of llama3 keys is refused", () => {
      const cfg = readFileSync(join(out, "config.ini"), "utf8");
      const plain = join(work.path, "plain.ini"), partial = join(work.path, "partial.ini");
      writeFileSync(plain, cfg.split("\n").filter((l) => !l.startsWith("rope_llama3_")).join("\n"));
      writeFileSync(partial, cfg.replace(/rope_llama3_high_freq_factor=.*\n/, ""));
      const base = [...args(out), "--prompt", "the in the in the in", "--tokens", 2, "--hash-logits"];
      const scaled = runOk(base).stats.logits_hash;
      const unscaled = runOk([...base, "--config", plain]).stats.logits_hash;
      expect(unscaled).not.toBe(scaled);
      const r = run([...base, "--config", partial]);
      expect(r.code).not.toBe(0);
      expect(r.stderr.toString()).toContain("llama3 RoPE scaling needs all of");
    });
  });
});

describe("Llama gates", () => {
  test.each([
    ["another model type", { model_type: "mistral" }, /model_type must be llama/],
    ["attention biases", { attention_bias: true }, /attention_bias is not supported/],
    ["another RoPE scaling", { rope_scaling: { rope_type: "yarn", factor: 4 } }, /rope_type llama3/],
    ["a non-SiLU activation", { hidden_act: "gelu" }, /silu/],
  ])("the config gate rejects %s", (_what, patch, message) => {
    expect(() => llamaConfig({ ...rawConfig(UNTIED), ...patch })).toThrow(message);
  });

  test("a config without rope_scaling has no llama3 keys", () => {
    const { config } = llamaConfig({ ...rawConfig(UNTIED), rope_scaling: null });
    expect(config.rope_llama3_factor).toBeUndefined();
  });

  test("the tokenizer gate wants ignore_merges and the Llama 3 regex", () => {
    const t = rawTokenizer(UNTIED);
    const model = t.model as Record<string, unknown>;
    expect(() => llamaTokenizer({ ...t, model: { ...model, ignore_merges: false } }, LLAMA_SHAPE.vocab)).toThrow(/ignore_merges/);
    const pre = JSON.parse(JSON.stringify(t.pre_tokenizer).replace("{1,3}", ""));
    expect(() => llamaTokenizer({ ...t, pre_tokenizer: pre }, LLAMA_SHAPE.vocab)).toThrow(/Llama 3 Split/);
  });

  test("without ChatML tokens there is no chat template and EOS is <|eot_id|>", () => {
    const t = rawTokenizer(UNTIED);
    const added = (t.added_tokens as { content: string }[]).filter((a) => !a.content.startsWith("<|im_"));
    const tok = llamaTokenizer({ ...t, added_tokens: added }, LLAMA_SHAPE.vocab);
    expect([tok.chat, tok.eos]).toEqual([false, "<|eot_id|>"]);
    expect(parseQbp(tok.bytes, LLAMA_SHAPE.vocab).header).toMatchObject({ template: TEMPLATE.none });
  });

  test("a tied config with an lm_head tensor is refused", async () => {
    const dir = join(work.path, "tied-with-head");
    writeLlamaSynthetic(dir, 5, false, (c) => { c.tie_word_embeddings = true; });
    await expect(convertLlama(options(dir, join(work.path, "tied-with-head-out")))).rejects.toThrow(/has lm_head\.weight/);
  });
});
