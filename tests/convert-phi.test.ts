/**
 * Phi-4-mini-instruct against the real pinned source (skipped when absent):
 * config mapping, tokenizer export (ids, flags, sizes), shard reads past
 * 4 GiB, and spot checks of a converted model.smoe row by row.
 * Source: downloads/phi-4-mini-instruct (or SMOLMOE_PHI_SOURCE); converted
 * model: models/phi-4-mini-instruct (or SMOLMOE_PHI_OUTPUT).
 */
import { describe, expect, test } from "bun:test";
import { closeSync, existsSync, openSync, readFileSync, readSync } from "node:fs";
import { join, resolve } from "node:path";

import { decodeRow } from "../scripts/hf/decode";
import { ropePairOrder } from "../scripts/hf/permute";
import { phi3Config } from "../scripts/hf/phi3";
import { PHI4_MINI_TOKENIZER, phi3Tokenizer } from "../scripts/hf/phi3-tokenizer";
import { type HfTensor, openCheckpoint } from "../scripts/hf/safetensors";
import { checkFiles } from "../scripts/smoe/check";
import { formatConfig } from "../scripts/smoe/config";
import { q8RowBytes, quantizeQ8 } from "../scripts/smoe/quant";
import { readContainer } from "../scripts/smoe/reader";
import { ROOT } from "./support/run";

const SOURCE = resolve(ROOT, process.env.SMOLMOE_PHI_SOURCE ?? join("downloads", "phi-4-mini-instruct"));
const OUTPUT = resolve(ROOT, process.env.SMOLMOE_PHI_OUTPUT ?? join("models", "phi-4-mini-instruct"));
const hasSource = existsSync(join(SOURCE, "tokenizer.json")) && existsSync(join(SOURCE, "model.safetensors.index.json"));
const hasModel = hasSource && existsSync(join(OUTPUT, "model.smoe"));

const EXPECTED_CONFIG = "architecture=smolmoe_v1\ndim=3072\nhidden_dim=8192\nlayers=32\nheads=24\nkv_heads=8\nexperts=1\n" +
  "top_k=1\nvocab_size=200064\nmax_context=4096\nshared_experts=0\nbos_id=256\neos_id=257\nrms_eps=1e-05\n" +
  "rope_theta=10000.0\nrotary_dim=96\nrope_attention_factor=1.1902380714238083\n";

function readAt(path: string, position: number, length: number): Uint8Array {
  const fd = openSync(path, "r");
  try {
    const buf = new Uint8Array(length);
    for (let done = 0; done < length;) done += readSync(fd, buf, done, length - done, position + done);
    return buf;
  } finally {
    closeSync(fd);
  }
}

/** Q8 of one source row, read straight from its shard. */
function sourceRowQ8(t: HfTensor, row: number): Buffer {
  const cols = t.shape[1]!;
  const x = new Float32Array(cols);
  decodeRow(t.dtype, readAt(t.file, t.offset + row * cols * 2, cols * 2), 0, cols, x);
  return Buffer.from(quantizeQ8(x, 1, cols));
}

describe.skipIf(!hasSource)("Phi-4-mini-instruct source", () => {
  test("config maps to smolmoe_v1 with the original 4096 context and the new RoPE keys", () => {
    const { config } = phi3Config(JSON.parse(readFileSync(join(SOURCE, "config.json"), "utf8")));
    expect(formatConfig(config)).toBe(EXPECTED_CONFIG);
  });

  test("tokenizer exports as .qbp v2 with the documented ids and flags", () => {
    const t = phi3Tokenizer(JSON.parse(readFileSync(join(SOURCE, "tokenizer.json"), "utf8")), 200064, PHI4_MINI_TOKENIZER);
    const view = new DataView(t.bytes.buffer);
    expect(t.bytes.length).toBe(5_396_062);
    expect([view.getUint32(12, true), view.getUint32(16, true), view.getUint32(20, true), view.getUint32(24, true),
      view.getUint32(28, true)]).toEqual([199_742, 1_398_614, 0, 2, 2]);
    /* <|system|>S<|end|><|user|>U<|end|><|assistant|> in HF ids, then runtime ids */
    expect([200022, 50, 200020, 200021, 52, 200020, 200019].map((id) => t.toNew[id])).toEqual([200022, 83, 257, 200021, 85, 257, 200020]);
    expect([199999, 200018, 200028, 200029, 200063, 199998, 200017].map((id) => t.toNew[id])).toEqual([256, 200019, 200028, 200029, 200063, 200000, 200018]);
    expect(t.special.reduce((n, f) => n + f, 0)).toBe(12 + 19 + 35);
    expect(t.special[t.toNew[199998]!]).toBe(1); /* <|endoftext98|> placeholder */
    expect(t.special[t.toNew[199997]!]).toBe(0);
  });

  test("194 BF16 tensors in two shards; reads past 4 GiB return the right bytes", async () => {
    const ckpt = openCheckpoint(SOURCE);
    expect(ckpt.tensors.size).toBe(194);
    expect([...ckpt.tensors.values()].every((t) => t.dtype === "BF16")).toBe(true);
    const far = [...ckpt.tensors.values()].filter((t) => t.file === ckpt.files[0]).sort((a, b) => b.offset - a.offset)[0]!;
    expect(far.offset).toBeGreaterThan(2 ** 32);
    const position = far.offset + far.bytes - 4096;
    const viaSync = readAt(far.file, position, 4096);
    const viaBun = new Uint8Array(await Bun.file(far.file).slice(position, position + 4096).arrayBuffer());
    expect(Buffer.from(viaSync).equals(Buffer.from(viaBun))).toBe(true);
  });
});

describe.skipIf(!hasModel)("converted Phi-4-mini model.smoe", () => {
  test("passes the TS loader check", () => {
    const r = checkFiles({ model: join(OUTPUT, "model.smoe"), config: readFileSync(join(OUTPUT, "config.ini")),
      tokenizer: readFileSync(join(OUTPUT, "tokenizer.qbp")) });
    expect([r.container.version, r.container.count, r.container.size]).toEqual([2, 323, 4_316_232_424]);
  }, 300_000);

  test("rows match the source: token map, partial Q/K permutation, fused splits", () => {
    const ckpt = openCheckpoint(SOURCE);
    const model = join(OUTPUT, "model.smoe");
    const records = readContainer(model).records;
    const row = (name: string, r: number) => {
      const rec = records.find((x) => x.name === name)!;
      const n = q8RowBytes(rec.cols);
      return Buffer.from(readAt(model, rec.offset + r * n, n));
    };
    const src = (name: string) => ckpt.tensors.get(name)!;
    const order = ropePairOrder(128, 96);
    expect(row("embed", 256).equals(sourceRowQ8(src("model.embed_tokens.weight"), 199999))).toBe(true);
    expect(row("embed", 0x53).equals(sourceRowQ8(src("model.embed_tokens.weight"), 50))).toBe(true);
    expect(row("embed", 200020).equals(sourceRowQ8(src("model.embed_tokens.weight"), 200019))).toBe(true);
    const qkv = src("model.layers.18.self_attn.qkv_proj.weight");
    for (const n of [0, 1, 95, 96, 127, 128 + 3]) {
      const old = Math.floor(n / 128) * 128 + order[n % 128]!;
      expect(row("l.18.q", n).equals(sourceRowQ8(qkv, old))).toBe(true);
      expect(row("l.18.k", n).equals(sourceRowQ8(qkv, 3072 + old))).toBe(true);
    }
    expect(row("l.18.v", 7).equals(sourceRowQ8(qkv, 4096 + 7))).toBe(true);
    const gu = src("model.layers.31.mlp.gate_up_proj.weight");
    expect(row("l.31.e.0.gate", 9).equals(sourceRowQ8(gu, 9))).toBe(true);
    expect(row("l.31.e.0.up", 9).equals(sourceRowQ8(gu, 8192 + 9))).toBe(true);
    expect(records.find((x) => x.name === "lm_head")!.alias).toBe(1);
  }, 120_000);
});
