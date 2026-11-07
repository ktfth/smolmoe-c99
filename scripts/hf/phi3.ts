/**
 * Phi-3 family (pinned: microsoft/Phi-4-mini-instruct) -> smolmoe_v1.
 *
 * Fused qkv_proj is split into q/k/v rows and gate_up_proj into gate/up halves.
 * Q and K rows are permuted per head within the first rotary_dim dims only
 * (HF partial rotate_half -> adjacent pairs); pass-through dims, V and O are
 * untouched. The tied embedding's rows follow the tokenizer's id permutation
 * and lm_head aliases it. LongRoPE is accepted only when short_factor is all
 * 1.0, and max_context is original_max_position_embeddings (the long factors
 * are never used), so the config adds rotary_dim and rope_attention_factor.
 */
import { existsSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";

import { type ModelConfig, formatConfig, parseConfig } from "../smoe/config";
import {
  CONVERTER, type ConvertOptions, type ConvertReport, type Json, asObject, copyLicense, intField, numField,
  readJsonFile, report, writeAtomic,
} from "./common";
import type { PinnedFile } from "./fetch";
import { sha256File, sha256Files } from "./hash";
import { qkRows, rangeRows } from "./permute";
import { PHI4_MINI_TOKENIZER, phi3Tokenizer } from "./phi3-tokenizer";
import { type ModelPlan, type TensorSource, checkPlan, writeModel } from "./pipeline";
import { type Checkpoint, openCheckpoint } from "./safetensors";

export const PHI4_MINI = {
  repo: "microsoft/Phi-4-mini-instruct",
  revision: "cfbefacb99257ffa30c83adab238a50856ac3083",
  sha256: {
    "config.json": "ac65d86061d3d0d704ee2511fd0eb8713ef19eb6eedba17c3080a4165d5b933b",
    "tokenizer.json": "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea",
    "model.safetensors.index.json": "613a98d5e5716ca96fa75931abedc9c5a5d95f488ce4d62df71e639fe3ac6c59",
    /* HF LFS metadata at the pinned revision */
    "model-00001-of-00002.safetensors": "bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6",
    "model-00002-of-00002.safetensors": "7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983",
  } as Readonly<Record<string, string | undefined>>,
} as const;

/* A pinned file without a pinned hash would download unverified: fail at load instead. */
function pinnedHash(name: string): string {
  const hash = PHI4_MINI.sha256[name];
  if (!hash) throw new Error(`internal error: no pinned SHA-256 for ${name}`);
  return hash;
}

/** Source files for `convert.ts fetch-phi3`; LICENSE is copied next to the model, not hash-pinned. */
export const PHI4_MINI_FILES: readonly PinnedFile[] = [
  ...["config.json", "tokenizer.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors"].map((name) => ({ name, revision: PHI4_MINI.revision, sha256: pinnedHash(name) })),
  { name: "LICENSE", revision: PHI4_MINI.revision },
];

export interface Phi3Rope {
  readonly rotaryDim: number;
  readonly attentionFactor: number;
  readonly sourceMaxPositions: number;
}

const falsy = (v: unknown) => v === undefined || v === null || v === false;

function longRope(raw: Json, rotaryDim: number, maxPos: number, original: number): number {
  const s = asObject(raw.rope_scaling, "config.json rope_scaling");
  const known = new Set(["type", "rope_type", "short_factor", "long_factor", "factor", "attention_factor"]);
  const unknown = Object.keys(s).filter((k) => !known.has(k));
  if (unknown.length) throw new Error(`config.json: unsupported rope_scaling keys ${unknown.join(", ")}`);
  if ((s.type ?? s.rope_type) !== "longrope") throw new Error("config.json: rope_scaling type must be longrope");
  const factors = (k: string) => {
    const f = s[k];
    if (!Array.isArray(f) || f.length !== rotaryDim / 2 || !f.every((x) => typeof x === "number" && x > 0 && Number.isFinite(x)))
      throw new Error(`config.json: rope_scaling.${k} must hold ${rotaryDim / 2} positive numbers`);
    return f as number[];
  };
  if (!factors("short_factor").every((x) => x === 1)) throw new Error("config.json: short_factor must be all 1.0 (plain RoPE up to the original context)");
  factors("long_factor");
  if (s.attention_factor !== undefined) {
    if (typeof s.attention_factor !== "number" || !(s.attention_factor > 0)) throw new Error("config.json: invalid attention_factor");
    return s.attention_factor;
  }
  const factor = s.factor ?? maxPos / original;
  if (typeof factor !== "number" || !(factor > 0)) throw new Error("config.json: invalid rope_scaling factor");
  /* transformers _compute_longrope_parameters */
  return factor <= 1 ? 1 : Math.sqrt(1 + Math.log(factor) / Math.log(original));
}

/** Strict gate for what the runtime can represent; maps to a dense one-expert smolmoe_v1 config. */
export function phi3Config(json: unknown): { readonly config: ModelConfig; readonly rope: Phi3Rope } {
  const raw = asObject(json, "config.json");
  const what = "config.json";
  if (raw.model_type !== "phi3") throw new Error(`config.json: model_type must be phi3, got ${String(raw.model_type)}`);
  if (raw.hidden_act !== "silu") throw new Error("config.json: hidden_act must be silu");
  if (raw.tie_word_embeddings !== true) throw new Error("config.json: tie_word_embeddings must be true");
  for (const k of ["attention_bias", "mlp_bias", "lm_head_bias"]) if (!falsy(raw[k])) throw new Error(`config.json: ${k} is not supported`);
  const dim = intField(raw, "hidden_size", what), heads = intField(raw, "num_attention_heads", what);
  const hd = dim / heads;
  if (!Number.isInteger(hd) || (raw.head_dim !== undefined && raw.head_dim !== hd)) throw new Error("config.json: head_dim must be hidden_size / heads");
  const prf = raw.partial_rotary_factor ?? 1;
  const rotaryDim = typeof prf === "number" ? hd * prf : Number.NaN;
  if (!Number.isInteger(rotaryDim) || rotaryDim < 2 || rotaryDim % 2 || rotaryDim > hd)
    throw new Error(`config.json: partial_rotary_factor ${String(prf)} does not give an even rotary_dim within the head`);
  const maxPos = intField(raw, "max_position_embeddings", what);
  const original = intField(raw, "original_max_position_embeddings", what);
  if (original > maxPos) throw new Error("config.json: original_max_position_embeddings exceeds max_position_embeddings");
  const window = raw.sliding_window;
  if (window !== null && window !== undefined && (typeof window !== "number" || window < original))
    throw new Error("config.json: a sliding_window shorter than the context is not supported");
  const attentionFactor = longRope(raw, rotaryDim, maxPos, original);
  const config: ModelConfig = {
    architecture: "smolmoe_v1", dim, hidden_dim: intField(raw, "intermediate_size", what),
    layers: intField(raw, "num_hidden_layers", what), heads, kv_heads: intField(raw, "num_key_value_heads", what, heads),
    experts: 1, top_k: 1, vocab_size: intField(raw, "vocab_size", what), max_context: original, shared_experts: 0,
    bos_id: 256, eos_id: 257, rms_eps: numField(raw, "rms_norm_eps", what), rope_theta: numField(raw, "rope_theta", what),
    ...(rotaryDim < hd ? { rotary_dim: rotaryDim } : {}),
    ...(attentionFactor !== 1 ? { rope_attention_factor: attentionFactor } : {}),
  };
  return { config: parseConfig(formatConfig(config)), rope: { rotaryDim, attentionFactor, sourceMaxPositions: maxPos } };
}

export function phi3Plan(c: ModelConfig, rotaryDim: number, ckpt: Checkpoint, oldIds: Uint32Array): ModelPlan {
  const get = (name: string) => ckpt.tensors.get(name) ?? (() => { throw new Error(`missing source tensor ${name}`); })();
  const hd = c.dim / c.heads, q = c.heads * hd, kv = c.kv_heads * hd;
  const expectShape = (name: string, shape: readonly number[]) => {
    const t = get(name);
    if (t.shape.join(",") !== shape.join(",")) throw new Error(`${name}: shape [${t.shape.join(", ")}], expected [${shape.join(", ")}]`);
    return t;
  };
  const rows = (name: string, shape: readonly number[], map: Uint32Array): TensorSource =>
    ({ kind: "rows", tensor: expectShape(name, shape), rows: map });
  const vector = (name: string): TensorSource => ({ kind: "vector", tensor: expectShape(name, [c.dim]) });
  const s = new Map<string, TensorSource>([
    ["embed", rows("model.embed_tokens.weight", [c.vocab_size, c.dim], oldIds)],
    ["final_norm", vector("model.norm.weight")], ["lm_head", { kind: "alias" }],
  ]);
  for (let l = 0; l < c.layers; l++) {
    const p = `model.layers.${l}.`, n = `l.${l}.`, qkv = `${p}self_attn.qkv_proj.weight`, gu = `${p}mlp.gate_up_proj.weight`;
    s.set(`${n}attn_norm`, vector(`${p}input_layernorm.weight`));
    s.set(`${n}ffn_norm`, vector(`${p}post_attention_layernorm.weight`));
    s.set(`${n}q`, rows(qkv, [q + 2 * kv, c.dim], qkRows(0, c.heads, hd, rotaryDim)));
    s.set(`${n}k`, rows(qkv, [q + 2 * kv, c.dim], qkRows(q, c.kv_heads, hd, rotaryDim)));
    s.set(`${n}v`, rows(qkv, [q + 2 * kv, c.dim], rangeRows(q + kv, kv)));
    s.set(`${n}o`, rows(`${p}self_attn.o_proj.weight`, [c.dim, q], rangeRows(0, c.dim)));
    s.set(`${n}router`, { kind: "zeros" });
    s.set(`${n}e.0.gate`, rows(gu, [2 * c.hidden_dim, c.dim], rangeRows(0, c.hidden_dim)));
    s.set(`${n}e.0.up`, rows(gu, [2 * c.hidden_dim, c.dim], rangeRows(c.hidden_dim, c.hidden_dim)));
    s.set(`${n}e.0.down`, rows(`${p}mlp.down_proj.weight`, [c.dim, c.hidden_dim], rangeRows(0, c.dim)));
  }
  const plan = { config: c, sources: s };
  checkPlan(plan, ckpt);
  return plan;
}

function provenance(c: ModelConfig, rope: Phi3Rope, hashes: ReadonlyMap<string, string>, pinned: boolean): string {
  const byName = Object.fromEntries([...hashes].map(([path, h]) => [basename(path), h]));
  const shards = Object.fromEntries(Object.entries(byName).filter(([n]) => n.endsWith(".safetensors")));
  return `${JSON.stringify({
    model: pinned ? PHI4_MINI.repo : "unpinned phi3 checkpoint", revision: pinned ? PHI4_MINI.revision : "unpinned",
    converter: CONVERTER,
    source_sha256: shards, config_sha256: byName["config.json"], tokenizer_sha256: byName["tokenizer.json"],
    index_sha256: byName["model.safetensors.index.json"] ?? null,
    quantization: "lossy Q8 block32",
    qk_permutation: `split-half rotary to adjacent-pair within the first ${rope.rotaryDim} of ${c.dim / c.heads} dims per head; pass-through dims, V and O unchanged`,
    rope: { rotary_dim: rope.rotaryDim, theta: c.rope_theta, attention_factor: rope.attentionFactor,
      scaling: "longrope with all-1.0 short_factor (plain RoPE); long_factor unused", max_context: c.max_context,
      source_max_position_embeddings: rope.sourceMaxPositions },
    token_remapping: "runtime ids: bytes 0..255, <|endoftext|> -> 256, <|end|> -> 257, all other ids ascending; embedding rows follow",
    tied_lm_head: true, dense_mapping: "one expert with a zero router", full_kimi_compatible: false,
  }, null, 2)}\n`;
}

export async function convertPhi3(o: ConvertOptions): Promise<ConvertReport> {
  const started = performance.now();
  const configPath = join(o.source, "config.json"), tokenizerPath = join(o.source, "tokenizer.json");
  if (o.pinned) {
    for (const path of [configPath, tokenizerPath]) /* small: fail fast, before any weight is read */
      if (sha256File(path) !== PHI4_MINI.sha256[basename(path)])
        throw new Error(`${path}: SHA-256 does not match revision ${PHI4_MINI.revision}`);
  }
  const { config, rope } = phi3Config(readJsonFile(configPath));
  const tokenizer = phi3Tokenizer(readJsonFile(tokenizerPath), config.vocab_size, o.pinned ? PHI4_MINI_TOKENIZER : null);
  const ckpt = openCheckpoint(o.source);
  const plan = phi3Plan(config, rope.rotaryDim, ckpt, tokenizer.oldIds);
  mkdirSync(o.output, { recursive: true });
  const index = join(o.source, "model.safetensors.index.json");
  const hashes = sha256Files([...ckpt.files, configPath, tokenizerPath, ...(existsSync(index) ? [index] : [])]);
  hashes.catch(() => undefined); /* observed before the rename below */
  const files = [join(o.output, "config.ini"), join(o.output, "tokenizer.qbp"), join(o.output, "model.smoe"),
    join(o.output, "provenance.json")];
  /* Side files are written only once every source is verified, and before the model lands. */
  const beforeRename = async () => {
    const h = await hashes;
    if (o.pinned) {
      for (const [path, digest] of h)
        if (digest !== PHI4_MINI.sha256[basename(path)]) throw new Error(`${basename(path)}: SHA-256 ${digest} does not match the pinned revision`);
    }
    writeAtomic(files[0]!, formatConfig(config));
    writeAtomic(files[1]!, tokenizer.bytes);
    writeAtomic(files[3]!, provenance(config, rope, h, o.pinned));
  };
  o.log(`converting ${config.layers} layers (${ckpt.files.length} shard(s)) with ${o.threads} thread(s)`);
  await writeModel(plan, { output: files[2]!, container: o.container, threads: o.threads, tokenizer: tokenizer.bytes,
    log: o.log, beforeRename });
  return report([...files, copyLicense(o.source, o.output, "LICENSE-Phi.txt")], started);
}
