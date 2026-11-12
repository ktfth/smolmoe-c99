/**
 * Llama 3.x family (LlamaForCausalLM) -> smolmoe_v1. Pinned download:
 * dphn/Dolphin3.0-Llama3.2-1B (ChatML fine-tune of Llama 3.2 1B); any other
 * Llama 3.x checkpoint with the same tokenizer layout converts unpinned
 * (Dolphin 3.0 Llama 3.2 3B, Llama 3.1 8B, ...).
 *
 * The transformer block is Qwen2's without biases: RMSNorm, GQA, SwiGLU and
 * HF rotate_half RoPE, so Q and K rows are permuted per head to the runtime's
 * adjacent pairs. rope_scaling "llama3" maps to the four rope_llama3_* keys.
 * The head is aliased to the embedding when tie_word_embeddings (3.2 1B/3B)
 * and stored on its own otherwise (3.1 8B); both follow the tokenizer's id
 * permutation.
 *
 * Tokenizer: byte-level BPE, no normalizer, the Llama 3 Split regex
 * (\p{N}{1,3}) and ignore_merges -> SMQBP1 v2 with pre-tokenizer llama3.
 * Runtime ids: bytes 0..255, <|begin_of_text|> -> 256, EOS -> 257, every other
 * id ascending. With <|im_start|>/<|im_end|> (Dolphin) the template is
 * ChatML-BOS and EOS is <|im_end|>; without them the file has no chat template
 * and EOS is <|eot_id|> (or <|end_of_text|> in a base model).
 */
import { existsSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";

import { type ModelConfig, formatConfig, parseConfig } from "../smoe/config";
import { encodeQbp, NORMALIZER, parseQbp, PRETOKENIZER, TEMPLATE } from "../smoe/qbp";
import { buildTokenizer, readHfBpe } from "./bpe-tokenizer";
import {
  CONVERTER, type ConvertOptions, type ConvertReport, type Json, asObject, canonicalJson, copyLicense, intField,
  numField, readJsonFile, report, writeAtomic,
} from "./common";
import type { PinnedFile } from "./fetch";
import { sha256Files } from "./hash";
import { qkRows, rangeRows } from "./permute";
import { type ModelPlan, type TensorSource, checkPlan, writeModel } from "./pipeline";
import { type Checkpoint, type HfTensor, openCheckpoint } from "./safetensors";

export const DOLPHIN3_1B = {
  repo: "dphn/Dolphin3.0-Llama3.2-1B",
  revision: "e753b6ebd7adf87036eb6a3e6de68acca5850e2f",
  sha256: {
    "config.json": "e6d344056272bcbaa2779b6430429492e6d7805f946f02f2293e0bd300fdc7af",
    "tokenizer.json": "e40b93124a3e29f62d5f4ff41be56cb2af34ecacf9239acd9da53a98860380b5",
    "model.safetensors": "fa82d5c8c1cf12762a318daa6a6fd90ba0322c989b66cd54e565b685ab3294b0",
  } as Readonly<Record<string, string>>,
} as const;

export const DOLPHIN3_1B_FILES: readonly PinnedFile[] = [
  { name: "config.json", revision: DOLPHIN3_1B.revision, sha256: DOLPHIN3_1B.sha256["config.json"] },
  { name: "tokenizer.json", revision: DOLPHIN3_1B.revision, sha256: DOLPHIN3_1B.sha256["tokenizer.json"] },
  { name: "README.md", revision: DOLPHIN3_1B.revision },
  { name: "model.safetensors", revision: DOLPHIN3_1B.revision, sha256: DOLPHIN3_1B.sha256["model.safetensors"] },
];

export const LLAMA3_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|" +
  "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

const PRE_TOKENIZER = {
  type: "Sequence", pretokenizers: [
    { type: "Split", pattern: { Regex: LLAMA3_REGEX }, behavior: "Isolated", invert: false },
    { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: false },
  ],
};
const MODEL_FLAGS = { type: "BPE", dropout: null, unk_token: null, continuing_subword_prefix: null,
  end_of_word_suffix: null, fuse_unk: false, byte_fallback: false, ignore_merges: true };
const BOS = "<|begin_of_text|>";
/** Largest context the runtime config accepts from a checkpoint (KV cache is sized by --context). */
const MAX_CONTEXT = 131072;

const falsy = (v: unknown) => v === undefined || v === null || v === false;

export interface LlamaConfig {
  readonly config: ModelConfig;
  readonly tied: boolean;
}

/** Strict gate for what the runtime represents; maps to a dense one-expert smolmoe_v1 config. */
export function llamaConfig(json: unknown): LlamaConfig {
  const raw = asObject(json, "config.json"), what = "config.json";
  if (raw.model_type !== "llama") throw new Error(`config.json: model_type must be llama, got ${String(raw.model_type)}`);
  if (raw.hidden_act !== "silu") throw new Error("config.json: hidden_act must be silu");
  for (const k of ["attention_bias", "mlp_bias"]) if (!falsy(raw[k])) throw new Error(`config.json: ${k} is not supported`);
  if (raw.pretraining_tp !== undefined && raw.pretraining_tp !== 1) throw new Error("config.json: pretraining_tp must be 1");
  const dim = intField(raw, "hidden_size", what), heads = intField(raw, "num_attention_heads", what);
  const headDim = raw.head_dim === undefined || raw.head_dim === null ? dim / heads : intField(raw, "head_dim", what);
  const config: Record<string, unknown> = {
    architecture: "smolmoe_v1", dim, hidden_dim: intField(raw, "intermediate_size", what),
    layers: intField(raw, "num_hidden_layers", what), heads,
    kv_heads: intField(raw, "num_key_value_heads", what, heads), experts: 1, top_k: 1,
    vocab_size: intField(raw, "vocab_size", what),
    max_context: Math.min(intField(raw, "max_position_embeddings", what), MAX_CONTEXT),
    shared_experts: 0, bos_id: 256, eos_id: 257, rms_eps: numField(raw, "rms_norm_eps", what),
    rope_theta: numField(raw, "rope_theta", what),
  };
  if (headDim !== dim / heads) config.head_dim = headDim;
  if (raw.rope_scaling !== undefined && raw.rope_scaling !== null) {
    const s = asObject(raw.rope_scaling, "config.json rope_scaling");
    const type = s.rope_type ?? s.type;
    const known = new Set(["rope_type", "type", "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"]);
    const unknown = Object.keys(s).filter((k) => !known.has(k));
    if (type !== "llama3" || unknown.length) throw new Error("config.json: only rope_scaling of rope_type llama3 is supported");
    const sw = "config.json rope_scaling";
    config.rope_llama3_factor = numField(s, "factor", sw);
    config.rope_llama3_low_freq_factor = numField(s, "low_freq_factor", sw);
    config.rope_llama3_high_freq_factor = numField(s, "high_freq_factor", sw);
    config.rope_llama3_original_context = intField(s, "original_max_position_embeddings", sw);
  }
  const c = config as unknown as ModelConfig;
  return { config: parseConfig(formatConfig(c)), tied: raw.tie_word_embeddings === true };
}

function gateTokenizer(t: Json): void {
  const fail = (what: string) => new Error(`tokenizer.json: ${what}`);
  if (t.normalizer !== null && t.normalizer !== undefined) throw fail("normalizer must be null");
  if (canonicalJson(t.pre_tokenizer) !== canonicalJson(PRE_TOKENIZER)) throw fail("pre_tokenizer must be the Llama 3 Split + ByteLevel");
  if (asObject(t.decoder, "tokenizer.json decoder").type !== "ByteLevel") throw fail("decoder must be ByteLevel");
  const { vocab: _vocab, merges: _merges, ...flags } = asObject(t.model, "tokenizer.json model");
  if (canonicalJson(flags) !== canonicalJson(MODEL_FLAGS))
    throw fail("model must be plain BPE with ignore_merges (no dropout, unk, prefixes, byte fallback)");
}

export interface LlamaTokenizer {
  readonly bytes: Uint8Array;
  readonly oldIds: Uint32Array;
  readonly eos: string;
  readonly chat: boolean;
}

export function llamaTokenizer(json: unknown, vocab: number): LlamaTokenizer {
  const t = asObject(json, "tokenizer.json");
  gateTokenizer(t);
  const hf = readHfBpe(t);
  const has = (content: string) => hf.added.some((a) => a.content === content);
  if (!has(BOS)) throw new Error(`tokenizer.json: ${BOS} is missing`);
  const chat = has("<|im_start|>") && has("<|im_end|>");
  const eos = chat ? "<|im_end|>" : has("<|eot_id|>") ? "<|eot_id|>" : "<|end_of_text|>";
  if (!has(eos)) throw new Error(`tokenizer.json: ${eos} is missing`);
  /* No unreachable flags: with ignore_merges a base piece no merge produces is still a whole pre-token. */
  const built = buildTokenizer(hf, { vocabSize: vocab, bos: BOS, eos, flagUnreachable: false });
  const bytes = encodeQbp({
    header: { version: 2, normalizer: NORMALIZER.none, pretokenizer: PRETOKENIZER.llama3,
      template: chat ? TEMPLATE.chatmlBos : TEMPLATE.none },
    pieces: built.pieces, special: built.special, merges: built.merges,
  });
  parseQbp(bytes, vocab);
  return { bytes, oldIds: built.oldIds, eos, chat };
}

export function llamaPlan(c: ModelConfig, tied: boolean, ckpt: Checkpoint, oldIds: Uint32Array): ModelPlan {
  const hd = c.head_dim ?? c.dim / c.heads, qd = c.heads * hd, kd = c.kv_heads * hd;
  const get = (name: string, shape: readonly number[]): HfTensor => {
    const t = ckpt.tensors.get(name);
    if (!t) throw new Error(`missing source tensor ${name}`);
    if (t.dtype !== "BF16") throw new Error(`unexpected tensor: ${name} is ${t.dtype}, expected BF16`);
    if (t.shape.join(",") !== shape.join(",")) throw new Error(`${name}: shape [${t.shape.join(", ")}], expected [${shape.join(", ")}]`);
    return t;
  };
  const rows = (name: string, shape: readonly [number, number], map?: Uint32Array): TensorSource =>
    ({ kind: "rows", tensor: get(name, shape), rows: map ?? rangeRows(0, shape[0]) });
  const vector = (name: string, len: number): TensorSource => ({ kind: "vector", tensor: get(name, [len]) });
  if (tied && ckpt.tensors.has("lm_head.weight")) throw new Error("tie_word_embeddings is true, but the checkpoint has lm_head.weight");
  const s = new Map<string, TensorSource>([
    ["embed", rows("model.embed_tokens.weight", [c.vocab_size, c.dim], oldIds)],
    ["final_norm", vector("model.norm.weight", c.dim)],
    ["lm_head", tied ? { kind: "alias" } : rows("lm_head.weight", [c.vocab_size, c.dim], oldIds)],
  ]);
  for (let l = 0; l < c.layers; l++) {
    const p = `model.layers.${l}.`, n = `l.${l}.`;
    s.set(`${n}attn_norm`, vector(`${p}input_layernorm.weight`, c.dim));
    s.set(`${n}ffn_norm`, vector(`${p}post_attention_layernorm.weight`, c.dim));
    s.set(`${n}q`, rows(`${p}self_attn.q_proj.weight`, [qd, c.dim], qkRows(0, c.heads, hd, hd)));
    s.set(`${n}k`, rows(`${p}self_attn.k_proj.weight`, [kd, c.dim], qkRows(0, c.kv_heads, hd, hd)));
    s.set(`${n}v`, rows(`${p}self_attn.v_proj.weight`, [kd, c.dim]));
    s.set(`${n}o`, rows(`${p}self_attn.o_proj.weight`, [c.dim, qd]));
    s.set(`${n}router`, { kind: "zeros" });
    s.set(`${n}e.0.gate`, rows(`${p}mlp.gate_proj.weight`, [c.hidden_dim, c.dim]));
    s.set(`${n}e.0.up`, rows(`${p}mlp.up_proj.weight`, [c.hidden_dim, c.dim]));
    s.set(`${n}e.0.down`, rows(`${p}mlp.down_proj.weight`, [c.dim, c.hidden_dim]));
  }
  const plan = { config: c, sources: s };
  checkPlan(plan, ckpt);
  return plan;
}

/** Whether every source file hashes to the pinned Dolphin 3.0 Llama 3.2 1B revision. */
function isPinned(byName: Readonly<Record<string, string>>): boolean {
  return Object.entries(DOLPHIN3_1B.sha256).every(([name, digest]) => byName[name] === digest);
}

function provenance(c: ModelConfig, tied: boolean, tok: LlamaTokenizer, hashes: ReadonlyMap<string, string>): string {
  const byName = Object.fromEntries([...hashes].map(([path, h]) => [basename(path), h]));
  const pinned = isPinned(byName);
  const shards = Object.fromEntries(Object.entries(byName).filter(([n]) => n.endsWith(".safetensors")));
  return `${JSON.stringify({
    model: pinned ? DOLPHIN3_1B.repo : "unpinned Llama 3.x checkpoint", revision: pinned ? DOLPHIN3_1B.revision : "unpinned",
    converter: CONVERTER,
    source_sha256: shards, config_sha256: byName["config.json"], tokenizer_sha256: byName["tokenizer.json"],
    index_sha256: byName["model.safetensors.index.json"] ?? null,
    quantization: "lossy Q8 block32",
    qk_permutation: "split-half rotary to adjacent-pair per head; V and O unchanged",
    rope: c.rope_llama3_factor === undefined ? { theta: c.rope_theta } : {
      theta: c.rope_theta, scaling: "llama3", factor: c.rope_llama3_factor,
      low_freq_factor: c.rope_llama3_low_freq_factor, high_freq_factor: c.rope_llama3_high_freq_factor,
      original_max_position_embeddings: c.rope_llama3_original_context,
    },
    tokenizer: `byte-level BPE, Llama 3 pre-tokenizer with ignore_merges, template ${tok.chat ? "ChatML opened with BOS" : "none"}`,
    token_remapping: `runtime ids: bytes 0..255, ${BOS} -> 256, ${tok.eos} -> 257, all other ids ascending; embedding rows follow`,
    tied_lm_head: tied, dense_mapping: "one expert with a zero router", full_kimi_compatible: false,
  }, null, 2)}\n`;
}

export async function convertLlama(o: ConvertOptions): Promise<ConvertReport> {
  const started = performance.now();
  const configPath = join(o.source, "config.json"), tokenizerPath = join(o.source, "tokenizer.json");
  const { config, tied } = llamaConfig(readJsonFile(configPath));
  const tokenizer = llamaTokenizer(readJsonFile(tokenizerPath), config.vocab_size);
  const ckpt = openCheckpoint(o.source);
  const plan = llamaPlan(config, tied, ckpt, tokenizer.oldIds);
  mkdirSync(o.output, { recursive: true });
  const index = join(o.source, "model.safetensors.index.json");
  const hashes = sha256Files([...ckpt.files, configPath, tokenizerPath, ...(existsSync(index) ? [index] : [])]);
  hashes.catch(() => undefined); /* observed before the rename below */
  const files = [join(o.output, "config.ini"), join(o.output, "tokenizer.qbp"), join(o.output, "model.smoe"),
    join(o.output, "provenance.json")];
  const beforeRename = async () => {
    const h = await hashes;
    if (o.pinned) {
      const byName = Object.fromEntries([...h].map(([path, d]) => [basename(path), d]));
      if (!isPinned(byName)) throw new Error(`source files do not match the pinned ${DOLPHIN3_1B.repo} revision`);
    }
    writeAtomic(files[0]!, formatConfig(config));
    writeAtomic(files[1]!, tokenizer.bytes);
    writeAtomic(files[3]!, provenance(config, tied, tokenizer, h));
  };
  o.log(`converting ${config.layers} layers (${ckpt.files.length} shard(s), ${tied ? "tied" : "separate"} head) with ${o.threads} thread(s)`);
  await writeModel(plan, { output: files[2]!, container: o.container, threads: o.threads, tokenizer: tokenizer.bytes,
    log: o.log, beforeRename });
  return report([...files, copyLicense(o.source, o.output, "LICENSE-Llama.txt")], started);
}
