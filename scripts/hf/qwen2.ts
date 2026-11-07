/**
 * Qwen2 (Qwen2.5-0.5B-Instruct) -> smolmoe, a TypeScript port of
 * tools/import_qwen2.py + tools/q8pack.c that reproduces the shipped files
 * byte for byte: config.ini, tokenizer.qbp (v1), model.smoe (v2, compact),
 * provenance.json and LICENSE-Qwen.txt.
 */
import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { type ModelConfig, formatConfig, parseConfig } from "../smoe/config";
import { encodeQbp, parseQbp } from "../smoe/qbp";
import { buildTokenizer, readHfBpe } from "./bpe-tokenizer";
import {
  type ConvertOptions, type ConvertReport, type Json, asObject, canonicalJson, copyLicense, intField, numField,
  readJsonFile, report, writeAtomic,
} from "./common";
import type { PinnedFile } from "./fetch";
import { sha256File, sha256Files } from "./hash";
import { qkRows, rangeRows } from "./permute";
import { type ModelPlan, type TensorSource, checkPlan, writeModel } from "./pipeline";
import { type Checkpoint, type HfTensor, openCheckpoint } from "./safetensors";

export const QWEN2 = {
  repo: "Qwen/Qwen2.5-0.5B-Instruct",
  revision: "eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2",
  modelSha256: "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
  tokenizerSha256: "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
  configSha256: "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
  specialIds: { "<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645 },
} as const;

/**
 * The files tools/import_qwen2.py fetch() downloads. LICENSE comes from main, like the Python, and is not
 * hash-pinned (it is only copied next to the model).
 */
export const QWEN2_FILES: readonly PinnedFile[] = [
  { name: "config.json", revision: QWEN2.revision, sha256: QWEN2.configSha256 },
  { name: "tokenizer.json", revision: QWEN2.revision, sha256: QWEN2.tokenizerSha256 },
  { name: "LICENSE", revision: "main" },
  { name: "model.safetensors", revision: QWEN2.revision, sha256: QWEN2.modelSha256 },
];

const QWEN2_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

/** The import_qwen2.py gate (plus head_dim), mapped to a dense one-expert config. */
export function qwen2Config(json: unknown): ModelConfig {
  const raw = asObject(json, "config.json");
  if (raw.model_type !== "qwen2" || raw.use_sliding_window || raw.rope_scaling || raw.hidden_act !== "silu" ||
    !raw.tie_word_embeddings)
    throw new Error("config.json: only full-attention, tied-head Qwen2 SiLU supported");
  const dim = intField(raw, "hidden_size", "config.json"), heads = intField(raw, "num_attention_heads", "config.json");
  if (raw.head_dim !== undefined && raw.head_dim !== dim / heads) throw new Error("config.json: head_dim must be hidden_size / heads");
  const config: ModelConfig = {
    architecture: "smolmoe_qwen2", dim, hidden_dim: intField(raw, "intermediate_size", "config.json"),
    layers: intField(raw, "num_hidden_layers", "config.json"), heads,
    kv_heads: intField(raw, "num_key_value_heads", "config.json"), experts: 1, top_k: 1,
    vocab_size: intField(raw, "vocab_size", "config.json"), max_context: intField(raw, "max_position_embeddings", "config.json"),
    shared_experts: 0, bos_id: 256, eos_id: 257, rms_eps: numField(raw, "rms_norm_eps", "config.json"),
    rope_theta: numField(raw, "rope_theta", "config.json"),
  };
  return parseConfig(formatConfig(config)); /* written, then re-read through the validator */
}

function gateTokenizer(t: Json): void {
  if (canonicalJson(t.normalizer) !== canonicalJson({ type: "NFC" })) throw new Error("tokenizer.json: normalizer must be NFC");
  const pre = asObject(t.pre_tokenizer, "tokenizer.json pre_tokenizer");
  const steps = Array.isArray(pre.pretokenizers) ? pre.pretokenizers as unknown[] : [];
  const split = asObject(steps[0] ?? {}, "pre_tokenizer step"), bytes = asObject(steps[1] ?? {}, "pre_tokenizer step");
  if (pre.type !== "Sequence" || steps.length !== 2 || split.type !== "Split" ||
    canonicalJson(split.pattern) !== canonicalJson({ Regex: QWEN2_REGEX }) || bytes.type !== "ByteLevel" || bytes.use_regex !== false)
    throw new Error("tokenizer.json: pre_tokenizer must be Qwen2's Split regex + ByteLevel");
}

export interface QwenTokenizer {
  readonly bytes: Uint8Array;
  readonly oldIds: Uint32Array;
}

export function qwen2Tokenizer(json: unknown, vocab: number, pinned: boolean): QwenTokenizer {
  const t = asObject(json, "tokenizer.json");
  gateTokenizer(t);
  const hf = readHfBpe(t);
  if (pinned) {
    for (const [content, id] of Object.entries(QWEN2.specialIds))
      if (!hf.added.some((a) => a.id === id && a.content === content)) throw new Error(`tokenizer.json: ${content} is not id ${id}`);
  }
  const built = buildTokenizer(hf, { vocabSize: vocab, bos: "<|endoftext|>", eos: "<|im_end|>", flagUnreachable: false });
  const imStarts = hf.added.filter((a) => a.content === "<|im_start|>");
  if (imStarts.length !== 1) throw new Error("tokenizer.json: exactly one <|im_start|> added token is required");
  const imStart = imStarts[0]!;
  const bytes = encodeQbp({ header: { version: 1, imStart: built.toNew[imStart.id]! }, pieces: built.pieces,
    special: built.special, merges: built.merges });
  parseQbp(bytes, vocab);
  return { bytes, oldIds: built.oldIds };
}

export function qwen2Plan(c: ModelConfig, ckpt: Checkpoint, oldIds: Uint32Array): ModelPlan {
  const get = (name: string): HfTensor => {
    const t = ckpt.tensors.get(name);
    if (!t) throw new Error(`missing source tensor ${name}`);
    if (t.dtype !== "BF16") throw new Error(`unexpected tensor: ${name} is ${t.dtype}, expected BF16`);
    return t;
  };
  const hd = c.dim / c.heads;
  const rows = (name: string, map?: Uint32Array): TensorSource => {
    const tensor = get(name);
    return { kind: "rows", tensor, rows: map ?? rangeRows(0, tensor.shape[0] ?? 0) };
  };
  const vector = (name: string, count?: number): TensorSource =>
    ({ kind: "vector", tensor: get(name), ...(count ? { heads: { count, headDim: hd, rotaryDim: hd } } : {}) });
  const s = new Map<string, TensorSource>([
    ["embed", rows("model.embed_tokens.weight", oldIds)], ["final_norm", vector("model.norm.weight")],
    ["lm_head", { kind: "alias" }],
  ]);
  for (let l = 0; l < c.layers; l++) {
    const p = `model.layers.${l}.`, n = `l.${l}.`;
    s.set(`${n}attn_norm`, vector(`${p}input_layernorm.weight`));
    s.set(`${n}ffn_norm`, vector(`${p}post_attention_layernorm.weight`));
    s.set(`${n}q`, rows(`${p}self_attn.q_proj.weight`, qkRows(0, c.heads, hd, hd)));
    s.set(`${n}k`, rows(`${p}self_attn.k_proj.weight`, qkRows(0, c.kv_heads, hd, hd)));
    s.set(`${n}v`, rows(`${p}self_attn.v_proj.weight`));
    s.set(`${n}o`, rows(`${p}self_attn.o_proj.weight`));
    s.set(`${n}q_bias`, vector(`${p}self_attn.q_proj.bias`, c.heads));
    s.set(`${n}k_bias`, vector(`${p}self_attn.k_proj.bias`, c.kv_heads));
    s.set(`${n}v_bias`, vector(`${p}self_attn.v_proj.bias`));
    s.set(`${n}router`, { kind: "zeros" });
    for (const m of ["gate", "up", "down"]) s.set(`${n}e.0.${m}`, rows(`${p}mlp.${m}_proj.weight`));
  }
  const plan = { config: c, sources: s };
  checkPlan(plan, ckpt);
  return plan;
}

/** The exact record import_qwen2.py writes (json.dumps(indent=2) + newline). */
export function qwen2Provenance(modelSha256: string, tokenizerSha256: string): string {
  return `${JSON.stringify({
    model: QWEN2.repo, revision: QWEN2.revision, source_sha256: modelSha256, tokenizer_sha256: tokenizerSha256,
    quantization: "lossy Q8 block32", qk_permutation: "split-half rotary to adjacent-pair, including bias",
    tied_lm_head: true, full_kimi_compatible: false,
  }, null, 2)}\n`;
}

export async function convertQwen2(o: ConvertOptions): Promise<ConvertReport> {
  const started = performance.now();
  const configPath = join(o.source, "config.json"), tokenizerPath = join(o.source, "tokenizer.json");
  if (o.pinned) {
    for (const [path, want] of [[configPath, QWEN2.configSha256], [tokenizerPath, QWEN2.tokenizerSha256]] as const)
      if (sha256File(path) !== want) throw new Error(`${path}: SHA-256 does not match revision ${QWEN2.revision}`);
  }
  const config = qwen2Config(readJsonFile(configPath));
  const tokenizer = qwen2Tokenizer(readJsonFile(tokenizerPath), config.vocab_size, o.pinned);
  const ckpt = openCheckpoint(o.source);
  const modelPath = join(o.source, "model.safetensors");
  if (ckpt.files.length !== 1 || ckpt.files[0] !== modelPath) throw new Error("Qwen2 conversion expects a single model.safetensors");
  const plan = qwen2Plan(config, ckpt, tokenizer.oldIds);
  mkdirSync(o.output, { recursive: true });
  const hashes = sha256Files([modelPath, tokenizerPath]);
  hashes.catch(() => undefined); /* observed before the rename below */
  const files = [join(o.output, "config.ini"), join(o.output, "tokenizer.qbp"), join(o.output, "model.smoe"),
    join(o.output, "provenance.json")];
  /* Side files are written only once the sources are verified, and before the model lands. */
  const beforeRename = async () => {
    const h = await hashes;
    if (o.pinned && h.get(modelPath) !== QWEN2.modelSha256) throw new Error("official model checksum mismatch");
    writeAtomic(files[0]!, formatConfig(config));
    writeAtomic(files[1]!, tokenizer.bytes);
    writeAtomic(files[3]!, qwen2Provenance(h.get(modelPath)!, h.get(tokenizerPath)!));
  };
  o.log(`converting ${config.layers} layers with ${o.threads} thread(s)`);
  await writeModel(plan, { output: files[2]!, container: o.container, threads: o.threads, tokenizer: tokenizer.bytes,
    log: o.log, beforeRename });
  return report([...files, copyLicense(o.source, o.output, "LICENSE-Qwen.txt")], started);
}
