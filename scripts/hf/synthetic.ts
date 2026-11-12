/**
 * Tiny synthetic Hugging Face checkpoints (random weights, NOT a trained
 * model) shaped like Qwen2, Phi-3 and Llama 3, for end-to-end converter tests:
 * config.json, a byte-level BPE tokenizer.json and safetensors (sharded for
 * Phi-3). Weights are seeded and reproducible.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { byteAlphabet } from "./bpe-tokenizer";
import { LLAMA3_REGEX } from "./llama";
import { O200K_REGEX } from "./phi3-tokenizer";
import type { SourceDtype } from "./safetensors";
import { f32ToF16 } from "../smoe/quant";

export interface SyntheticTensor {
  readonly name: string;
  readonly dtype: SourceDtype;
  readonly shape: readonly number[];
  readonly values: Float32Array;
}

const QWEN_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

function encode(t: SyntheticTensor): Uint8Array {
  if (t.dtype === "F32") return new Uint8Array(t.values.slice().buffer);
  const bits = new Uint32Array(t.values.slice().buffer);
  const out = t.dtype === "BF16"
    ? Uint16Array.from(bits, (u) => (u + 0x7fff + ((u >>> 16) & 1)) >>> 16) /* round to nearest even */
    : Uint16Array.from(t.values, (v) => f32ToF16(v));
  return new Uint8Array(out.buffer);
}

export function writeSafetensors(path: string, tensors: readonly SyntheticTensor[]): void {
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  const data = tensors.map(encode);
  let offset = 0;
  tensors.forEach((t, i) => {
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + data[i]!.length] };
    offset += data[i]!.length;
  });
  const json = new TextEncoder().encode(JSON.stringify(header));
  const head = new Uint8Array(8);
  new DataView(head.buffer).setBigUint64(0, BigInt(json.length), true);
  writeFileSync(path, Buffer.concat([head, json, ...data]));
}

export function seeded(seed: number): (scale: number) => number {
  let s = seed >>> 0;
  return (scale) => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return (s / 2 ** 32 - 0.5) * scale; };
}

/** 256 byte tokens in GPT-2 order plus a few merges and the given extra base entries. */
function bpeModel(extra: readonly string[]): { vocab: Record<string, number>; merges: string[][] } {
  const alphabet = byteAlphabet();
  const order = [...Array(256).keys()].sort((a, b) => alphabet[a]!.codePointAt(0)! - alphabet[b]!.codePointAt(0)!);
  const vocab: Record<string, number> = {};
  order.forEach((byte, id) => { vocab[alphabet[byte]!] = id; });
  const merges = [["Ġ", "t"], ["h", "e"], ["Ġt", "he"], ["i", "n"], ["Ġ", "in"]];
  for (const [a, b] of merges) vocab[a! + b!] = Object.keys(vocab).length;
  for (const e of extra) vocab[e] = Object.keys(vocab).length;
  return { vocab, merges };
}

export interface SyntheticShape {
  readonly dim: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly layers: number;
  readonly hidden: number;
  readonly vocab: number;
}

export const QWEN_SHAPE: SyntheticShape = { dim: 64, heads: 4, kvHeads: 2, layers: 2, hidden: 96, vocab: 300 };
export const PHI_SHAPE: SyntheticShape = { dim: 64, heads: 4, kvHeads: 2, layers: 2, hidden: 48, vocab: 300 };
export const LLAMA_SHAPE: SyntheticShape = { dim: 64, heads: 4, kvHeads: 2, layers: 2, hidden: 80, vocab: 300 };
/** Base entries of the Llama synthetic tokenizer that no merge produces: reachable only through ignore_merges. */
export const LLAMA_WHOLE_PIECES = ["Ġzq", "123"] as const;

/** A Qwen2-shaped directory: config.json, tokenizer.json, model.safetensors (BF16). */
export function writeQwen2Synthetic(dir: string, seed = 1, patch: (config: Record<string, unknown>) => void = () => {}): void {
  const s = QWEN_SHAPE, rnd = seeded(seed), kd = s.kvHeads * (s.dim / s.heads);
  mkdirSync(dir, { recursive: true });
  const config: Record<string, unknown> = {
    model_type: "qwen2", hidden_act: "silu", hidden_size: s.dim, intermediate_size: s.hidden, num_hidden_layers: s.layers,
    num_attention_heads: s.heads, num_key_value_heads: s.kvHeads, vocab_size: s.vocab, max_position_embeddings: 128,
    rms_norm_eps: 1e-6, rope_theta: 10000.0, tie_word_embeddings: true, use_sliding_window: false, rope_scaling: null,
  };
  patch(config);
  writeFileSync(join(dir, "config.json"), JSON.stringify(config));
  const model = bpeModel([]);
  const base = Object.keys(model.vocab).length;
  writeFileSync(join(dir, "tokenizer.json"), JSON.stringify({
    normalizer: { type: "NFC" },
    pre_tokenizer: { type: "Sequence", pretokenizers: [
      { type: "Split", pattern: { Regex: QWEN_REGEX }, behavior: "Isolated", invert: false },
      { type: "ByteLevel", add_prefix_space: false, trim_offsets: false, use_regex: false }] },
    model: { type: "BPE", vocab: model.vocab, merges: model.merges.map((m) => m.join(" ")) },
    added_tokens: ["<|endoftext|>", "<|im_start|>", "<|im_end|>"].map((content, i) => ({ id: base + i, content, special: true })),
  }));
  const t = (name: string, shape: number[], scale = 0.2): SyntheticTensor =>
    ({ name, dtype: "BF16", shape, values: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => rnd(scale)) });
  const norm = (name: string, n: number): SyntheticTensor =>
    ({ name, dtype: "BF16", shape: [n], values: Float32Array.from({ length: n }, () => 1 + rnd(0.2)) });
  const tensors = [t("model.embed_tokens.weight", [s.vocab, s.dim], 2), norm("model.norm.weight", s.dim)];
  for (let l = 0; l < s.layers; l++) {
    const p = `model.layers.${l}.`;
    tensors.push(norm(`${p}input_layernorm.weight`, s.dim), norm(`${p}post_attention_layernorm.weight`, s.dim),
      t(`${p}self_attn.q_proj.weight`, [s.dim, s.dim]), t(`${p}self_attn.q_proj.bias`, [s.dim]),
      t(`${p}self_attn.k_proj.weight`, [kd, s.dim]), t(`${p}self_attn.k_proj.bias`, [kd]),
      t(`${p}self_attn.v_proj.weight`, [kd, s.dim]), t(`${p}self_attn.v_proj.bias`, [kd]),
      t(`${p}self_attn.o_proj.weight`, [s.dim, s.dim]), t(`${p}mlp.gate_proj.weight`, [s.hidden, s.dim]),
      t(`${p}mlp.up_proj.weight`, [s.hidden, s.dim]), t(`${p}mlp.down_proj.weight`, [s.dim, s.hidden]));
  }
  writeSafetensors(join(dir, "model.safetensors"), tensors);
}

/** A Phi-3-shaped directory: fused qkv/gate_up, partial rotary 0.75, LongRoPE, two shards. */
export function writePhi3Synthetic(dir: string, seed = 2, patch: (config: Record<string, unknown>) => void = () => {},
  dtype: SourceDtype = "BF16"): void {
  const s = PHI_SHAPE, rnd = seeded(seed), hd = s.dim / s.heads, rot = hd * 0.75;
  mkdirSync(dir, { recursive: true });
  const config: Record<string, unknown> = {
    model_type: "phi3", hidden_act: "silu", hidden_size: s.dim, intermediate_size: s.hidden, num_hidden_layers: s.layers,
    num_attention_heads: s.heads, num_key_value_heads: s.kvHeads, vocab_size: s.vocab, max_position_embeddings: 256,
    original_max_position_embeddings: 64, partial_rotary_factor: 0.75, rms_norm_eps: 1e-5, rope_theta: 10000.0,
    tie_word_embeddings: true, sliding_window: 512, attention_bias: false, mlp_bias: false, lm_head_bias: false,
    rope_scaling: { type: "longrope", short_factor: Array(rot / 2).fill(1.0), long_factor: Array(rot / 2).fill(1.5) },
  };
  patch(config);
  writeFileSync(join(dir, "config.json"), JSON.stringify(config));
  const model = bpeModel(["<|endoftext07|>"]);
  const base = Object.keys(model.vocab).length;
  writeFileSync(join(dir, "tokenizer.json"), JSON.stringify({
    normalizer: null, post_processor: null, decoder: { type: "ByteLevel", add_prefix_space: true, trim_offsets: true, use_regex: true },
    pre_tokenizer: { type: "Sequence", pretokenizers: [
      { type: "Split", pattern: { Regex: O200K_REGEX }, behavior: "Removed", invert: true },
      { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: false }] },
    model: { type: "BPE", dropout: null, unk_token: null, continuing_subword_prefix: "", end_of_word_suffix: "",
      fuse_unk: false, byte_fallback: false, ignore_merges: false, vocab: model.vocab, merges: model.merges },
    added_tokens: ["<|endoftext|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"]
      .map((content, i) => ({ id: base + i, content, special: true })),
  }));
  const t = (name: string, shape: number[], scale = 0.2): SyntheticTensor =>
    ({ name, dtype, shape, values: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => rnd(scale)) });
  const norm = (name: string): SyntheticTensor =>
    ({ name, dtype, shape: [s.dim], values: Float32Array.from({ length: s.dim }, () => 1 + rnd(0.2)) });
  const layer = (l: number): SyntheticTensor[] => {
    const p = `model.layers.${l}.`, qkv = s.dim + 2 * s.kvHeads * hd;
    return [norm(`${p}input_layernorm.weight`), norm(`${p}post_attention_layernorm.weight`),
      t(`${p}self_attn.qkv_proj.weight`, [qkv, s.dim]), t(`${p}self_attn.o_proj.weight`, [s.dim, s.dim]),
      t(`${p}mlp.gate_up_proj.weight`, [2 * s.hidden, s.dim]), t(`${p}mlp.down_proj.weight`, [s.dim, s.hidden])];
  };
  const shards = [[t("model.embed_tokens.weight", [s.vocab, s.dim], 2), ...layer(0)],
    [...Array.from({ length: s.layers - 1 }, (_, i) => layer(i + 1)).flat(), norm("model.norm.weight")]];
  const weightMap: Record<string, string> = {};
  shards.forEach((tensors, i) => {
    const file = `model-0000${i + 1}-of-00002.safetensors`;
    writeSafetensors(join(dir, file), tensors);
    for (const x of tensors) weightMap[x.name] = file;
  });
  writeFileSync(join(dir, "model.safetensors.index.json"), JSON.stringify({ metadata: {}, weight_map: weightMap }));
}

/**
 * A Llama-3-shaped directory: no biases, llama3 rope_scaling, untied head unless
 * `tied`, and a tokenizer with the Llama 3 regex, ignore_merges and the Dolphin
 * ChatML tokens.
 */
export function writeLlamaSynthetic(dir: string, seed = 3, tied = false,
  patch: (config: Record<string, unknown>) => void = () => {}): void {
  const s = LLAMA_SHAPE, rnd = seeded(seed), kd = s.kvHeads * (s.dim / s.heads);
  mkdirSync(dir, { recursive: true });
  const config: Record<string, unknown> = {
    model_type: "llama", architectures: ["LlamaForCausalLM"], hidden_act: "silu", hidden_size: s.dim,
    intermediate_size: s.hidden, num_hidden_layers: s.layers, num_attention_heads: s.heads, num_key_value_heads: s.kvHeads,
    head_dim: s.dim / s.heads, vocab_size: s.vocab, max_position_embeddings: 256, rms_norm_eps: 1e-5, rope_theta: 500000.0,
    attention_bias: false, mlp_bias: false, pretraining_tp: 1, tie_word_embeddings: tied,
    rope_scaling: { factor: 8.0, high_freq_factor: 4.0, low_freq_factor: 1.0, original_max_position_embeddings: 64,
      rope_type: "llama3" },
  };
  patch(config);
  writeFileSync(join(dir, "config.json"), JSON.stringify(config));
  const model = bpeModel([...LLAMA_WHOLE_PIECES]);
  const base = Object.keys(model.vocab).length;
  writeFileSync(join(dir, "tokenizer.json"), JSON.stringify({
    normalizer: null, decoder: { type: "ByteLevel", add_prefix_space: true, trim_offsets: true, use_regex: true },
    pre_tokenizer: { type: "Sequence", pretokenizers: [
      { type: "Split", pattern: { Regex: LLAMA3_REGEX }, behavior: "Isolated", invert: false },
      { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: false }] },
    model: { type: "BPE", dropout: null, unk_token: null, continuing_subword_prefix: null, end_of_word_suffix: null,
      fuse_unk: false, byte_fallback: false, ignore_merges: true, vocab: model.vocab, merges: model.merges.map((m) => m.join(" ")) },
    added_tokens: ["<|begin_of_text|>", "<|end_of_text|>", "<|eot_id|>", "<|im_end|>", "<|im_start|>"]
      .map((content, i) => ({ id: base + i, content, special: true })),
  }));
  const t = (name: string, shape: number[], scale = 0.2): SyntheticTensor =>
    ({ name, dtype: "BF16", shape, values: Float32Array.from({ length: shape.reduce((a, b) => a * b, 1) }, () => rnd(scale)) });
  const norm = (name: string, n: number): SyntheticTensor =>
    ({ name, dtype: "BF16", shape: [n], values: Float32Array.from({ length: n }, () => 1 + rnd(0.2)) });
  const tensors = [t("model.embed_tokens.weight", [s.vocab, s.dim], 2), norm("model.norm.weight", s.dim)];
  if (!tied) tensors.push(t("lm_head.weight", [s.vocab, s.dim], 2));
  for (let l = 0; l < s.layers; l++) {
    const p = `model.layers.${l}.`;
    tensors.push(norm(`${p}input_layernorm.weight`, s.dim), norm(`${p}post_attention_layernorm.weight`, s.dim),
      t(`${p}self_attn.q_proj.weight`, [s.dim, s.dim]), t(`${p}self_attn.k_proj.weight`, [kd, s.dim]),
      t(`${p}self_attn.v_proj.weight`, [kd, s.dim]), t(`${p}self_attn.o_proj.weight`, [s.dim, s.dim]),
      t(`${p}mlp.gate_proj.weight`, [s.hidden, s.dim]), t(`${p}mlp.up_proj.weight`, [s.hidden, s.dim]),
      t(`${p}mlp.down_proj.weight`, [s.dim, s.hidden]));
  }
  writeSafetensors(join(dir, "model.safetensors"), tensors);
}
