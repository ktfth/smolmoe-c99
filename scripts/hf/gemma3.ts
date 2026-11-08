/**
 * Gemma 3 family (google/gemma-3-1b-it, google/gemma-3-270m-it) -> smolmoe_gemma3.
 *
 * Gemma 3 uses:
 * - Decoupled head_dim (e.g., 256 with dim=1152 or 640, heads=4, kv_heads=1)
 * - GeGLU activation (gelu_pytorch_tanh)
 * - 4 RMSNorms per layer (attn_norm, post_attn_norm, ffn_norm, post_ffn_norm)
 *   plus per-head Q/K RMSNorms (q_norm, k_norm), all with (1 + weight) scale
 * - Dual RoPE frequencies: rope_local_theta (10000) + sliding_window (512) on
 *   5 of every 6 layers, and rope_theta (1000000) on every 6th layer
 * - SentencePiece-style BPE (▁ -> space, <0x00>..<0xFF> byte fallback) or
 *   byte-level BPE, with <bos> at 256, <end_of_turn> at 257, <start_of_turn>,
 *   and <tool_call> / </tool_call> tool markers.
 */
import { existsSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";

import { type ModelConfig, formatConfig, parseConfig } from "../smoe/config";
import { encodeQbp, NORMALIZER, parseQbp, PRETOKENIZER, TEMPLATE } from "../smoe/qbp";
import { buildTokenizer, readHfBpe } from "./bpe-tokenizer";
import {
  CONVERTER, type ConvertOptions, type ConvertReport, type Json, asObject, copyLicense, intField, numField,
  readJsonFile, report, writeAtomic,
} from "./common";
import type { PinnedFile } from "./fetch";
import { sha256Files } from "./hash";
import { qkRows, rangeRows } from "./permute";
import { type ModelPlan, type TensorSource, checkPlan, writeModel } from "./pipeline";
import { type Checkpoint, openCheckpoint } from "./safetensors";

export const GEMMA3_1B = {
  repo: "unsloth/gemma-3-1b-it",
  revision: "5b11413a10db4e486ef16a20101fd028f8f2499c",
  modelSha256: "3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6",
  tokenizerSha256: "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795",
  configSha256: "c5b2136c452797c0c52cba5793ced59d95453dfdc3444433b03e27699ff1c2cd",
} as const;

export const GEMMA3_FILES: readonly PinnedFile[] = [
  { name: "config.json", revision: GEMMA3_1B.revision, sha256: GEMMA3_1B.configSha256 },
  { name: "tokenizer.json", revision: GEMMA3_1B.revision, sha256: GEMMA3_1B.tokenizerSha256 },
  { name: "model.safetensors", revision: GEMMA3_1B.revision, sha256: GEMMA3_1B.modelSha256 },
];

const falsy = (v: unknown) => v === undefined || v === null || v === false;

export function gemma3Config(json: unknown): ModelConfig {
  const root = asObject(json, "config.json");
  const raw = typeof root.text_config === "object" && root.text_config !== null
    ? asObject(root.text_config, "config.json text_config") : root;
  const what = "config.json";
  if (raw.model_type !== "gemma3_text" && raw.model_type !== "gemma3")
    throw new Error(`config.json: model_type must be gemma3_text or gemma3, got ${String(raw.model_type)}`);
  const act = raw.hidden_activation ?? raw.hidden_act ?? "gelu_pytorch_tanh";
  if (act !== "gelu_pytorch_tanh" && act !== "gelu_approx")
    throw new Error(`config.json: hidden_activation must be gelu_pytorch_tanh, got ${String(act)}`);
  if (raw.tie_word_embeddings === false) throw new Error("config.json: tie_word_embeddings must be true");
  for (const k of ["attention_bias", "mlp_bias"]) if (!falsy(raw[k])) throw new Error(`config.json: ${k} is not supported`);

  const dim = intField(raw, "hidden_size", what);
  const heads = intField(raw, "num_attention_heads", what);
  const kvHeads = intField(raw, "num_key_value_heads", what, 1);
  const hd = intField(raw, "head_dim", what, Math.floor(dim / heads));
  const maxPos = intField(raw, "max_position_embeddings", what, 32768);
  const window = raw.sliding_window !== undefined && raw.sliding_window !== null
    ? intField(raw, "sliding_window", what) : 512;
  const theta = typeof raw.rope_theta === "number" ? numField(raw, "rope_theta", what) : 1_000_000;
  const localTheta = typeof raw.rope_local_base_freq === "number"
    ? numField(raw, "rope_local_base_freq", what) : 10_000;

  const config: ModelConfig = {
    architecture: "smolmoe_gemma3",
    dim,
    hidden_dim: intField(raw, "intermediate_size", what),
    layers: intField(raw, "num_hidden_layers", what),
    heads,
    kv_heads: kvHeads,
    experts: 1,
    top_k: 1,
    vocab_size: intField(raw, "vocab_size", what, 262144),
    max_context: Math.min(maxPos, 32768),
    shared_experts: 0,
    bos_id: 256,
    eos_id: 257,
    rms_eps: typeof raw.rms_norm_eps === "number" ? numField(raw, "rms_norm_eps", what) : 1e-6,
    rope_theta: theta,
    head_dim: hd,
    sliding_window: window,
    rope_local_theta: localTheta,
  };
  return parseConfig(formatConfig(config));
}

export interface GemmaTokenizer {
  readonly bytes: Uint8Array;
  readonly oldIds: Uint32Array;
}

/** Ensures <tool_call> and </tool_call> exist in special tokens (reusing <unused:...> slots when absent). */
function ensureToolMarkers(pieces: Uint8Array[], special: Uint8Array): void {
  const encoder = new TextEncoder(), decoder = new TextDecoder();
  for (const marker of ["<tool_call>", "</tool_call>"]) {
    const exists = pieces.some((p, i) => i >= 258 && special[i] === 1 && decoder.decode(p) === marker);
    if (exists) continue;
    const slot = pieces.findIndex((p, i) => i >= 258 && special[i] === 1 && /^<unused[:\d]/.test(decoder.decode(p)));
    if (slot >= 258) pieces[slot] = encoder.encode(marker);
  }
}

/** Converts either SentencePiece BPE (<0x00>..<0xFF> + ▁) or GPT-2 byte-level BPE into SMQBP1 v2 (template=gemma). */
export function gemma3Tokenizer(json: unknown, vocabSize: number): GemmaTokenizer {
  const t = asObject(json, "tokenizer.json");
  const hf = readHfBpe(t);
  for (const req of ["<bos>", "<end_of_turn>", "<start_of_turn>"]) {
    if (!hf.added.some((a) => a.content === req) && !hf.vocab.has(req))
      throw new Error(`tokenizer.json: missing required Gemma token ${req}`);
  }
  const hasByteFallback = hf.vocab.has("<0x00>") && hf.vocab.has("<0xFF>");
  if (!hasByteFallback) {
    const built = buildTokenizer(hf, { vocabSize, bos: "<bos>", eos: "<end_of_turn>", flagUnreachable: true });
    const mutablePieces = [...built.pieces];
    ensureToolMarkers(mutablePieces, built.special);
    const bytes = encodeQbp({
      header: { version: 2, normalizer: NORMALIZER.none, pretokenizer: PRETOKENIZER.gemma, template: TEMPLATE.gemma },
      pieces: mutablePieces, special: built.special, merges: built.merges,
    });
    parseQbp(bytes, vocabSize);
    return { bytes, oldIds: built.oldIds };
  }

  const encoder = new TextEncoder();
  const findId = (content: string): number => {
    const a = hf.added.find((x) => x.content === content);
    if (a) return a.id;
    const v = hf.vocab.get(content);
    if (v !== undefined) return v;
    throw new Error(`tokenizer.json: missing token ${content}`);
  };
  const isControlTag = (s: string) => /^<.*>$|^\[.*\]$/.test(s);
  const head: number[] = [];
  for (let b = 0; b < 256; b++) {
    const ch = b === 0x20 ? "\u2581" : b < 0x80 ? String.fromCharCode(b) : null;
    const charId = ch !== null && !isControlTag(ch) ? hf.vocab.get(ch) : undefined;
    if (charId !== undefined) {
      head.push(charId);
    } else {
      const hex = `<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`;
      head.push(findId(hex));
    }
  }
  head.push(findId("<bos>"), findId("<end_of_turn>"));
  const oldIds = new Uint32Array(vocabSize);
  const toNew = new Uint32Array(vocabSize).fill(0xffffffff);
  head.forEach((id, n) => {
    if (id >= vocabSize || toNew[id] !== 0xffffffff) throw new Error(`tokenizer.json: duplicate or out-of-range head id ${id}`);
    oldIds[n] = id;
    toNew[id] = n;
  });
  let nextNew = head.length;
  for (let id = 0; id < vocabSize; id++) {
    if (toNew[id] === 0xffffffff) {
      oldIds[nextNew] = id;
      toNew[id] = nextNew++;
    }
  }

  const pieces = new Array<Uint8Array>(vocabSize);
  const special = new Uint8Array(vocabSize).fill(1);
  for (let b = 0; b < 256; b++) {
    pieces[b] = Uint8Array.of(b);
    special[b] = 0;
  }
  const spPieceBytes = (piece: string): Uint8Array => {
    const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
    if (m) return Uint8Array.of(parseInt(m[1]!, 16));
    return encoder.encode(piece.replaceAll("\u2581", " "));
  };
  for (const [piece, id] of hf.vocab) {
    if (id >= vocabSize) continue;
    const nid = toNew[id]!;
    if (nid < 258) continue;
    const raw = spPieceBytes(piece);
    pieces[nid] = raw.length > 0 ? raw : encoder.encode(`<empty:${id}>`);
    /* Single-byte tokens at nid >= 258 (e.g. redundant <0xXX> fallbacks) and <...> tags are marked special. */
    special[nid] = raw.length <= 1 || isControlTag(piece) ? 1 : 0;
  }
  for (const a of hf.added) {
    if (a.id >= vocabSize) continue;
    const nid = toNew[a.id]!;
    if (nid < 256) continue;
    const raw = spPieceBytes(a.content);
    pieces[nid] = raw.length > 0 ? raw : encoder.encode(a.content);
    special[nid] = raw.length <= 1 || isControlTag(a.content) ? 1 : 0;
  }
  for (let n = 0; n < vocabSize; n++) {
    if (!pieces[n]) {
      pieces[n] = encoder.encode(`<unused:${oldIds[n]}>`);
      special[n] = 1;
    }
  }

  const merges: number[] = [];
  const seen = new Set<number>();
  const addMerge = (na: number, nb: number, no: number) => {
    if (special[na] || special[nb] || special[no]) return;
    const key = na * 0x80000 + nb;
    if (seen.has(key)) return;
    seen.add(key);
    merges.push(na, nb, no);
  };
  /* Prepend 2-byte UTF-8 codepoint merges (e.g. accented Latin characters like é, ã, ç) so byte-level BPE forms them first. */
  for (const [piece, id] of hf.vocab) {
    if (id >= vocabSize || [...piece].length !== 1 || piece === "\u2581") continue;
    const nid = toNew[id]!;
    const bytes = pieces[nid]!;
    if (nid >= 258 && !special[nid] && bytes.length === 2) addMerge(bytes[0]!, bytes[1]!, nid);
  }
  const resolveMergeOperand = (piece: string): number => {
    const bytes = spPieceBytes(piece);
    if (bytes.length === 1) return bytes[0]!;
    const id = hf.vocab.get(piece);
    if (id === undefined) throw new Error(`tokenizer.json: unknown merge piece ${piece}`);
    return toNew[id]!;
  };
  for (const [a, b] of hf.merges) {
    const outId = hf.vocab.get(a + b);
    if (outId === undefined || outId >= vocabSize) continue;
    addMerge(resolveMergeOperand(a), resolveMergeOperand(b), toNew[outId]!);
  }

  ensureToolMarkers(pieces, special);
  const bytes = encodeQbp({
    header: { version: 2, normalizer: NORMALIZER.none, pretokenizer: PRETOKENIZER.gemma, template: TEMPLATE.gemma },
    pieces, special, merges: new Uint32Array(merges),
  });
  parseQbp(bytes, vocabSize);
  return { bytes, oldIds };
}

export function gemma3Plan(c: ModelConfig, ckpt: Checkpoint, oldIds: Uint32Array): ModelPlan {
  const prefix = ckpt.tensors.has("language_model.model.embed_tokens.weight") ? "language_model.model." : "model.";
  const get = (name: string) => ckpt.tensors.get(name) ?? (() => { throw new Error(`missing source tensor ${name}`); })();
  const hd = c.head_dim ?? (c.dim / c.heads), qd = c.heads * hd, kd = c.kv_heads * hd;
  const expectShape = (name: string, shape: readonly number[]) => {
    const t = get(name);
    if (t.shape.join(",") !== shape.join(",")) throw new Error(`${name}: shape [${t.shape.join(", ")}], expected [${shape.join(", ")}]`);
    return t;
  };
  const rows = (name: string, shape: readonly number[], map: Uint32Array): TensorSource =>
    ({ kind: "rows", tensor: expectShape(name, shape), rows: map });
  const normVec = (name: string, len: number, permuteHead = false): TensorSource => ({
    kind: "vector",
    tensor: expectShape(name, [len]),
    addOne: true,
    ...(permuteHead ? { heads: { count: 1, headDim: hd, rotaryDim: hd } } : {}),
  });
  const s = new Map<string, TensorSource>([
    ["embed", rows(`${prefix}embed_tokens.weight`, [c.vocab_size, c.dim], oldIds)],
    ["final_norm", normVec(`${prefix}norm.weight`, c.dim)],
    ["lm_head", { kind: "alias" }],
  ]);
  for (let l = 0; l < c.layers; l++) {
    const p = `${prefix}layers.${l}.`, n = `l.${l}.`;
    s.set(`${n}attn_norm`, normVec(`${p}input_layernorm.weight`, c.dim));
    s.set(`${n}post_attn_norm`, normVec(`${p}post_attention_layernorm.weight`, c.dim));
    s.set(`${n}ffn_norm`, normVec(`${p}pre_feedforward_layernorm.weight`, c.dim));
    s.set(`${n}post_ffn_norm`, normVec(`${p}post_feedforward_layernorm.weight`, c.dim));
    s.set(`${n}q_norm`, normVec(`${p}self_attn.q_norm.weight`, hd, true));
    s.set(`${n}k_norm`, normVec(`${p}self_attn.k_norm.weight`, hd, true));
    s.set(`${n}q`, rows(`${p}self_attn.q_proj.weight`, [qd, c.dim], qkRows(0, c.heads, hd, hd)));
    s.set(`${n}k`, rows(`${p}self_attn.k_proj.weight`, [kd, c.dim], qkRows(0, c.kv_heads, hd, hd)));
    s.set(`${n}v`, rows(`${p}self_attn.v_proj.weight`, [kd, c.dim], rangeRows(0, kd)));
    s.set(`${n}o`, rows(`${p}self_attn.o_proj.weight`, [c.dim, qd], rangeRows(0, c.dim)));
    s.set(`${n}router`, { kind: "zeros" });
    for (const m of ["gate", "up"] as const)
      s.set(`${n}e.0.${m}`, rows(`${p}mlp.${m}_proj.weight`, [c.hidden_dim, c.dim], rangeRows(0, c.hidden_dim)));
    s.set(`${n}e.0.down`, rows(`${p}mlp.down_proj.weight`, [c.dim, c.hidden_dim], rangeRows(0, c.dim)));
  }
  const plan = { config: c, sources: s };
  checkPlan(plan, ckpt);
  return plan;
}

export async function convertGemma3(o: ConvertOptions): Promise<ConvertReport> {
  const started = performance.now();
  const configPath = join(o.source, "config.json"), tokenizerPath = join(o.source, "tokenizer.json");
  const config = gemma3Config(readJsonFile(configPath));
  const tokenizer = gemma3Tokenizer(readJsonFile(tokenizerPath), config.vocab_size);
  const ckpt = openCheckpoint(o.source);
  const plan = gemma3Plan(config, ckpt, tokenizer.oldIds);
  mkdirSync(o.output, { recursive: true });
  const index = join(o.source, "model.safetensors.index.json");
  const hashes = sha256Files([...ckpt.files, configPath, tokenizerPath, ...(existsSync(index) ? [index] : [])]);
  hashes.catch(() => undefined);
  const files = [join(o.output, "config.ini"), join(o.output, "tokenizer.qbp"), join(o.output, "model.smoe"),
    join(o.output, "provenance.json")];
  const beforeRename = async () => {
    const h = await hashes;
    const byName = Object.fromEntries([...h].map(([path, d]) => [basename(path), d]));
    writeAtomic(files[0]!, formatConfig(config));
    writeAtomic(files[1]!, tokenizer.bytes);
    writeAtomic(files[3]!, `${JSON.stringify({
      model: "Gemma 3 checkpoint", converter: CONVERTER, source_sha256: byName,
      quantization: "lossy Q8 block32",
      norms: "Gemma (1 + weight) baked into F32 norm vectors; q_norm/k_norm permuted to adjacent-pair rotary",
      tied_lm_head: true,
    }, null, 2)}\n`);
  };
  o.log(`converting ${config.layers} layers (${ckpt.files.length} shard(s)) with ${o.threads} thread(s)`);
  await writeModel(plan, { output: files[2]!, container: o.container, threads: o.threads, tokenizer: tokenizer.bytes,
    log: o.log, beforeRename });
  return report([...files, copyLicense(o.source, o.output, "LICENSE-Gemma.txt")], started);
}
