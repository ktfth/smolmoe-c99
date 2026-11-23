/**
 * Qwen3 mixture of experts (Qwen3MoeForCausalLM) -> smolmoe: the pinned
 * Qwen3-30B-A3B-Instruct-2507 (128 experts, 8 per token, ~3B active of 30B,
 * Apache 2.0). The tokenizer and the chat template are Qwen2.5's family
 * (ChatML), so the chat and its tools work unchanged.
 */
import { existsSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";

import { type ModelConfig, formatConfig, parseConfig } from "../smoe/config";
import { type ConvertOptions, type ConvertReport, asObject, copyLicense, intField, numField, readJsonFile, report,
  writeAtomic } from "./common";
import type { PinnedFile } from "./fetch";
import { sha256Files } from "./hash";
import { qkRows, rangeRows } from "./permute";
import { type ModelPlan, type TensorSource, checkPlan, writeModel } from "./pipeline";
import { qwen2Tokenizer } from "./qwen2";
import { type Checkpoint, type HfTensor, openCheckpoint } from "./safetensors";

export interface Qwen3MoePin {
  readonly repo: string;
  readonly revision: string;
  readonly shards: readonly { readonly name: string; readonly sha256: string }[];
  readonly tokenizerSha256: string;
}

export const QWEN3_30B_A3B: Qwen3MoePin = {
  repo: "Qwen/Qwen3-30B-A3B-Instruct-2507",
  revision: "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
  tokenizerSha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
  shards: [
    ["afe775cbdefb257e1bf1d8551fb930bd4b78d74da0b10bbb651f6b6a6f67c325"],
    ["c0eefe9ed74050f4144fcb1b614c6915462e756b6673a9cca67f446b00c3434f"],
    ["18ebd81e648667b1513786fc75886044e8ccb06b04562a35cbea59a41e4bcbbf"],
    ["5df2f491c8c96651963a0d84db89d7647febc0e5f4f96cc67c2edac2d4a731b2"],
    ["276477e938cd0de42c967faa7c10bc03a1a4b3e82135d5e17469152d0dd5374e"],
    ["ac03cec889c40085113ddcb8f73216a157e7b83f06bb37aaf6a4699b84fa4abe"],
    ["382aee85a70fccca2bde6043a0aefc16f3fd2e87e2a8d1ccf1eba442f7a98e92"],
    ["effec719bf78d74170f294d94f1240b8f6f30c5402a04a3dc2cb42c8cdaa101a"],
    ["6f272f00be2100f34d3a0463fb6e66b106f015bbc9aaba02b1898a102938d7c1"],
    ["c2bc4101cde6b7c9dda36bbdff815c2ff19541d7732e94d306d87a0806c133b3"],
    ["adb34c5b4c1e4c85527218638c246181ffb7763ed033c0f3d09f5242e7022712"],
    ["d4276b408eca9c249fd317f725074e9a443107ca6cfd518d5cea3911f1cefce2"],
    ["c7070b4e079477665f2a6f5a26704bdcbd757afa4b7edc3d628128abe9157c51"],
    ["c50e816929903e80e9e6ca4d6eec3b643a50478e61407495fbc10f3c999b0fd5"],
    ["5351255875a9b79507db1357e61f109ec880fd3e4a2000d5f7c51494aa77d288"],
    ["fa5623e55641935f938894cc91090732452ef22da12596e857c8dff4fa71627b"],
  ].map(([sha256], i) => ({ name: `model-${String(i + 1).padStart(5, "0")}-of-00016.safetensors`, sha256: sha256! })),
};

/**
 * The files to download. config.json and the index are plain git files (no
 * LFS hash); they come from the same pinned revision, and the converter
 * checks every shape against them.
 */
export function qwen3MoeFiles(pin: Qwen3MoePin): readonly PinnedFile[] {
  return [
    { name: "config.json", revision: pin.revision },
    { name: "model.safetensors.index.json", revision: pin.revision },
    { name: "tokenizer.json", revision: pin.revision, sha256: pin.tokenizerSha256 },
    { name: "LICENSE", revision: pin.revision },
    ...pin.shards.map((s) => ({ name: s.name, revision: pin.revision, sha256: s.sha256 })),
  ];
}

/** config.json -> config.ini: the gate on what this converter understands, then the runtime's own validation. */
export function qwen3MoeConfig(json: unknown): ModelConfig {
  const what = "config.json";
  const raw = asObject(json, what);
  if (raw.model_type !== "qwen3_moe" || raw.hidden_act !== "silu" || raw.attention_bias || raw.use_sliding_window ||
    raw.rope_scaling || raw.tie_word_embeddings || raw.norm_topk_prob !== true ||
    (raw.decoder_sparse_step !== undefined && raw.decoder_sparse_step !== 1) ||
    (Array.isArray(raw.mlp_only_layers) && raw.mlp_only_layers.length > 0))
    throw new Error(`${what}: only Qwen3-MoE with every layer sparse, SiLU, no bias, full attention, renormalized top-k`);
  const config: ModelConfig = {
    architecture: "smolmoe_qwen3", dim: intField(raw, "hidden_size", what),
    hidden_dim: intField(raw, "moe_intermediate_size", what), layers: intField(raw, "num_hidden_layers", what),
    heads: intField(raw, "num_attention_heads", what), kv_heads: intField(raw, "num_key_value_heads", what),
    experts: intField(raw, "num_experts", what), top_k: intField(raw, "num_experts_per_tok", what),
    vocab_size: intField(raw, "vocab_size", what), max_context: intField(raw, "max_position_embeddings", what),
    shared_experts: 0, bos_id: 256, eos_id: 257, rms_eps: numField(raw, "rms_norm_eps", what),
    rope_theta: numField(raw, "rope_theta", what), head_dim: intField(raw, "head_dim", what),
  };
  return parseConfig(formatConfig(config));
}

export function qwen3MoePlan(c: ModelConfig, ckpt: Checkpoint, oldIds: Uint32Array): ModelPlan {
  const hd = c.head_dim ?? c.dim / c.heads, qd = c.heads * hd, kd = c.kv_heads * hd;
  const get = (name: string, shape: readonly number[]): HfTensor => {
    const t = ckpt.tensors.get(name);
    if (!t) throw new Error(`missing source tensor ${name}`);
    if (t.dtype !== "BF16") throw new Error(`${name} is ${t.dtype}, expected BF16`);
    if (t.shape.join(",") !== shape.join(",")) throw new Error(`${name}: shape [${t.shape.join(", ")}], expected [${shape.join(", ")}]`);
    return t;
  };
  const rows = (name: string, shape: readonly [number, number], map?: Uint32Array): TensorSource =>
    ({ kind: "rows", tensor: get(name, shape), rows: map ?? rangeRows(0, shape[0]) });
  const vector = (name: string, len: number, perHead = false): TensorSource =>
    ({ kind: "vector", tensor: get(name, [len]), ...(perHead ? { heads: { count: 1, headDim: hd, rotaryDim: hd } } : {}) });
  const s = new Map<string, TensorSource>([
    ["embed", rows("model.embed_tokens.weight", [c.vocab_size, c.dim], oldIds)],
    ["final_norm", vector("model.norm.weight", c.dim)],
    ["lm_head", rows("lm_head.weight", [c.vocab_size, c.dim], oldIds)],
  ]);
  for (let l = 0; l < c.layers; l++) {
    const p = `model.layers.${l}.`, n = `l.${l}.`;
    s.set(`${n}attn_norm`, vector(`${p}input_layernorm.weight`, c.dim));
    s.set(`${n}ffn_norm`, vector(`${p}post_attention_layernorm.weight`, c.dim));
    /* RoPE pairs adjacent coordinates here (split halves in transformers): Q, K and their norms are permuted alike. */
    s.set(`${n}q_norm`, vector(`${p}self_attn.q_norm.weight`, hd, true));
    s.set(`${n}k_norm`, vector(`${p}self_attn.k_norm.weight`, hd, true));
    s.set(`${n}q`, rows(`${p}self_attn.q_proj.weight`, [qd, c.dim], qkRows(0, c.heads, hd, hd)));
    s.set(`${n}k`, rows(`${p}self_attn.k_proj.weight`, [kd, c.dim], qkRows(0, c.kv_heads, hd, hd)));
    s.set(`${n}v`, rows(`${p}self_attn.v_proj.weight`, [kd, c.dim]));
    s.set(`${n}o`, rows(`${p}self_attn.o_proj.weight`, [c.dim, qd]));
    s.set(`${n}router`, rows(`${p}mlp.gate.weight`, [c.experts, c.dim]));
    for (let e = 0; e < c.experts; e++) {
      const x = `${p}mlp.experts.${e}.`;
      s.set(`${n}e.${e}.gate`, rows(`${x}gate_proj.weight`, [c.hidden_dim, c.dim]));
      s.set(`${n}e.${e}.up`, rows(`${x}up_proj.weight`, [c.hidden_dim, c.dim]));
      s.set(`${n}e.${e}.down`, rows(`${x}down_proj.weight`, [c.dim, c.hidden_dim]));
    }
  }
  const plan = { config: c, sources: s };
  checkPlan(plan, ckpt);
  return plan;
}

function isPinned(pin: Qwen3MoePin, byName: Readonly<Record<string, string>>): boolean {
  return byName["tokenizer.json"] === pin.tokenizerSha256 && pin.shards.every((s) => byName[s.name] === s.sha256);
}

export async function convertQwen3Moe(o: ConvertOptions, pin: Qwen3MoePin = QWEN3_30B_A3B): Promise<ConvertReport> {
  const started = performance.now();
  const configPath = join(o.source, "config.json"), tokenizerPath = join(o.source, "tokenizer.json");
  const index = join(o.source, "model.safetensors.index.json");
  const config = qwen3MoeConfig(readJsonFile(configPath));
  /* The same pre-tokenizer and ChatML ids as Qwen2.5; the vocabulary adds <think> and the tool tokens. */
  const tokenizer = qwen2Tokenizer(readJsonFile(tokenizerPath), config.vocab_size, o.pinned);
  const ckpt = openCheckpoint(o.source);
  const plan = qwen3MoePlan(config, ckpt, tokenizer.oldIds);
  mkdirSync(o.output, { recursive: true });
  const hashes = sha256Files([...ckpt.files, configPath, tokenizerPath, ...(existsSync(index) ? [index] : [])]);
  hashes.catch(() => undefined); /* observed before the rename below */
  const files = [join(o.output, "config.ini"), join(o.output, "tokenizer.qbp"), join(o.output, "model.smoe"),
    join(o.output, "provenance.json")];
  const beforeRename = async () => {
    const h = await hashes;
    const byName = Object.fromEntries([...h].map(([path, d]) => [basename(path), d]));
    if (o.pinned && !isPinned(pin, byName)) throw new Error(`source files do not match the pinned ${pin.repo} revision`);
    writeAtomic(files[0]!, formatConfig(config));
    writeAtomic(files[1]!, tokenizer.bytes);
    writeAtomic(files[3]!, `${JSON.stringify({
      model: o.pinned ? pin.repo : "unpinned qwen3_moe checkpoint", revision: o.pinned ? pin.revision : "unpinned",
      source_sha256: byName, quantization: "lossy Q8 block32, router included",
      qk_permutation: "split-half rotary to adjacent-pair, including q_norm/k_norm",
      routing: "softmax over the experts, top-k renormalized (norm_topk_prob)", tied_lm_head: false,
    }, null, 2)}\n`);
  };
  o.log(`converting ${config.layers} layers x ${config.experts} experts (${ckpt.files.length} shard(s)) with ${o.threads} thread(s)`);
  await writeModel(plan, { output: files[2]!, container: o.container, threads: o.threads, tokenizer: tokenizer.bytes,
    log: o.log, beforeRename });
  return report([...files, copyLicense(o.source, o.output, "LICENSE-Qwen.txt")], started);
}
