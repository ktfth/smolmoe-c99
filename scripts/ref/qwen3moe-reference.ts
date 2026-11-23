#!/usr/bin/env bun
/**
 * Independent float64 reference forward pass of Qwen3-MoE (Qwen3-30B-A3B),
 * computed straight from the original BF16 safetensors with Hugging Face
 * semantics. It shares no code with the C runtime or the converter: original
 * token ids, split-half RoPE, unpermuted Q/K, the router in BF16.
 *
 * transformers v4.51 models/qwen3_moe/modeling_qwen3_moe.py:
 *   Qwen3MoeAttention.forward: q = q_norm(q_proj(h).view(T, H, D)), k = k_norm(k_proj(h).view(T, KV, D)),
 *     v = v_proj(h); apply_rotary_pos_emb (rotate_half); softmax(q k^T * D^-0.5 + causal) v; o_proj.
 *   Qwen3MoeSparseMoeBlock.forward: router_logits = gate(h); w = softmax(router_logits, fp32);
 *     w, sel = topk(w, top_k); if norm_topk_prob: w /= w.sum(); out = sum_k w_k * expert_sel_k(h),
 *     expert(h) = down_proj(silu(gate_proj(h)) * up_proj(h)).
 *   Decoder layer: x += attn(input_layernorm(x)); x += moe(post_attention_layernorm(x)).
 *   Model: x = embed(ids); layers; norm; lm_head (untied).
 *
 * usage: bun scripts/ref/qwen3moe-reference.ts <model_dir> --ids 151644,872,... [--out logits.f32] [--top 5]
 *          [--layers N]
 * --out     logits for every position, little-endian float32, positions x vocab_size, original ids.
 * --layers  run only the first N layers (a quick partial check; the logits are then not the model's).
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { streamLinear, topK } from "./linear";
import { applyRope, invFreq, kvHeadFor, rmsNorm, ropeTable, silu, softmax } from "./math";
import { WeightStore } from "./safetensors";

interface Qwen3MoeConfig {
  readonly dim: number;
  readonly layers: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly experts: number;
  readonly topK: number;
  readonly vocab: number;
  readonly eps: number;
  readonly theta: number;
}

export function loadConfig(dir: string): Qwen3MoeConfig {
  const raw = JSON.parse(readFileSync(join(dir, "config.json"), "utf8")) as Record<string, unknown>;
  const n = (k: string): number => {
    const v = raw[k];
    if (typeof v !== "number") throw new Error(`config.json: ${k} is not a number`);
    return v;
  };
  if (raw.model_type !== "qwen3_moe" || raw.norm_topk_prob !== true || raw.tie_word_embeddings)
    throw new Error("config.json: expected qwen3_moe with norm_topk_prob and an untied head");
  return { dim: n("hidden_size"), layers: n("num_hidden_layers"), heads: n("num_attention_heads"),
    kvHeads: n("num_key_value_heads"), headDim: n("head_dim"), experts: n("num_experts"),
    topK: n("num_experts_per_tok"), vocab: n("vocab_size"), eps: n("rms_norm_eps"), theta: n("rope_theta") };
}

/** Row t of a T x width matrix. */
const row = (m: Float64Array, t: number, width: number): Float64Array => m.subarray(t * width, (t + 1) * width);

function normRows(x: Float64Array, T: number, width: number, weight: Float32Array, eps: number): Float64Array {
  const out = new Float64Array(T * width);
  for (let t = 0; t < T; t++) out.set(rmsNorm(row(x, t, width), weight, eps), t * width);
  return out;
}

function attention(store: WeightStore, c: Qwen3MoeConfig, l: number, h: Float64Array, T: number): Float64Array {
  const p = `model.layers.${l}.self_attn.`, D = c.headDim;
  const q = streamLinear(store, `${p}q_proj.weight`, h, T);
  const k = streamLinear(store, `${p}k_proj.weight`, h, T);
  const v = streamLinear(store, `${p}v_proj.weight`, h, T);
  const qNorm = store.readVector(`${p}q_norm.weight`), kNorm = store.readVector(`${p}k_norm.weight`);
  const rope = ropeTable(Array.from({ length: T }, (_, i) => i), invFreq(c.theta, D, new Array(D / 2).fill(1)), 1);
  const head = (m: Float64Array, t: number, i: number, count: number, w: Float32Array): Float64Array => {
    const cos = rope.cos.subarray(t * rope.half, (t + 1) * rope.half), sin = rope.sin.subarray(t * rope.half, (t + 1) * rope.half);
    return applyRope(rmsNorm(m.subarray(t * count * D + i * D, t * count * D + (i + 1) * D), w, c.eps), cos, sin);
  };
  const keys = Array.from({ length: T }, (_, t) => Array.from({ length: c.kvHeads }, (_, g) => head(k, t, g, c.kvHeads, kNorm)));
  const out = new Float64Array(T * c.heads * D);
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < c.heads; i++) {
      const g = kvHeadFor(i, c.heads, c.kvHeads), qh = head(q, t, i, c.heads, qNorm);
      const scores = new Float64Array(t + 1);
      for (let j = 0; j <= t; j++) {
        let s = 0;
        const kh = keys[j]![g]!;
        for (let d = 0; d < D; d++) s += qh[d]! * kh[d]!;
        scores[j] = s * D ** -0.5;
      }
      const w = softmax(scores), base = (t * c.heads + i) * D;
      for (let j = 0; j <= t; j++)
        for (let d = 0; d < D; d++) out[base + d] = out[base + d]! + w[j]! * v[j * c.kvHeads * D + g * D + d]!;
    }
  }
  return streamLinear(store, `${p}o_proj.weight`, out, T);
}

/** The sparse block for T positions; each expert runs only on the positions that chose it. */
function moe(store: WeightStore, c: Qwen3MoeConfig, l: number, h: Float64Array, T: number): Float64Array {
  const p = `model.layers.${l}.mlp.`;
  const logits = streamLinear(store, `${p}gate.weight`, h, T);
  const chosen = new Map<number, { t: number; w: number }[]>();
  for (let t = 0; t < T; t++) {
    const probs = softmax(row(logits, t, c.experts));
    const best = topK(probs, 0, c.experts, c.topK);
    const sum = best.reduce((s, b) => s + b.logit, 0);
    for (const b of best) {
      const list = chosen.get(b.id) ?? [];
      list.push({ t, w: b.logit / sum });
      chosen.set(b.id, list);
    }
  }
  const out = new Float64Array(T * c.dim);
  for (const [e, users] of chosen) {
    const x = new Float64Array(users.length * c.dim);
    users.forEach((u, i) => x.set(row(h, u.t, c.dim), i * c.dim));
    const ep = `${p}experts.${e}.`, n = users.length;
    const gate = streamLinear(store, `${ep}gate_proj.weight`, x, n), up = streamLinear(store, `${ep}up_proj.weight`, x, n);
    for (let i = 0; i < gate.length; i++) gate[i] = silu(gate[i]!) * up[i]!;
    const y = streamLinear(store, `${ep}down_proj.weight`, gate, n);
    users.forEach((u, i) => {
      for (let d = 0; d < c.dim; d++) out[u.t * c.dim + d] = out[u.t * c.dim + d]! + u.w * y[i * c.dim + d]!;
    });
  }
  return out;
}

export function forward(store: WeightStore, c: Qwen3MoeConfig, ids: readonly number[], layers = c.layers,
  log: (s: string) => void = () => undefined): Float64Array {
  const T = ids.length;
  const x = new Float64Array(T * c.dim);
  ids.forEach((id, t) => x.set(store.readRows("model.embed_tokens.weight", id, 1), t * c.dim));
  for (let l = 0; l < layers; l++) {
    const started = performance.now();
    const p = `model.layers.${l}.`;
    const a = attention(store, c, l, normRows(x, T, c.dim, store.readVector(`${p}input_layernorm.weight`), c.eps), T);
    for (let i = 0; i < x.length; i++) x[i] = x[i]! + a[i]!;
    const m = moe(store, c, l, normRows(x, T, c.dim, store.readVector(`${p}post_attention_layernorm.weight`), c.eps), T);
    for (let i = 0; i < x.length; i++) x[i] = x[i]! + m[i]!;
    log(`layer ${l} in ${((performance.now() - started) / 1000).toFixed(1)} s`);
  }
  return streamLinear(store, "lm_head.weight", normRows(x, T, c.dim, store.readVector("model.norm.weight"), c.eps), T);
}

function main(argv: readonly string[]): void {
  const dir = argv[0];
  const flag = (name: string): string | undefined => {
    const i = argv.indexOf(name);
    return i > 0 ? argv[i + 1] : undefined;
  };
  const idsText = flag("--ids");
  if (!dir || !idsText) throw new Error("usage: qwen3moe-reference.ts <model_dir> --ids a,b,... [--out F] [--top K] [--layers N]");
  const ids = idsText.split(",").map((s) => Number(s.trim()));
  const c = loadConfig(dir);
  if (ids.some((id) => !Number.isSafeInteger(id) || id < 0 || id >= c.vocab)) throw new Error("--ids out of range");
  const layers = Number(flag("--layers") ?? c.layers), top = Number(flag("--top") ?? 5);
  const store = WeightStore.open(dir);
  try {
    const logits = forward(store, c, ids, layers, (s) => console.error(`ref: ${s}`));
    const out = flag("--out");
    if (out) writeFileSync(out, new Uint8Array(Float32Array.from(logits).buffer));
    for (let t = 0; t < ids.length; t++) {
      const best = topK(logits, t * c.vocab, c.vocab, top).map((b) => `${b.id}:${b.logit.toFixed(3)}`);
      process.stdout.write(`${t} ${ids[t]} -> ${best.join(" ")}\n`);
    }
  } finally {
    store.close();
  }
}

if (import.meta.main) main(Bun.argv.slice(2));
