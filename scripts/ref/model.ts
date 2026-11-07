/**
 * Whole-model causal forward of Phi-3 / Phi-4-mini in float64, reading the
 * original BF16 checkpoint tensor by tensor (independent reference).
 *
 * Phi3Model.forward (transformers v4.49 modeling_phi3.py = checkpoint remote code):
 *   x = embed_tokens(ids)                      (no scaling)
 *   per layer: x = x + o_proj(attn(input_layernorm(x)))
 *              x = x + down_proj(up * silu(gate)), gate|up = gate_up_proj(post_attention_layernorm(x))
 *   x = norm(x); logits = lm_head(x) with lm_head tied to embed_tokens.
 * Positions are 0..T-1 (cache_position = arange(0, T)); cos/sin are computed once
 * per call and shared by every layer.
 */
import type { Phi3Config } from "./config";
import { causalAttention, qkvLayout, swiGlu } from "./layers";
import { streamLinear } from "./linear";
import { attentionFactor, invFreq, rmsNorm, ropeTable, selectRopeFactors, type RopeTable } from "./math";
import type { WeightStore } from "./safetensors";

export const EMBED = "model.embed_tokens.weight";
export const FINAL_NORM = "model.norm.weight";
const layerName = (l: number, suffix: string) => `model.layers.${l}.${suffix}`;

export interface RopeSetup {
  readonly attentionFactor: number;
  readonly invFreq: Float64Array;
  readonly factorKind: "short" | "long";
  readonly table: RopeTable;
}

/** LongRoPE for positions 0..T-1: factor choice from the max position, attention factor always applied. */
export function ropeForPositions(cfg: Phi3Config, T: number): RopeSetup {
  const positions = Array.from({ length: T }, (_, i) => i);
  const factors = selectRopeFactors(T - 1, cfg.originalMaxPositionEmbeddings, cfg.shortFactor, cfg.longFactor);
  const inv = invFreq(cfg.ropeTheta, cfg.rotaryDim, factors);
  const a = attentionFactor(cfg.maxPositionEmbeddings, cfg.originalMaxPositionEmbeddings);
  return { attentionFactor: a, invFreq: inv, factorKind: factors === cfg.longFactor ? "long" : "short", table: ropeTable(positions, inv, a) };
}

function expectShape(store: WeightStore, name: string, want: readonly number[]): void {
  const got = store.shape(name);
  if (got.length !== want.length || got.some((d, i) => d !== want[i])) {
    throw new Error(`${name}: shape [${got}], expected [${want}]`);
  }
}

/** Checks every tensor this forward touches before any heavy work starts. */
export function validateCheckpoint(store: WeightStore, cfg: Phi3Config): void {
  const H = cfg.hiddenSize;
  const qDim = cfg.numHeads * cfg.headDim;
  const { width } = qkvLayout(cfg);
  if (store.has("lm_head.weight")) throw new Error("checkpoint has lm_head.weight, but this reference assumes tied embeddings");
  expectShape(store, EMBED, [cfg.vocabSize, H]);
  expectShape(store, FINAL_NORM, [H]);
  for (let l = 0; l < cfg.numLayers; l++) {
    expectShape(store, layerName(l, "input_layernorm.weight"), [H]);
    expectShape(store, layerName(l, "self_attn.qkv_proj.weight"), [width, H]);
    expectShape(store, layerName(l, "self_attn.o_proj.weight"), [H, qDim]);
    expectShape(store, layerName(l, "post_attention_layernorm.weight"), [H]);
    expectShape(store, layerName(l, "mlp.gate_up_proj.weight"), [2 * cfg.intermediateSize, H]);
    expectShape(store, layerName(l, "mlp.down_proj.weight"), [H, cfg.intermediateSize]);
  }
}

/** Embedding lookup: reads only the rows of the given ids. */
export function embed(store: WeightStore, cfg: Phi3Config, ids: readonly number[]): Float64Array {
  const H = cfg.hiddenSize;
  const out = new Float64Array(ids.length * H);
  ids.forEach((id, t) => {
    if (!Number.isSafeInteger(id) || id < 0 || id >= cfg.vocabSize) throw new RangeError(`token id ${id} outside [0, ${cfg.vocabSize})`);
    out.set(store.readRows(EMBED, id, 1), t * H);
  });
  return out;
}

/** RMSNorm applied to each of the T rows of x. */
export function normRows(x: Float64Array, T: number, weight: Float32Array, eps: number): Float64Array {
  const H = weight.length;
  const out = new Float64Array(T * H);
  for (let t = 0; t < T; t++) out.set(rmsNorm(x.subarray(t * H, (t + 1) * H), weight, eps), t * H);
  return out;
}

function add(a: Float64Array, b: Float64Array): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! + b[i]!;
  return out;
}

/** One Phi3DecoderLayer over all T positions; returns the new residual stream. */
export function decoderLayer(store: WeightStore, cfg: Phi3Config, l: number, x: Float64Array, T: number, rope: RopeTable): Float64Array {
  const eps = cfg.rmsNormEps;
  const h1 = normRows(x, T, store.readVector(layerName(l, "input_layernorm.weight")), eps);
  const qkv = streamLinear(store, layerName(l, "self_attn.qkv_proj.weight"), h1, T);
  const attn = causalAttention(qkv, T, cfg, rope);
  const x1 = add(x, streamLinear(store, layerName(l, "self_attn.o_proj.weight"), attn, T));
  const h2 = normRows(x1, T, store.readVector(layerName(l, "post_attention_layernorm.weight")), eps);
  const gateUp = streamLinear(store, layerName(l, "mlp.gate_up_proj.weight"), h2, T);
  const mlp = streamLinear(store, layerName(l, "mlp.down_proj.weight"), swiGlu(gateUp, T, cfg.intermediateSize), T);
  return add(x1, mlp);
}

export interface ForwardOptions {
  /** Progress lines (stderr in the CLI). */
  readonly log?: (message: string) => void;
  /**
   * Residual-stream snapshots: index 0 = embeddings, l + 1 = output of layer l,
   * numLayers + 1 = after the final norm. Each is T x hidden_size.
   */
  readonly onHidden?: (index: number, hidden: Float64Array) => void;
}

export interface ForwardResult {
  /** T x vocab_size, position-major. */
  readonly logits: Float64Array;
  readonly rope: RopeSetup;
}

export function runForward(store: WeightStore, cfg: Phi3Config, ids: readonly number[], options: ForwardOptions = {}): ForwardResult {
  const T = ids.length;
  if (T === 0) throw new RangeError("need at least one token");
  if (cfg.slidingWindow !== null && T > cfg.slidingWindow) throw new RangeError("sequence longer than the sliding window");
  const log = options.log ?? (() => undefined);
  const rope = ropeForPositions(cfg, T);
  let x = embed(store, cfg, ids);
  options.onHidden?.(0, x);
  for (let l = 0; l < cfg.numLayers; l++) {
    const started = performance.now();
    x = decoderLayer(store, cfg, l, x, T, rope.table);
    options.onHidden?.(l + 1, x);
    log(`layer ${l + 1}/${cfg.numLayers} ${((performance.now() - started) / 1000).toFixed(2)} s`);
  }
  const final = normRows(x, T, store.readVector(FINAL_NORM), cfg.rmsNormEps);
  options.onHidden?.(cfg.numLayers + 1, final);
  const started = performance.now();
  const logits = streamLinear(store, EMBED, final, T); // tied lm_head
  log(`lm_head ${((performance.now() - started) / 1000).toFixed(2)} s`);
  return { logits, rope };
}
