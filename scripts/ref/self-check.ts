/**
 * Self-check for the RoPE setup (spec 10.4): at position 0 the rotation is the
 * identity, so after RoPE the first rotary_dim dims of every q and k head equal
 * attention_factor x (pre-RoPE values) and the remaining dims are unchanged.
 * Checked on a synthetic head and on the real layer-0 q/k of one token.
 */
import type { Phi3Config } from "./config";
import { qkvLayout } from "./layers";
import { streamLinear } from "./linear";
import { applyRope, attentionFactor, invFreq, ropeTable } from "./math";
import { embed, normRows } from "./model";
import type { WeightStore } from "./safetensors";

const TOLERANCE = 1e-12;

/** Max |after - expected| over one head, where expected = A*x on rotary dims and x elsewhere. */
export function positionZeroError(head: Float64Array, rotaryDim: number, inv: Float64Array, a: number): number {
  const table = ropeTable([0], inv, a);
  const after = applyRope(head, table.cos, table.sin);
  let worst = 0;
  for (let d = 0; d < head.length; d++) {
    const want = d < rotaryDim ? a * head[d]! : head[d]!;
    worst = Math.max(worst, Math.abs(after[d]! - want) / Math.max(1, Math.abs(want)));
  }
  return worst;
}

const fmt = (values: ArrayLike<number>) => Array.from(values, (v) => v.toPrecision(10)).join(", ");

/** Prints the RoPE constants and runs both checks; returns true when every check passes. */
export function runSelfCheck(store: WeightStore, cfg: Phi3Config, tokenId: number, print: (line: string) => void): boolean {
  const a = attentionFactor(cfg.maxPositionEmbeddings, cfg.originalMaxPositionEmbeddings);
  const short = invFreq(cfg.ropeTheta, cfg.rotaryDim, cfg.shortFactor);
  const long = invFreq(cfg.ropeTheta, cfg.rotaryDim, cfg.longFactor);
  print(`head_dim ${cfg.headDim}, rotary_dim ${cfg.rotaryDim} (dims ${cfg.rotaryDim}..${cfg.headDim - 1} pass through)`);
  print(`attention_factor A = ${a} (A^2 = ${a * a}; sqrt(17/12) = ${Math.sqrt(17 / 12)})`);
  print(`inv_freq short [0..5]: ${fmt(short.subarray(0, 6))}`);
  print(`inv_freq long  [0..5]: ${fmt(long.subarray(0, 6))}`);
  print(`inv_freq short [${short.length - 1}]: ${short[short.length - 1]!.toPrecision(10)}`);

  const synthetic = Float64Array.from({ length: cfg.headDim }, (_, d) => Math.sin(d + 1) * (d + 1));
  const synthErr = positionZeroError(synthetic, cfg.rotaryDim, short, a);
  print(`synthetic head, position 0: max rel error ${synthErr.toExponential(3)}`);

  const x = normRows(embed(store, cfg, [tokenId]), 1, store.readVector("model.layers.0.input_layernorm.weight"), cfg.rmsNormEps);
  const qkv = streamLinear(store, "model.layers.0.self_attn.qkv_proj.weight", x, 1);
  const layout = qkvLayout(cfg);
  const heads = [
    ...Array.from({ length: cfg.numHeads }, (_, h) => layout.qStart + h * cfg.headDim),
    ...Array.from({ length: cfg.numKvHeads }, (_, g) => layout.kStart + g * cfg.headDim),
  ];
  const realErr = Math.max(...heads.map((at) => positionZeroError(qkv.slice(at, at + cfg.headDim), cfg.rotaryDim, short, a)));
  print(`layer 0, token ${tokenId}, all ${cfg.numHeads} q + ${cfg.numKvHeads} k heads at position 0: max rel error ${realErr.toExponential(3)}`);

  const q0 = qkv.slice(layout.qStart, layout.qStart + cfg.headDim);
  const table = ropeTable([0], short, a);
  const q0r = applyRope(q0, table.cos, table.sin);
  const r = cfg.rotaryDim;
  print(`  q head 0 pre-RoPE  (HF dims 0..3 | ${r / 2}..${r / 2 + 3} | ${r}..${r + 3}): ${fmt(q0.subarray(0, 4))} | ${fmt(q0.subarray(r / 2, r / 2 + 4))} | ${fmt(q0.subarray(r, r + 4))}`);
  print(`  q head 0 post-RoPE (same dims):                ${fmt(q0r.subarray(0, 4))} | ${fmt(q0r.subarray(r / 2, r / 2 + 4))} | ${fmt(q0r.subarray(r, r + 4))}`);

  const ok = synthErr < TOLERANCE && realErr < TOLERANCE;
  print(ok ? "self-check PASSED" : `self-check FAILED (tolerance ${TOLERANCE})`);
  return ok;
}
