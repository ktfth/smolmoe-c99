/**
 * Attention and MLP blocks of the independent Phi-3 reference, over all
 * positions of one causal forward call (no KV cache: every layer sees the whole
 * sequence at once). Float64 throughout.
 *
 * Sources (transformers v4.49 models/phi3/modeling_phi3.py, identical to the
 * checkpoint's remote code):
 *  - Phi3Attention.forward: qkv = qkv_proj(h); q = qkv[:, :H*D], k = next KV*D,
 *    v = last KV*D; each viewed as (heads, head_dim), head-major.
 *  - apply_rotary_pos_emb before attention; eager_attention_forward:
 *    softmax(q k^T * head_dim^-0.5 + causal mask) v, KV heads via repeat_kv.
 *  - attn_output.reshape(..., H*D) (head-major concat) feeds o_proj.
 *  - Phi3MLP.forward: gate, up = gate_up_proj(h).chunk(2); down_proj(up * silu(gate)).
 */
import { applyRope, kvHeadFor, silu, softmax, type RopeTable } from "./math";

export interface AttentionShape {
  readonly numHeads: number;
  readonly numKvHeads: number;
  readonly headDim: number;
}

export interface QkvLayout {
  readonly qStart: number;
  readonly kStart: number;
  readonly vStart: number;
  readonly width: number;
}

/** Column ranges of one fused qkv_proj output row. */
export function qkvLayout(shape: AttentionShape): QkvLayout {
  const qDim = shape.numHeads * shape.headDim;
  const kvDim = shape.numKvHeads * shape.headDim;
  return { qStart: 0, kStart: qDim, vStart: qDim + kvDim, width: qDim + 2 * kvDim };
}

/** Rotates every head of a block of `count` heads starting at `start` in a qkv row. */
function rotateHeads(
  qkv: Float64Array,
  rowStart: number,
  start: number,
  count: number,
  headDim: number,
  cos: Float64Array,
  sin: Float64Array,
): Float64Array[] {
  return Array.from({ length: count }, (_, h) => {
    const at = rowStart + start + h * headDim;
    return applyRope(qkv.subarray(at, at + headDim), cos, sin);
  });
}

function dot(a: Float64Array, b: Float64Array): number {
  let acc = 0;
  for (let i = 0; i < a.length; i++) acc += a[i]! * b[i]!;
  return acc;
}

/**
 * Causal self-attention for positions 0..T-1 (row t of `qkv` is position
 * `rope` row t). Returns the T x (H*D) head-major concat that feeds o_proj.
 * Query i attends to keys j <= i; the sliding window (262144) never applies
 * at the lengths this reference accepts (checked by the caller).
 */
export function causalAttention(qkv: Float64Array, T: number, shape: AttentionShape, rope: RopeTable): Float64Array {
  const { numHeads, numKvHeads, headDim } = shape;
  const layout = qkvLayout(shape);
  if (qkv.length !== T * layout.width) throw new RangeError(`attention: qkv has ${qkv.length}, want ${T * layout.width}`);
  if (rope.cos.length < T * rope.half) throw new RangeError(`attention: rope table shorter than ${T} positions`);

  const queries: Float64Array[][] = [];
  const keys: Float64Array[][] = [];
  const values: Float64Array[][] = [];
  for (let t = 0; t < T; t++) {
    const cos = rope.cos.subarray(t * rope.half, (t + 1) * rope.half);
    const sin = rope.sin.subarray(t * rope.half, (t + 1) * rope.half);
    const row = t * layout.width;
    queries.push(rotateHeads(qkv, row, layout.qStart, numHeads, headDim, cos, sin));
    keys.push(rotateHeads(qkv, row, layout.kStart, numKvHeads, headDim, cos, sin));
    values.push(
      Array.from({ length: numKvHeads }, (_, g) => qkv.slice(row + layout.vStart + g * headDim, row + layout.vStart + (g + 1) * headDim)),
    );
  }

  const scaling = headDim ** -0.5;
  const qDim = numHeads * headDim;
  const out = new Float64Array(T * qDim);
  for (let t = 0; t < T; t++) {
    for (let h = 0; h < numHeads; h++) {
      const g = kvHeadFor(h, numHeads, numKvHeads);
      const q = queries[t]![h]!;
      const scores = new Float64Array(t + 1);
      for (let j = 0; j <= t; j++) scores[j] = dot(q, keys[j]![g]!) * scaling;
      const weights = softmax(scores);
      const base = t * qDim + h * headDim;
      for (let j = 0; j <= t; j++) {
        const v = values[j]![g]!;
        const a = weights[j]!;
        for (let d = 0; d < headDim; d++) out[base + d] = out[base + d]! + a * v[d]!;
      }
    }
  }
  return out;
}

/** up * silu(gate) per position, where each gate_up row is [gate (I) | up (I)]. */
export function swiGlu(gateUp: Float64Array, T: number, intermediate: number): Float64Array {
  if (gateUp.length !== T * 2 * intermediate) {
    throw new RangeError(`swiGlu: ${gateUp.length} values, want ${T * 2 * intermediate}`);
  }
  const out = new Float64Array(T * intermediate);
  for (let t = 0; t < T; t++) {
    const row = t * 2 * intermediate;
    for (let i = 0; i < intermediate; i++) {
      out[t * intermediate + i] = gateUp[row + intermediate + i]! * silu(gateUp[row + i]!);
    }
  }
  return out;
}
