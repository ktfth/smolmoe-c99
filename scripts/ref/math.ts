/**
 * Scalar math for the independent Phi-3 (Phi-4-mini) reference, in float64.
 * Each function follows the Hugging Face semantics cited next to it
 * (transformers v4.49 models/phi3/modeling_phi3.py = the checkpoint's remote
 * code, and modeling_rope_utils.py `_compute_longrope_parameters`).
 */

/**
 * Phi3RMSNorm.forward: `weight * (x * rsqrt(mean(x^2) + eps))`.
 * No mean subtraction, eps inside the square root, gain is `weight` itself.
 */
export function rmsNorm(x: Float64Array, weight: ArrayLike<number>, eps: number): Float64Array {
  if (x.length !== weight.length) throw new RangeError(`rmsNorm: ${x.length} values, ${weight.length} weights`);
  let sumSq = 0;
  for (let i = 0; i < x.length; i++) sumSq += x[i]! * x[i]!;
  const inv = 1 / Math.sqrt(sumSq / x.length + eps);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = weight[i]! * (x[i]! * inv);
  return out;
}

/** softmax(dim=-1) with the usual max subtraction (exact in real arithmetic). */
export function softmax(scores: Float64Array): Float64Array {
  if (scores.length === 0) throw new RangeError("softmax of an empty vector");
  let max = -Infinity;
  for (const s of scores) if (s > max) max = s;
  const out = new Float64Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) {
    const e = Math.exp(scores[i]! - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] = out[i]! / sum;
  return out;
}

/** nn.functional.silu: x * sigmoid(x) = x / (1 + e^-x). */
export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

/**
 * repeat_kv: KV head g is expanded to query heads g*n_rep .. g*n_rep + n_rep - 1,
 * so query head h reads KV head floor(h / n_rep).
 */
export function kvHeadFor(qHead: number, numHeads: number, numKvHeads: number): number {
  if (numKvHeads <= 0 || numHeads % numKvHeads !== 0) {
    throw new RangeError(`GQA: ${numHeads} query heads not divisible by ${numKvHeads} KV heads`);
  }
  if (!Number.isInteger(qHead) || qHead < 0 || qHead >= numHeads) throw new RangeError(`GQA: bad head ${qHead}`);
  return Math.floor(qHead / (numHeads / numKvHeads));
}

/**
 * `_compute_longrope_parameters`: factor = max_position_embeddings /
 * original_max_position_embeddings; attention_factor = sqrt(1 + ln(factor) /
 * ln(original_max)) when factor > 1, else 1. (No explicit factor or
 * attention_factor key exists in this checkpoint's rope_scaling.)
 */
export function attentionFactor(maxPositionEmbeddings: number, originalMaxPositionEmbeddings: number): number {
  const factor = maxPositionEmbeddings / originalMaxPositionEmbeddings;
  if (factor <= 1) return 1;
  return Math.sqrt(1 + Math.log(factor) / Math.log(originalMaxPositionEmbeddings));
}

/**
 * inv_freq[i] = 1 / (ext_factors[i] * base^(2i / dim)), i = 0 .. dim/2 - 1, where
 * dim = int(head_dim * partial_rotary_factor) is the rotary dimension.
 */
export function invFreq(base: number, rotaryDim: number, factors: readonly number[]): Float64Array {
  if (rotaryDim <= 0 || rotaryDim % 2 !== 0) throw new RangeError(`rotary dim must be even: ${rotaryDim}`);
  const half = rotaryDim / 2;
  if (factors.length !== half) throw new RangeError(`need ${half} rope factors, got ${factors.length}`);
  const out = new Float64Array(half);
  for (let i = 0; i < half; i++) out[i] = 1 / (factors[i]! * base ** ((2 * i) / rotaryDim));
  return out;
}

/**
 * Longrope frequency update: seq_len = max(position_ids) + 1; long factors when
 * seq_len > original_max_position_embeddings, short factors otherwise. The
 * choice applies to every token of the forward call.
 */
export function selectRopeFactors(
  maxPositionId: number,
  originalMaxPositionEmbeddings: number,
  shortFactor: readonly number[],
  longFactor: readonly number[],
): readonly number[] {
  return maxPositionId + 1 > originalMaxPositionEmbeddings ? longFactor : shortFactor;
}

/** cos/sin per position, flattened as [position][frequency], with the attention factor folded in. */
export interface RopeTable {
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly half: number;
}

/**
 * Phi3RotaryEmbedding.forward: freqs = pos * inv_freq; emb = cat(freqs, freqs);
 * cos = emb.cos() * attention_scaling; sin = emb.sin() * attention_scaling.
 * Only the `half` unique entries per position are stored (emb repeats them).
 */
export function ropeTable(positions: readonly number[], inv: Float64Array, scale: number): RopeTable {
  const half = inv.length;
  const cos = new Float64Array(positions.length * half);
  const sin = new Float64Array(positions.length * half);
  positions.forEach((p, t) => {
    for (let i = 0; i < half; i++) {
      const angle = p * inv[i]!;
      cos[t * half + i] = scale * Math.cos(angle);
      sin[t * half + i] = scale * Math.sin(angle);
    }
  });
  return { cos, sin, half };
}

/**
 * apply_rotary_pos_emb for one head: rotary_dim = 2 * cos.length; the first
 * rotary_dim dims get `x*cos + rotate_half(x)*sin` with the split-half
 * rotate_half (pairs (j, j + half)); the remaining dims pass through unchanged.
 *   y[j]        = x[j]        * c_j - x[j + half] * s_j
 *   y[j + half] = x[j + half] * c_j + x[j]        * s_j
 */
export function applyRope(head: Float64Array, cos: ArrayLike<number>, sin: ArrayLike<number>): Float64Array {
  const half = cos.length;
  if (sin.length !== half || 2 * half > head.length) {
    throw new RangeError(`rope: rotary span ${2 * half} does not fit a head of ${head.length}`);
  }
  const out = Float64Array.from(head);
  for (let j = 0; j < half; j++) {
    const a = head[j]!;
    const b = head[j + half]!;
    out[j] = a * cos[j]! - b * sin[j]!;
    out[j + half] = b * cos[j]! + a * sin[j]!;
  }
  return out;
}
