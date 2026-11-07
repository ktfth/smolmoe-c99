/**
 * Q/K head permutation: HF rotate_half pairs dims (i, i + rot/2) of the first
 * `rot` coordinates of each head; the runtime rotates adjacent pairs
 * (2i, 2i+1) with the same frequency. Permuting the rows of Q and K alike
 * keeps q.k unchanged; V and O are never permuted.
 *
 *   new n < rot:  old(n) = (n >> 1) + (rot / 2) * (n & 1)
 *   new n >= rot: old(n) = n          (pass-through dims of a partial rotary)
 *
 * With rot = head_dim this is exactly q8pack.c's PERMUTE_HEAD_DIM formula.
 */

/** In-head order: entry n is the source dim of new dim n. */
export function ropePairOrder(headDim: number, rotaryDim: number): Uint32Array {
  if (!Number.isInteger(headDim) || headDim < 2 || headDim % 2 || !Number.isInteger(rotaryDim) || rotaryDim < 2 ||
    rotaryDim % 2 || rotaryDim > headDim)
    throw new Error(`invalid rotary layout: head_dim ${headDim}, rotary_dim ${rotaryDim}`);
  const half = rotaryDim / 2;
  return Uint32Array.from({ length: headDim }, (_, n) => (n < rotaryDim ? (n >> 1) + half * (n & 1) : n));
}

/** Source rows for `heads` permuted heads stored from row `start` (Q or K block of a projection). */
export function qkRows(start: number, heads: number, headDim: number, rotaryDim: number): Uint32Array {
  const order = ropePairOrder(headDim, rotaryDim);
  const rows = new Uint32Array(heads * headDim);
  for (let h = 0; h < heads; h++) for (let n = 0; n < headDim; n++) rows[h * headDim + n] = start + h * headDim + order[n]!;
  return rows;
}

/** Identity rows [start, start + count). */
export function rangeRows(start: number, count: number): Uint32Array {
  return Uint32Array.from({ length: count }, (_, i) => start + i);
}

/** Applies the head permutation to a vector (q/k biases). */
export function permuteHeads(x: Float32Array, heads: number, headDim: number, rotaryDim: number): Float32Array {
  if (x.length !== heads * headDim) throw new Error(`vector of ${x.length} values is not ${heads} heads of ${headDim}`);
  const rows = qkRows(0, heads, headDim, rotaryDim);
  return Float32Array.from(rows, (r) => x[r]!);
}
