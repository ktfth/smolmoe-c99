/**
 * nn.Linear without bias, batched over positions: out[t][r] = sum_c W[r][c] x[t][c]
 * with W in PyTorch layout [out_features, in_features]. Weights are streamed
 * from disk in row chunks, decoded once, and applied to every position while the
 * row is hot, so each weight is read exactly once per forward call. Sums are
 * plain sequential float64 sums over c.
 */
import { decodeBf16Into } from "./bf16";
import type { WeightStore } from "./safetensors";

const CHUNK_BYTES = 8 * 1024 * 1024;

/**
 * Writes rows [0, rows) of `w` (row-major, `cols` wide) applied to the T
 * positions of `x` into out[t * outStride + outCol + r]. Positions are processed
 * four at a time so each weight is loaded once per group; every accumulator is
 * still an in-order sum over c.
 */
export function matmulRowsInto(
  w: Float32Array,
  rows: number,
  cols: number,
  x: Float64Array,
  T: number,
  out: Float64Array,
  outStride: number,
  outCol: number,
): void {
  if (w.length < rows * cols || x.length !== T * cols) throw new RangeError("matmul: size mismatch");
  for (let r = 0; r < rows; r++) {
    const wb = r * cols;
    let t = 0;
    for (; t + 4 <= T; t += 4) {
      const o0 = t * cols;
      const o1 = o0 + cols;
      const o2 = o1 + cols;
      const o3 = o2 + cols;
      let a0 = 0;
      let a1 = 0;
      let a2 = 0;
      let a3 = 0;
      for (let c = 0; c < cols; c++) {
        const wc = w[wb + c]!;
        a0 += wc * x[o0 + c]!;
        a1 += wc * x[o1 + c]!;
        a2 += wc * x[o2 + c]!;
        a3 += wc * x[o3 + c]!;
      }
      out[t * outStride + outCol + r] = a0;
      out[(t + 1) * outStride + outCol + r] = a1;
      out[(t + 2) * outStride + outCol + r] = a2;
      out[(t + 3) * outStride + outCol + r] = a3;
    }
    for (; t < T; t++) {
      const o = t * cols;
      let a = 0;
      for (let c = 0; c < cols; c++) a += w[wb + c]! * x[o + c]!;
      out[t * outStride + outCol + r] = a;
    }
  }
}

/** Pure form for small in-memory weights: returns T x rows. */
export function matmulRows(w: Float32Array, rows: number, cols: number, x: Float64Array, T: number): Float64Array {
  if (w.length !== rows * cols) throw new RangeError(`matmul: weight has ${w.length}, want ${rows * cols}`);
  const out = new Float64Array(T * rows);
  matmulRowsInto(w, rows, cols, x, T, out, rows, 0);
  return out;
}

/** Applies the 2-D BF16 weight `name` to T positions of `x`, streaming it from disk. Returns T x rows. */
export function streamLinear(store: WeightStore, name: string, x: Float64Array, T: number): Float64Array {
  const shape = store.shape(name);
  if (shape.length !== 2) throw new Error(`${name}: shape [${shape}] is not a matrix`);
  const [rows, cols] = shape as [number, number];
  if (x.length !== T * cols) throw new RangeError(`${name}: input has ${x.length}, want ${T} x ${cols}`);
  const chunkRows = Math.max(1, Math.floor(CHUNK_BYTES / (cols * 2)));
  const scratch = new Float32Array(chunkRows * cols);
  const out = new Float64Array(T * rows);
  for (let r0 = 0; r0 < rows; r0 += chunkRows) {
    const n = Math.min(chunkRows, rows - r0);
    decodeBf16Into(store.readRowsBf16(name, r0, n), scratch);
    matmulRowsInto(scratch, n, cols, x, T, out, rows, r0);
  }
  return out;
}

export interface Scored {
  readonly id: number;
  readonly logit: number;
}

/** The k largest of values[offset .. offset+length), ids relative to offset; ties go to the lower id. */
export function topK(values: Float64Array, offset: number, length: number, k: number): Scored[] {
  const best: Scored[] = [];
  for (let i = 0; i < length; i++) {
    const v = values[offset + i]!;
    if (best.length === k && v <= best[k - 1]!.logit) continue;
    let at = best.length;
    while (at > 0 && best[at - 1]!.logit < v) at--;
    best.splice(at, 0, { id: i, logit: v });
    if (best.length > k) best.pop();
  }
  return best;
}
