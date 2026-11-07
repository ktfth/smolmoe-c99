/**
 * Generic HF -> SMOEQ8 pipeline. A model module maps every runtime tensor to
 * a source (rows of a source tensor, a vector, zeros, or the tied alias);
 * this module validates the mapping against the config's tensor set, streams
 * the payloads through the worker pool in container order, re-checks the
 * written file like the C loader, and only then renames it into place.
 *
 *   v2: compact layout, version by rule (v2 when tied): byte-identical to the Python importers
 *   v3: aligned64 layout with __config__/__tokenizer__ blobs: identical to smolmoe-bundle
 */
import { renameSync, rmSync } from "node:fs";

import { type ModelConfig, formatConfig, tensorSpecs } from "../smoe/config";
import { checkFiles } from "../smoe/check";
import { BLOB_CONFIG, BLOB_TOKENIZER, DTYPE, compareNames } from "../smoe/format";
import { type PlannedRecord, type RecordSpec, blobRecord, chooseVersion, planContainer, tensorRecord } from "../smoe/layout";
import { encodeF32 } from "../smoe/quant";
import { type Chunk, writeContainer } from "../smoe/writer";
import { decodeVector } from "./decode";
import { permuteHeads } from "./permute";
import { createPool, type RowPool } from "./pool";
import { type Checkpoint, type HfTensor, readTensorBytes } from "./safetensors";

export type TensorSource =
  /** Q8 matrix whose row i is source row rows[i] of `tensor` (row maps, slices, head permutations) */
  | { readonly kind: "rows"; readonly tensor: HfTensor; readonly rows: Uint32Array }
  /** F32 vector, optionally head-permuted (q/k biases) */
  | { readonly kind: "vector"; readonly tensor: HfTensor; readonly heads?: { readonly count: number;
    readonly headDim: number; readonly rotaryDim: number } }
  /** all-zero Q8 matrix (the router of a dense model: one expert with weight exactly 1) */
  | { readonly kind: "zeros" }
  /** lm_head sharing embed's payload */
  | { readonly kind: "alias" };

export interface ModelPlan {
  readonly config: ModelConfig;
  readonly sources: ReadonlyMap<string, TensorSource>;
}

export interface WriteOptions {
  readonly output: string;
  readonly container: "v2" | "v3";
  readonly threads: number;
  readonly tokenizer: Uint8Array;
  readonly log?: (line: string) => void;
  /**
   * Awaited after the re-check, before model.smoe is renamed into place: lets source hashing overlap the
   * conversion and still veto the output, and writes the side files first so the model lands last.
   */
  readonly beforeRename?: () => Promise<void>;
}

const JOB_WEIGHTS = 1 << 22; /* ~4M weights per job: 8 MiB of BF16 in, 4.5 MiB of Q8 out */

const sourcesOf = (s: TensorSource): readonly HfTensor[] => ("tensor" in s ? [s.tensor] : []);

/** Shape and kind checks of every mapping; every source tensor must be consumed. */
export function checkPlan(plan: ModelPlan, checkpoint: Checkpoint): void {
  const used = new Set<string>();
  const specs = tensorSpecs(plan.config);
  const names = new Set(specs.map((t) => t.name));
  for (const spec of specs) {
    const s = plan.sources.get(spec.name);
    if (!s) throw new Error(`no source for tensor ${spec.name}`);
    const bad = (what: string) => new Error(`${spec.name}: ${what}`);
    if (s.kind === "alias" && spec.name !== "lm_head") throw bad("only lm_head may alias embed");
    if (s.kind === "vector") {
      if (spec.kind !== "vector") throw bad("a matrix cannot come from a vector");
      if (s.tensor.shape.length !== 1 || s.tensor.shape[0] !== spec.cols)
        throw bad(`source ${s.tensor.name} has shape [${s.tensor.shape.join(", ")}], expected [${spec.cols}]`);
    } else if (spec.kind === "vector") throw bad("a vector must come from a 1-D source tensor");
    if (s.kind === "rows") {
      const [rows, cols] = s.tensor.shape;
      if (s.tensor.shape.length !== 2 || cols !== spec.cols)
        throw bad(`source ${s.tensor.name} has shape [${s.tensor.shape.join(", ")}], expected [*, ${spec.cols}]`);
      if (s.rows.length !== spec.rows) throw bad(`row map has ${s.rows.length} rows, expected ${spec.rows}`);
      for (const r of s.rows) if (r >= rows!) throw bad(`row map points past ${s.tensor.name}`);
    }
    for (const t of sourcesOf(s)) {
      if (checkpoint.tensors.get(t.name) !== t) throw bad(`source ${t.name} is not from this checkpoint`);
      used.add(t.name);
    }
  }
  const extra = [...plan.sources.keys()].filter((n) => !names.has(n));
  if (extra.length) throw new Error(`sources for unknown tensors: ${extra.join(", ")}`);
  const unused = [...checkpoint.tensors.keys()].filter((n) => !used.has(n));
  if (unused.length) throw new Error(`unused source tensors: ${unused.slice(0, 8).join(", ")}${unused.length > 8 ? ", ..." : ""}`);
}

function records(plan: ModelPlan, blobs: ReadonlyMap<string, Uint8Array>): RecordSpec[] {
  const list: RecordSpec[] = tensorSpecs(plan.config).map((spec) => {
    if (spec.kind === "vector") return tensorRecord(spec.name, spec.rows, spec.cols, DTYPE.F32);
    return tensorRecord(spec.name, spec.rows, spec.cols, DTYPE.Q8, plan.sources.get(spec.name)!.kind === "alias" ? 1 : 0);
  });
  for (const [name, bytes] of blobs) list.push(blobRecord(name, bytes.length));
  return list.sort((a, b) => compareNames(a.name, b.name));
}

function vectorBytes(s: Extract<TensorSource, { kind: "vector" }>): Uint8Array {
  const x = decodeVector(s.tensor.dtype, readTensorBytes(s.tensor));
  return encodeF32(s.heads ? permuteHeads(x, s.heads.count, s.heads.headDim, s.heads.rotaryDim) : x);
}

function* chunks(record: PlannedRecord, source: TensorSource | undefined, blob: Uint8Array | undefined,
  pool: RowPool): Generator<Chunk, void> {
  if (blob) {
    yield blob;
    return;
  }
  if (!source) throw new Error(`internal error: no source for ${record.name}`);
  if (source.kind === "zeros" || source.kind === "vector") {
    yield source.kind === "zeros" ? new Uint8Array(record.bytes) : vectorBytes(source);
    return;
  }
  if (source.kind !== "rows") throw new Error(`internal error: ${record.name} has no payload`);
  const t = source.tensor;
  const step = Math.max(1, Math.floor(JOB_WEIGHTS / record.cols));
  for (let r = 0; r < record.rows; r += step) {
    const rows = source.rows.subarray(r, Math.min(record.rows, r + step));
    yield () => pool.run({ file: t.file, offset: t.offset, dtype: t.dtype, cols: record.cols, sourceRows: t.shape[0]!, rows });
  }
}

/** Writes the model (temp file, self-check, rename). Returns the size in bytes. */
export async function writeModel(plan: ModelPlan, o: WriteOptions): Promise<number> {
  const configText = new TextEncoder().encode(formatConfig(plan.config));
  const blobs = new Map<string, Uint8Array>(o.container === "v3"
    ? [[BLOB_CONFIG, configText], [BLOB_TOKENIZER, o.tokenizer]] : []);
  const list = records(plan, blobs);
  const layout = planContainer(list, o.container === "v3" ? "aligned64" : "compact");
  const partial = `${o.output}.partial`;
  const pool = createPool(o.threads);
  try {
    const started = performance.now();
    const bytes = await writeContainer(partial, chooseVersion(list), layout,
      (r) => chunks(r, plan.sources.get(r.name), blobs.get(r.name), pool), Math.max(4, o.threads * 4));
    o.log?.(`wrote ${bytes} bytes in ${((performance.now() - started) / 1000).toFixed(1)} s; re-checking`);
    /* v3 is checked through its embedded blobs, exactly as the runtime would load it. */
    checkFiles(o.container === "v3" ? { model: partial } : { model: partial, config: configText, tokenizer: o.tokenizer });
    await o.beforeRename?.();
    renameSync(partial, o.output);
    return bytes;
  } catch (e) {
    try {
      rmSync(partial, { force: true });
    } catch (cleanup) {
      o.log?.(`could not remove ${partial}: ${cleanup instanceof Error ? cleanup.message : String(cleanup)}`);
    }
    throw e;
  } finally {
    await pool.close();
  }
}
