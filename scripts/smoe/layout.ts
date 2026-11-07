/**
 * Container planning: record validation, the two accepted payload layouts and
 * the header + directory encoding (docs/FORMAT.md).
 *
 *   compact    payloads contiguous right after the directory (tools/*.py, fixture.ts)
 *   aligned64  every payload on a 64-byte boundary (src/container.c: quantize, bundle)
 */
import {
  DTYPE, type Dtype, HEADER_BYTES, MAGIC, MAX_COLS, MAX_TENSORS, RECORD_BYTES, type Version, alignUp, assertName,
  compareNames, payloadBytes,
} from "./format";

export type Layout = "compact" | "aligned64";

export interface RecordSpec {
  readonly name: string;
  readonly rows: number;
  readonly cols: number;
  readonly dtype: Dtype;
  /** 1: lm_head shares embed's payload (tied head, v2+) */
  readonly alias: 0 | 1;
  readonly bytes: number;
}

export interface PlannedRecord extends RecordSpec {
  readonly offset: number;
}

export function tensorRecord(name: string, rows: number, cols: number, dtype: Dtype, alias: 0 | 1 = 0): RecordSpec {
  if (dtype === DTYPE.BLOB) throw new Error(`${name}: use blobRecord for blobs`);
  return { name, rows, cols, dtype, alias, bytes: payloadBytes(dtype, rows, cols) };
}

export function blobRecord(name: string, bytes: number): RecordSpec {
  return { name, rows: 1, cols: 1, dtype: DTYPE.BLOB, alias: 0, bytes };
}

/** v3 when blobs are present, else v2 when an alias is, else v1 (the checkpoint.py rule). */
export function chooseVersion(records: readonly RecordSpec[]): Version {
  if (records.some((r) => r.dtype === DTYPE.BLOB)) return 3;
  return records.some((r) => r.alias) ? 2 : 1;
}

function checkRecord(r: RecordSpec): void {
  assertName(r.name);
  if (!Number.isSafeInteger(r.bytes) || r.bytes < 0) throw new Error(`${r.name}: invalid byte count`);
  if (r.dtype === DTYPE.BLOB) {
    if (!r.name.startsWith("__") || r.rows !== 1 || r.cols !== 1 || r.alias)
      throw new Error(`invalid blob record: ${r.name} (blobs are 1x1, unaliased and named __*)`);
    return;
  }
  if (r.name.startsWith("__")) throw new Error(`tensor names must not start with __: ${r.name}`);
  if (!Number.isInteger(r.rows) || r.rows < 1 || r.rows > 0xffffffff || !Number.isInteger(r.cols) || r.cols < 1 ||
    r.cols > MAX_COLS || (r.dtype !== DTYPE.F32 && r.dtype !== DTYPE.Q8 && r.dtype !== DTYPE.Q5))
    throw new Error(`invalid tensor metadata: ${r.name}`);
  if (r.bytes !== payloadBytes(r.dtype, r.rows, r.cols)) throw new Error(`${r.name}: byte count does not match shape`);
}

function checkAlias(r: RecordSpec, embed: RecordSpec | undefined): void {
  if (r.name !== "lm_head" || !embed || embed.rows !== r.rows || embed.cols !== r.cols || embed.dtype !== r.dtype ||
    embed.bytes !== r.bytes)
    throw new Error(`invalid tied embedding alias: ${r.name}`);
}

/** Validates the record list (sorted, unique, legal alias) and assigns payload offsets. */
export function planContainer(records: readonly RecordSpec[], layout: Layout): readonly PlannedRecord[] {
  if (records.length < 1 || records.length > MAX_TENSORS) throw new Error("invalid record count");
  const align = layout === "aligned64" ? (x: number) => alignUp(x) : (x: number) => x;
  let offset = align(HEADER_BYTES + records.length * RECORD_BYTES);
  let embed: PlannedRecord | undefined;
  const plan: PlannedRecord[] = [];
  records.forEach((r, i) => {
    checkRecord(r);
    if (i && compareNames(records[i - 1]!.name, r.name) >= 0)
      throw new Error(`records must have unique sorted names (${records[i - 1]!.name}, ${r.name})`);
    if (r.alias) {
      checkAlias(r, embed);
      plan.push({ ...r, offset: embed!.offset });
      return;
    }
    const placed = { ...r, offset };
    if (r.name === "embed") embed = placed;
    plan.push(placed);
    offset = align(offset + r.bytes);
  });
  return plan;
}

/** End of the last payload (no trailing padding). */
export function containerBytes(plan: readonly PlannedRecord[]): number {
  return plan.reduce((end, r) => Math.max(end, r.offset + r.bytes), HEADER_BYTES + plan.length * RECORD_BYTES);
}

/** Header (16 bytes) and directory (72 bytes per record), little-endian. */
export function encodeDirectory(version: Version, plan: readonly PlannedRecord[]): Uint8Array {
  if (version < 2 && plan.some((r) => r.alias)) throw new Error("aliases need container version 2 or later");
  if (version < 3 && plan.some((r) => r.dtype === DTYPE.BLOB)) throw new Error("blobs need container version 3");
  const out = new Uint8Array(HEADER_BYTES + plan.length * RECORD_BYTES);
  const view = new DataView(out.buffer);
  out.set(MAGIC, 0);
  out[7] = version;
  view.setUint32(8, plan.length, true);
  plan.forEach((r, i) => {
    const base = HEADER_BYTES + i * RECORD_BYTES;
    for (let c = 0; c < r.name.length; c++) out[base + c] = r.name.charCodeAt(c);
    view.setUint32(base + 40, r.rows, true);
    view.setUint32(base + 44, r.cols, true);
    view.setUint32(base + 48, r.dtype, true);
    view.setUint32(base + 52, r.alias, true);
    view.setBigUint64(base + 56, BigInt(r.offset), true);
    view.setBigUint64(base + 64, BigInt(r.bytes), true);
  });
  return out;
}
