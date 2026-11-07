/**
 * TS re-check of a written container, mirroring the C loader:
 *   parseContainer  modelfile_parse (src/modelfile.c), same rules and messages
 *   verifyPayload   modelfile_verify_payload (Q8 scale finite and >= 0, Q5 f16
 *                   scale finite, F32 finite; aliases and blobs skipped)
 *   checkModel      model_init's tensor count and exact shape/dtype lookups
 */
import { closeSync, fstatSync, openSync, readSync } from "node:fs";

import { type ModelConfig, expectedTensorCount, tensorSpecs } from "./config";
import {
  DTYPE, type Dtype, HEADER_BYTES, MAGIC, MAX_COLS, MAX_TENSORS, NAME_BYTES, Q5_BLOCK_BYTES, Q8_BLOCK_BYTES, QK,
  RECORD_BYTES, type Version,
} from "./format";
import { f16ToF32 } from "./quant";

export interface ParsedRecord {
  readonly name: string;
  readonly rows: number;
  readonly cols: number;
  readonly dtype: Dtype | number;
  readonly alias: number;
  readonly offset: number;
  readonly bytes: number;
}

export interface ParsedContainer {
  readonly version: Version;
  readonly count: number;
  readonly blobs: number;
  readonly size: number;
  readonly records: readonly ParsedRecord[];
}

const fail = (text: string): never => { throw new Error(text); };

function expectedBytes(r: ParsedRecord): number {
  if (r.dtype === DTYPE.BLOB) return r.bytes;
  const blocks = Math.ceil(r.cols / QK);
  if (r.dtype === DTYPE.Q8) return r.rows * blocks * Q8_BLOCK_BYTES;
  if (r.dtype === DTYPE.Q5) return r.rows * blocks * Q5_BLOCK_BYTES;
  return r.rows * r.cols * 4;
}

function decodeName(bytes: Uint8Array): string {
  const nul = bytes.indexOf(0);
  if (nul <= 0) fail("invalid tensor name");
  return String.fromCharCode(...bytes.subarray(0, nul)); /* byte-for-byte: keeps strcmp order */
}

function checkMetadata(version: number, r: ParsedRecord): void {
  if (r.dtype === DTYPE.BLOB) {
    if (version < 3 || r.rows !== 1 || r.cols !== 1 || r.alias || !r.name.startsWith("__"))
      fail(`invalid blob record: ${r.name}`);
    return;
  }
  if (r.alias > 1 || (r.alias && version < 2) || !r.rows || !r.cols || r.cols > MAX_COLS || r.dtype > DTYPE.Q5 ||
    r.name.startsWith("__"))
    fail(`invalid tensor metadata: ${r.name}`);
}

function checkAlias(records: readonly ParsedRecord[], r: ParsedRecord): void {
  const e = records.find((x) => x.name === "embed");
  if (!e || r.name !== "lm_head" || r.offset !== e.offset || r.bytes !== e.bytes || r.rows !== e.rows ||
    r.cols !== e.cols || r.dtype !== e.dtype)
    fail("invalid tied embedding alias");
}

/** Port of modelfile_parse. `bytes` holds at least header and directory; `size` is the container size. */
export function parseContainer(bytes: Uint8Array, size: number): ParsedContainer {
  if (size < HEADER_BYTES || bytes.length < HEADER_BYTES) fail("invalid checkpoint size");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = bytes[7]!;
  if (MAGIC.some((b, i) => bytes[i] !== b) || version < 1 || version > 3 || view.getUint32(12, true))
    fail("unsupported checkpoint format (expected SMOEQ8 v1-v3)");
  const count = view.getUint32(8, true);
  if (!count || count > MAX_TENSORS) fail("invalid tensor count");
  let previous = HEADER_BYTES + count * RECORD_BYTES;
  if (previous > size) fail("truncated checkpoint directory");
  if (previous > bytes.length) fail("internal error: directory bytes not supplied");
  const records: ParsedRecord[] = [];
  let blobs = 0;
  for (let i = 0; i < count; i++) {
    const base = HEADER_BYTES + i * RECORD_BYTES;
    const r: ParsedRecord = {
      name: decodeName(bytes.subarray(base, base + NAME_BYTES)),
      rows: view.getUint32(base + 40, true), cols: view.getUint32(base + 44, true),
      dtype: view.getUint32(base + 48, true), alias: view.getUint32(base + 52, true),
      offset: Number(view.getBigUint64(base + 56, true)), bytes: Number(view.getBigUint64(base + 64, true)),
    };
    checkMetadata(version, r);
    if (r.bytes !== expectedBytes(r) || (!r.alias && r.offset < previous) || r.offset > size || r.bytes > size - r.offset)
      fail(`invalid tensor extent: ${r.name}`);
    if (r.alias) checkAlias(records, r);
    else previous = r.offset + r.bytes;
    if (i && records[i - 1]!.name >= r.name) fail("tensor directory must have unique sorted names");
    if (r.dtype === DTYPE.BLOB) ++blobs;
    records.push(r);
  }
  return { version: version as Version, count, blobs, size, records };
}

function readExact(fd: number, buf: Uint8Array, length: number, position: number): void {
  for (let done = 0; done < length;) {
    const n = readSync(fd, buf, done, length - done, position + done);
    if (n <= 0) fail("unexpected end of file");
    done += n;
  }
}

/** Reads and parses the directory of the container at `base` (0 for a plain .smoe). */
export function readContainer(path: string, base = 0, length?: number): ParsedContainer {
  const fd = openSync(path, "r");
  try {
    const size = length ?? fstatSync(fd).size - base;
    const header = new Uint8Array(HEADER_BYTES);
    readExact(fd, header, Math.min(HEADER_BYTES, size), base);
    const count = new DataView(header.buffer).getUint32(8, true);
    const dirEnd = HEADER_BYTES + Math.min(count, MAX_TENSORS) * RECORD_BYTES;
    if (dirEnd > size) return parseContainer(header, size); /* reports the truncation */
    const all = new Uint8Array(dirEnd);
    readExact(fd, all, dirEnd, base);
    return parseContainer(all, size);
  } finally {
    closeSync(fd);
  }
}

const VERIFY_CHUNK = 36 * 22 * 64 * 256; /* a multiple of every unit size, ~13 MB */

function badUnit(dtype: number, view: DataView, units: number): boolean {
  if (dtype === DTYPE.Q8) {
    for (let u = 0; u < units; u++) {
      const s = view.getFloat32(u * Q8_BLOCK_BYTES, true);
      if (!Number.isFinite(s) || s < 0) return true;
    }
  } else if (dtype === DTYPE.Q5) {
    for (let u = 0; u < units; u++) if (!Number.isFinite(f16ToF32(view.getUint16(u * Q5_BLOCK_BYTES, true)))) return true;
  } else {
    for (let u = 0; u < units; u++) if (!Number.isFinite(view.getFloat32(u * 4, true))) return true;
  }
  return false;
}

/** Port of modelfile_verify_payload, streamed in bounded chunks. */
export function verifyPayload(path: string, parsed: ParsedContainer, base = 0): void {
  const fd = openSync(path, "r");
  const buf = new Uint8Array(VERIFY_CHUNK);
  const view = new DataView(buf.buffer);
  try {
    for (const r of parsed.records) {
      if (r.alias || r.dtype === DTYPE.BLOB) continue;
      const unit = r.dtype === DTYPE.Q8 ? Q8_BLOCK_BYTES : r.dtype === DTYPE.Q5 ? Q5_BLOCK_BYTES : 4;
      for (let done = 0; done < r.bytes;) {
        const n = Math.min(VERIFY_CHUNK, r.bytes - done);
        readExact(fd, buf, n, base + r.offset + done);
        if (badUnit(r.dtype, view, n / unit)) fail(`non-finite value or invalid scale in tensor ${r.name}`);
        done += n;
      }
    }
  } finally {
    closeSync(fd);
  }
}

/** model_init: tensor count (blobs excluded) and every tensor with exact shape and dtype class. */
export function checkModel(c: ModelConfig, parsed: ParsedContainer): void {
  if (expectedTensorCount(c) !== parsed.count - parsed.blobs) fail("tensor count does not match config");
  const byName = new Map(parsed.records.map((r) => [r.name, r]));
  for (const spec of tensorSpecs(c)) {
    const t = byName.get(spec.name) ?? fail(`missing tensor: ${spec.name}`);
    const typeOk = spec.kind === "vector" ? t.dtype === DTYPE.F32 : t.dtype === DTYPE.Q8 || t.dtype === DTYPE.Q5;
    if (t.rows !== spec.rows || t.cols !== spec.cols || !typeOk) fail(`wrong tensor shape/type: ${spec.name}`);
  }
}

/** The payload of an embedded blob, or null. */
export function readBlob(path: string, parsed: ParsedContainer, name: string, base = 0): Uint8Array | null {
  const r = parsed.records.find((x) => x.dtype === DTYPE.BLOB && x.name === name);
  if (!r) return null;
  const fd = openSync(path, "r");
  try {
    const out = new Uint8Array(r.bytes);
    readExact(fd, out, r.bytes, base + r.offset);
    return out;
  } finally {
    closeSync(fd);
  }
}
