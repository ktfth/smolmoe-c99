/**
 * Safetensors headers, single-file or sharded (model.safetensors.index.json).
 * Only headers are read here; payload rows are read on demand (rows.ts), so
 * memory stays bounded whatever the checkpoint size.
 */
import { closeSync, existsSync, fstatSync, openSync, readSync } from "node:fs";
import { join } from "node:path";

import { asObject, readJsonFile } from "./common";

export type SourceDtype = "BF16" | "F16" | "F32";
export const DTYPE_BYTES: Readonly<Record<SourceDtype, number>> = { BF16: 2, F16: 2, F32: 4 };

export interface HfTensor {
  readonly name: string;
  readonly dtype: SourceDtype;
  readonly shape: readonly number[];
  /** path of the file holding the data */
  readonly file: string;
  /** absolute byte offset of the first element in `file` */
  readonly offset: number;
  readonly bytes: number;
}

export interface SafetensorsHeader {
  readonly tensors: readonly HfTensor[];
  readonly metadata: Readonly<Record<string, string>>;
  readonly dataStart: number;
}

export interface Checkpoint {
  readonly tensors: ReadonlyMap<string, HfTensor>;
  /** every weight file, in shard order */
  readonly files: readonly string[];
}

export const MAX_HEADER_BYTES = 16 << 20;
const SHARD_NAME = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*\.safetensors$/;
const INDEX = "model.safetensors.index.json";
const SINGLE = "model.safetensors";

const isCount = (x: unknown): x is number => typeof x === "number" && Number.isSafeInteger(x) && x >= 0;

function readAt(fd: number, length: number, position: number, what: string): Uint8Array {
  const buf = new Uint8Array(length);
  for (let done = 0; done < length;) {
    const n = readSync(fd, buf, done, length - done, position + done);
    if (n <= 0) throw new Error(`${what}: unexpected end of file`);
    done += n;
  }
  return buf;
}

function parseEntry(path: string, name: string, value: unknown, dataStart: number, dataBytes: number): HfTensor {
  const bad = (what: string) => new Error(`${path}: tensor ${name}: ${what}`);
  const e = (value ?? {}) as Record<string, unknown>;
  if (typeof e.dtype !== "string") throw bad("missing dtype");
  if (!Object.hasOwn(DTYPE_BYTES, e.dtype)) throw bad(`unsupported dtype ${e.dtype} (BF16, F16 and F32 only)`);
  const dtype = e.dtype as SourceDtype;
  if (!Array.isArray(e.shape) || !e.shape.every(isCount)) throw bad("invalid shape");
  const shape = e.shape as number[];
  const offsets = e.data_offsets;
  if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(isCount)) throw bad("invalid data_offsets");
  const [begin, end] = offsets as [number, number];
  if (begin > end) throw bad("invalid data_offsets");
  if (end > dataBytes) throw bad("data_offsets out of range");
  const elements = shape.reduce((n, d) => n * d, 1);
  if (!Number.isSafeInteger(elements) || elements * DTYPE_BYTES[dtype] !== end - begin) throw bad("shape and byte size disagree");
  return { name, dtype, shape, file: path, offset: dataStart + begin, bytes: end - begin };
}

/** Reads and validates one file's header: bounded size, JSON object, legal non-overlapping extents. */
export function readSafetensorsHeader(path: string, maxHeader = MAX_HEADER_BYTES): SafetensorsHeader {
  const fd = openSync(path, "r");
  try {
    const size = fstatSync(fd).size;
    if (size < 8) throw new Error(`${path}: not a safetensors file`);
    const n = Number(new DataView(readAt(fd, 8, 0, path).buffer).getBigUint64(0, true));
    if (n < 2 || n > maxHeader || 8 + n > size) throw new Error(`${path}: invalid header length ${n}`);
    let json: unknown;
    try {
      json = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(readAt(fd, n, 8, path)));
    } catch (e) {
      throw new Error(`${path}: header is not valid JSON (${e instanceof Error ? e.message : String(e)})`);
    }
    if (typeof json !== "object" || json === null || Array.isArray(json)) throw new Error(`${path}: header must be a JSON object`);
    const dataStart = 8 + n, dataBytes = size - dataStart;
    const metadata: Record<string, string> = {};
    const tensors: HfTensor[] = [];
    for (const [name, value] of Object.entries(json as Record<string, unknown>)) {
      if (name === "__metadata__") {
        for (const [k, v] of Object.entries(asObject(value, `${path}: __metadata__`))) {
          if (typeof v !== "string") throw new Error(`${path}: __metadata__ values must be strings`);
          metadata[k] = v;
        }
        continue;
      }
      if (!name) throw new Error(`${path}: empty tensor name`);
      tensors.push(parseEntry(path, name, value, dataStart, dataBytes));
    }
    /* Zero-size tensors occupy nothing and may share an offset. */
    const sorted = tensors.filter((t) => t.bytes > 0).sort((a, b) => a.offset - b.offset);
    sorted.forEach((t, i) => {
      const prev = sorted[i - 1];
      if (prev && t.offset < prev.offset + prev.bytes) throw new Error(`${path}: tensors ${prev.name} and ${t.name} overlap`);
    });
    return { tensors, metadata, dataStart };
  } finally {
    closeSync(fd);
  }
}

function readIndex(dir: string): Checkpoint {
  const index = asObject(readJsonFile(join(dir, INDEX)), INDEX);
  const map = asObject(index.weight_map, `${INDEX} weight_map`);
  const byShard = new Map<string, Set<string>>();
  for (const [name, file] of Object.entries(map)) {
    if (typeof file !== "string" || !SHARD_NAME.test(file) || file.includes(".."))
      throw new Error(`${INDEX}: invalid shard file name for ${name}: ${JSON.stringify(file)}`);
    if (!byShard.has(file)) byShard.set(file, new Set());
    byShard.get(file)!.add(name);
  }
  const tensors = new Map<string, HfTensor>();
  const files: string[] = [];
  for (const [file, names] of [...byShard].sort(([a], [b]) => (a < b ? -1 : 1))) {
    const path = join(dir, file);
    const header = readSafetensorsHeader(path);
    const found = new Set(header.tensors.map((t) => t.name));
    if (found.size !== names.size || [...names].some((x) => !found.has(x)))
      throw new Error(`${file}: tensor names do not match ${INDEX}`);
    for (const t of header.tensors) tensors.set(t.name, t);
    files.push(path);
  }
  const total = ((index.metadata ?? {}) as Record<string, unknown>).total_size;
  const sum = [...tensors.values()].reduce((n, t) => n + t.bytes, 0);
  if (total !== undefined && total !== sum) throw new Error(`${INDEX}: total_size ${String(total)} != ${sum} bytes of tensors`);
  return { tensors, files };
}

/** model.safetensors, or the shards listed by model.safetensors.index.json (never both). */
export function openCheckpoint(dir: string): Checkpoint {
  const hasIndex = existsSync(join(dir, INDEX)), hasSingle = existsSync(join(dir, SINGLE));
  if (hasIndex && hasSingle) throw new Error(`${dir}: both ${SINGLE} and ${INDEX} exist; remove one`);
  if (hasIndex) return readIndex(dir);
  if (!hasSingle) throw new Error(`${dir}: no ${SINGLE} or ${INDEX}`);
  const path = join(dir, SINGLE);
  const header = readSafetensorsHeader(path);
  return { tensors: new Map(header.tensors.map((t) => [t.name, t])), files: [path] };
}

/** Reads a whole (small) tensor's bytes. */
export function readTensorBytes(t: HfTensor): Uint8Array {
  const fd = openSync(t.file, "r");
  try {
    return readAt(fd, t.bytes, t.offset, t.name);
  } finally {
    closeSync(fd);
  }
}
