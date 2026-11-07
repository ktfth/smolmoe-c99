/**
 * Minimal safetensors reader for the independent reference. Parses the header
 * itself (u64 little-endian length, then JSON) and reads tensor rows straight
 * from the shard files on demand, so weights are streamed rather than loaded.
 * Format: https://github.com/huggingface/safetensors (data_offsets are relative
 * to the first byte after the header).
 */
import { closeSync, existsSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { join } from "node:path";

import { decodeBf16 } from "./bf16";

export interface TensorInfo {
  readonly name: string;
  readonly dtype: string;
  readonly shape: readonly number[];
  readonly file: string;
  /** Absolute byte offset of the first element in `file`. */
  readonly start: number;
  readonly byteLength: number;
}

const DTYPE_BYTES: Readonly<Record<string, number>> = {
  BOOL: 1, U8: 1, I8: 1, F8_E4M3: 1, F8_E5M2: 1, I16: 2, U16: 2, F16: 2, BF16: 2, I32: 4, U32: 4, F32: 4, I64: 8, U64: 8, F64: 8,
};
const MAX_HEADER_BYTES = 100 * 1024 * 1024;
const INDEX_FILE = "model.safetensors.index.json";
const SINGLE_FILE = "model.safetensors";

if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) {
  throw new Error("safetensors reader assumes a little-endian host");
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isCount = (v: unknown): v is number => typeof v === "number" && Number.isSafeInteger(v) && v >= 0;

function parseEntry(name: string, entry: unknown, dataStart: number, file: string): TensorInfo {
  if (!isRecord(entry)) throw new Error(`${file}: tensor ${name} is not an object`);
  const { dtype, shape, data_offsets: offsets } = entry;
  if (typeof dtype !== "string" || !Object.hasOwn(DTYPE_BYTES, dtype)) throw new Error(`${file}: ${name} has unknown dtype ${String(dtype)}`);
  if (!Array.isArray(shape) || !shape.every(isCount)) throw new Error(`${file}: ${name} has a bad shape`);
  if (!Array.isArray(offsets) || offsets.length !== 2 || !offsets.every(isCount)) {
    throw new Error(`${file}: ${name} has bad data_offsets`);
  }
  const [begin, end] = offsets as [number, number];
  if (begin > end) throw new Error(`${file}: ${name} has bad data_offsets`);
  const count = shape.reduce((a: number, b: number) => a * b, 1);
  const byteLength = end - begin;
  if (byteLength !== count * DTYPE_BYTES[dtype]!) throw new Error(`${file}: ${name} offsets disagree with shape ${shape}`);
  return { name, dtype, shape: [...shape], file, start: dataStart + begin, byteLength };
}

/** Parses `bytes` = the 8-byte length prefix plus the JSON header of one file. */
export function parseSafetensorsHeader(bytes: Uint8Array, file: string): Map<string, TensorInfo> {
  if (bytes.length < 8) throw new Error(`${file}: truncated safetensors prefix`);
  const headerLength = Number(new DataView(bytes.buffer, bytes.byteOffset, 8).getBigUint64(0, true));
  if (headerLength > MAX_HEADER_BYTES || bytes.length < 8 + headerLength) throw new Error(`${file}: bad header length ${headerLength}`);
  const parsed: unknown = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLength)));
  if (!isRecord(parsed)) throw new Error(`${file}: header is not a JSON object`);
  const out = new Map<string, TensorInfo>();
  for (const [name, entry] of Object.entries(parsed)) {
    if (name !== "__metadata__") out.set(name, parseEntry(name, entry, 8 + headerLength, file));
  }
  return out;
}

function readExactly(fd: number, length: number, position: number, file: string): Uint8Array {
  const buf = new Uint8Array(length); // fresh buffer: byteOffset 0, so typed views are aligned
  let done = 0;
  while (done < length) {
    const n = readSync(fd, buf, done, length - done, position + done);
    if (n <= 0) throw new Error(`${file}: unexpected end of file at byte ${position + done}`);
    done += n;
  }
  return buf;
}

function readHeader(fd: number, file: string): Map<string, TensorInfo> {
  const prefix = readExactly(fd, 8, 0, file);
  const headerLength = Number(new DataView(prefix.buffer).getBigUint64(0, true));
  if (headerLength > MAX_HEADER_BYTES) throw new Error(`${file}: bad header length ${headerLength}`);
  const tensors = parseSafetensorsHeader(readExactly(fd, 8 + headerLength, 0, file), file);
  const size = fstatSync(fd).size;
  for (const t of tensors.values()) {
    if (t.start + t.byteLength > size) throw new Error(`${file}: ${t.name} runs past the end of the file`);
  }
  return tensors;
}

function shardFiles(modelDir: string): Map<string, string> {
  const indexPath = join(modelDir, INDEX_FILE);
  if (!existsSync(indexPath)) return new Map();
  const index: unknown = JSON.parse(readFileSync(indexPath, "utf8"));
  if (!isRecord(index) || !isRecord(index.weight_map)) throw new Error(`${indexPath}: missing weight_map`);
  const map = new Map<string, string>();
  for (const [name, file] of Object.entries(index.weight_map)) {
    if (typeof file !== "string" || file.includes("/") || file.includes("\\")) throw new Error(`${indexPath}: bad shard for ${name}`);
    map.set(name, file);
  }
  return map;
}

/** Read-only view of a (possibly sharded) safetensors checkpoint. */
export class WeightStore {
  private constructor(
    private readonly tensors: ReadonlyMap<string, TensorInfo>,
    private readonly fds: ReadonlyMap<string, number>,
  ) {}

  static open(modelDir: string): WeightStore {
    const weightMap = shardFiles(modelDir);
    const files = weightMap.size > 0 ? [...new Set(weightMap.values())] : [SINGLE_FILE];
    const fds = new Map<string, number>();
    const tensors = new Map<string, TensorInfo>();
    try {
      for (const file of files) {
        const fd = openSync(join(modelDir, file), "r");
        fds.set(file, fd);
        for (const [name, info] of readHeader(fd, file)) tensors.set(name, info);
      }
      for (const [name, file] of weightMap) {
        if (tensors.get(name)?.file !== file) throw new Error(`index maps ${name} to ${file}, but it is not there`);
      }
    } catch (err) {
      for (const fd of fds.values()) closeSync(fd);
      throw err;
    }
    return new WeightStore(tensors, fds);
  }

  has(name: string): boolean {
    return this.tensors.has(name);
  }

  info(name: string): TensorInfo {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`tensor not found: ${name}`);
    return t;
  }

  shape(name: string): readonly number[] {
    return this.info(name).shape;
  }

  /** Raw bf16 bits of rows [rowStart, rowStart + rowCount) of a BF16 tensor (row = first axis). */
  readRowsBf16(name: string, rowStart: number, rowCount: number): Uint16Array {
    const t = this.info(name);
    if (t.dtype !== "BF16") throw new Error(`${name}: dtype ${t.dtype}, expected BF16`);
    const rows = t.shape.length === 0 ? 1 : t.shape[0]!;
    if (!isCount(rowStart) || !isCount(rowCount) || rowStart + rowCount > rows) {
      throw new RangeError(`${name}: rows [${rowStart}, ${rowStart + rowCount}) outside ${rows}`);
    }
    const rowBytes = rows === 0 ? 0 : t.byteLength / rows;
    const fd = this.fds.get(t.file)!;
    const bytes = readExactly(fd, rowCount * rowBytes, t.start + rowStart * rowBytes, t.file);
    return new Uint16Array(bytes.buffer, 0, bytes.length / 2);
  }

  /** Rows decoded to float32 (exact). */
  readRows(name: string, rowStart: number, rowCount: number): Float32Array {
    return decodeBf16(this.readRowsBf16(name, rowStart, rowCount));
  }

  /** A whole 1-D BF16 tensor, decoded. */
  readVector(name: string): Float32Array {
    const shape = this.shape(name);
    if (shape.length !== 1) throw new Error(`${name}: shape [${shape}] is not a vector`);
    return this.readRows(name, 0, shape[0]!);
  }

  close(): void {
    for (const fd of this.fds.values()) closeSync(fd);
  }
}
