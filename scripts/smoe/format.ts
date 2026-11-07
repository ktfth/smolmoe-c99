/**
 * SMOEQ8 container constants and small helpers shared by the writer, the
 * reader and the converters (docs/FORMAT.md, src/modelfile.c).
 */

export const MAGIC = new Uint8Array([0x53, 0x4d, 0x4f, 0x45, 0x51, 0x38, 0x00]); /* "SMOEQ8\0" */
export const HEADER_BYTES = 16;
export const RECORD_BYTES = 72;
export const NAME_BYTES = 40;
export const MAX_TENSORS = 2_000_000;
export const MAX_COLS = 65536;
export const QK = 32;
export const Q8_BLOCK_BYTES = 36;
export const Q5_BLOCK_BYTES = 22;
export const CONTAINER_ALIGN = 64;
export const BLOB_CONFIG = "__config__";
export const BLOB_TOKENIZER = "__tokenizer__";

export const DTYPE = { F32: 0, Q8: 1, Q5: 2, BLOB: 3 } as const;
export type Dtype = (typeof DTYPE)[keyof typeof DTYPE];
export type Version = 1 | 2 | 3;

const PRINTABLE_ASCII = /^[\x21-\x7e]+$/;

/** Exact payload size of a tensor record (blobs carry their own length). */
export function payloadBytes(dtype: Dtype, rows: number, cols: number): number {
  const blocks = Math.ceil(cols / QK);
  switch (dtype) {
    case DTYPE.F32: return rows * cols * 4;
    case DTYPE.Q8: return rows * blocks * Q8_BLOCK_BYTES;
    case DTYPE.Q5: return rows * blocks * Q5_BLOCK_BYTES;
    default: throw new Error("blob records have no computed payload size");
  }
}

/** strcmp order. Names are printable ASCII, so UTF-16 code unit order is byte order. */
export function compareNames(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Tensor and blob names: 1..39 printable ASCII bytes (they need a NUL inside 40 bytes). */
export function assertName(name: string): void {
  if (!PRINTABLE_ASCII.test(name) || name.length >= NAME_BYTES) throw new Error(`invalid record name: ${JSON.stringify(name)}`);
}

export const alignUp = (x: number, to: number = CONTAINER_ALIGN): number => Math.ceil(x / to) * to;

/** Typed-array views over file bytes assume a little-endian host. */
export function assertLittleEndian(): void {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error("a little-endian host is required");
}

/** Error text of an unknown thrown value. */
export const message = (e: unknown): string => (e instanceof Error ? e.message : String(e));
