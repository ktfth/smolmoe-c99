/**
 * Writer for the SMQBP1 byte-level BPE tokenizer format (docs/FORMAT.md):
 * version 1 (Qwen2.5, im_start in the header) and version 2 (normalizer,
 * pre-tokenizer and chat template enums).
 */

export const NORMALIZER = { none: 0, nfc: 1 } as const;
export const PRETOKENIZER = { qwen2: 1, o200k: 2 } as const;
export const TEMPLATE = { none: 0, chatml: 1, phi: 2 } as const;

export type QbpHeader =
  | { readonly version: 1; readonly imStart: number }
  | { readonly version: 2; readonly normalizer: number; readonly pretokenizer: number; readonly template: number };

export interface QbpPiece {
  readonly bytes: Uint8Array;
  /** 1 for special, added or unused entries (never produced by a merge, never sampled except EOS). */
  readonly flag: 0 | 1;
}

/** [left, right, output] runtime IDs; record order is rank. */
export type QbpMerge = readonly [number, number, number];

export function encodeQbp(header: QbpHeader, pieces: readonly QbpPiece[], merges: readonly QbpMerge[]): Uint8Array {
  const headerBytes = header.version === 1 ? 24 : 32;
  const pieceBytes = pieces.reduce((sum, p) => sum + p.bytes.length, 0);
  const out = new Uint8Array(headerBytes + pieces.length * 8 + pieceBytes + merges.length * 12);
  const view = new DataView(out.buffer);
  out.set([0x53, 0x4d, 0x51, 0x42, 0x50, 0x31, 0x00, header.version], 0);
  view.setUint32(8, pieces.length, true);
  view.setUint32(12, merges.length, true);
  view.setUint32(16, pieceBytes, true);
  if (header.version === 1) {
    view.setUint32(20, header.imStart, true);
  } else {
    view.setUint32(20, header.normalizer, true);
    view.setUint32(24, header.pretokenizer, true);
    view.setUint32(28, header.template, true);
  }
  let at = headerBytes;
  for (const p of pieces) {
    view.setUint32(at, p.bytes.length, true);
    view.setUint32(at + 4, p.flag, true);
    out.set(p.bytes, at + 8);
    at += 8 + p.bytes.length;
  }
  for (const [a, b, o] of merges) {
    view.setUint32(at, a, true);
    view.setUint32(at + 4, b, true);
    view.setUint32(at + 8, o, true);
    at += 12;
  }
  return out;
}

/** The 256 single-byte pieces every SMQBP1 vocabulary starts with. */
export function bytePieces(): QbpPiece[] {
  return Array.from({ length: 256 }, (_, b) => ({ bytes: Uint8Array.of(b), flag: 0 as const }));
}

export function textPiece(text: string, flag: 0 | 1): QbpPiece {
  return { bytes: new TextEncoder().encode(text), flag };
}
