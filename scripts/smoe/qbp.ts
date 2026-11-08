/**
 * Byte-level BPE tokenizer file (SMQBP1). v1 is what tools/import_qwen2.py
 * wrote (Qwen2: NFC, Qwen2 pre-tokenizer, ChatML). v2 replaces v1's im_start
 * field with three enums so other byte-level BPE models (Phi-4-mini: no
 * normalizer, o200k pre-tokenizer, Phi template) can be described.
 *
 *   "SMQBP1\0" u8 version | u32 vocab | u32 merges | u32 total piece bytes
 *   v1: u32 im_start                       (24-byte header)
 *   v2: u32 normalizer | u32 pretokenizer | u32 template   (32-byte header)
 *   vocab x { u32 len | u32 flag | len bytes } ; merges x { u32 a | u32 b | u32 out }
 *
 * parseQbp ports qwen_load (src/tokenizer_qwen.c) and adds the v2 rules.
 */
export const NORMALIZER = { none: 0, nfc: 1 } as const;
export const PRETOKENIZER = { qwen2: 1, o200k: 2, gemma: 3 } as const;
export const TEMPLATE = { none: 0, chatml: 1, phi: 2, gemma: 3 } as const;
export type Normalizer = (typeof NORMALIZER)[keyof typeof NORMALIZER];
export type PreTokenizer = (typeof PRETOKENIZER)[keyof typeof PRETOKENIZER];
export type ChatTemplate = (typeof TEMPLATE)[keyof typeof TEMPLATE];

export type QbpHeader =
  | { readonly version: 1; readonly imStart: number }
  | { readonly version: 2; readonly normalizer: Normalizer; readonly pretokenizer: PreTokenizer;
    readonly template: ChatTemplate };

export interface QbpTokenizer {
  readonly header: QbpHeader;
  /** piece bytes in runtime-id order */
  readonly pieces: readonly Uint8Array[];
  /** 1 = special/added/unused (never produced by the BPE, banned from sampling) */
  readonly special: Uint8Array;
  /** (a, b, out) triples in rank order */
  readonly merges: Uint32Array;
}

const MAX_MERGES = 1048576;
const MAX_PIECE = 65536;
const MAX_PAYLOAD = 67108864;
const MAGIC = [0x53, 0x4d, 0x51, 0x42, 0x50, 0x31, 0x00]; /* "SMQBP1\0" */
const PHI_MARKERS = ["<|system|>", "<|user|>", "<|assistant|>"];

const fail = (text: string): never => { throw new Error(text); };
const encoder = new TextEncoder();
const same = (a: Uint8Array, text: string) => {
  const b = encoder.encode(text);
  return a.length === b.length && a.every((x, i) => x === b[i]);
};

export function encodeQbp(t: QbpTokenizer): Uint8Array {
  const vocab = t.pieces.length;
  if (t.special.length !== vocab) throw new Error("qbp: flags and pieces differ in length");
  if (t.merges.length % 3) throw new Error("qbp: merges must be (a, b, out) triples");
  const headerBytes = t.header.version === 1 ? 24 : 32;
  const total = t.pieces.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(headerBytes + 8 * vocab + total + 4 * t.merges.length);
  const view = new DataView(out.buffer);
  out.set(MAGIC, 0);
  out[7] = t.header.version;
  view.setUint32(8, vocab, true);
  view.setUint32(12, t.merges.length / 3, true);
  view.setUint32(16, total, true);
  if (t.header.version === 1) view.setUint32(20, t.header.imStart, true);
  else {
    view.setUint32(20, t.header.normalizer, true);
    view.setUint32(24, t.header.pretokenizer, true);
    view.setUint32(28, t.header.template, true);
  }
  let p = headerBytes;
  t.pieces.forEach((piece, i) => {
    view.setUint32(p, piece.length, true);
    view.setUint32(p + 4, t.special[i]!, true);
    out.set(piece, p + 8);
    p += 8 + piece.length;
  });
  for (let i = 0; i < t.merges.length; i++, p += 4) view.setUint32(p, t.merges[i]!, true);
  return out;
}

function parseHeader(view: DataView, vocab: number): { header: QbpHeader; size: number; merges: number; total: number } {
  if (view.byteLength < 24 || MAGIC.some((b, i) => view.getUint8(i) !== b) || view.getUint32(8, true) !== vocab ||
    (view.getUint8(7) !== 1 && view.getUint8(7) !== 2))
    fail("invalid Qwen tokenizer header");
  const merges = view.getUint32(12, true), total = view.getUint32(16, true);
  if (merges > MAX_MERGES || total > MAX_PAYLOAD) fail("invalid Qwen tokenizer dimensions");
  if (view.getUint8(7) === 1) {
    const imStart = view.getUint32(20, true);
    if (imStart >= vocab) fail("invalid Qwen tokenizer dimensions");
    return { header: { version: 1, imStart }, size: 24, merges, total };
  }
  if (view.byteLength < 32) fail("truncated Qwen tokenizer: header");
  const normalizer = view.getUint32(20, true), pretokenizer = view.getUint32(24, true), template = view.getUint32(28, true);
  if (normalizer > 1) fail(`unknown normalizer ${normalizer}`);
  if (pretokenizer < 1 || pretokenizer > 3) fail(`unknown pre-tokenizer ${pretokenizer}`);
  if (template > 3) fail(`unknown chat template ${template}`);
  return {
    header: { version: 2, normalizer: normalizer as Normalizer, pretokenizer: pretokenizer as PreTokenizer,
      template: template as ChatTemplate },
    size: 32, merges, total,
  };
}

function checkMerge(pieces: readonly Uint8Array[], special: Uint8Array, a: number, b: number, out: number): void {
  const vocab = pieces.length;
  if (a >= vocab || b >= vocab || out >= vocab || special[a] || special[b] || special[out]) fail("invalid Qwen merge");
  const pa = pieces[a]!, pb = pieces[b]!, po = pieces[out]!;
  if (pa.length + pb.length !== po.length) fail("invalid Qwen merge");
  for (let i = 0; i < pa.length; i++) if (po[i] !== pa[i]) fail("invalid Qwen merge");
  for (let i = 0; i < pb.length; i++) if (po[pa.length + i] !== pb[i]) fail("invalid Qwen merge");
}

/** v2 only: 256/257 are special and the template's markers exist (found by text, like find_special). */
function checkTemplate(t: QbpTokenizer): void {
  if (t.header.version !== 2) return;
  if (t.special[256] !== 1 || t.special[257] !== 1) fail("tokens 256 and 257 must be flagged special");
  const hasSpecial = (text: string) => t.pieces.some((p, i) => i >= 256 && t.special[i] && same(p, text));
  if (t.header.template === TEMPLATE.chatml) {
    if (!hasSpecial("<|im_start|>") || !same(t.pieces[257]!, "<|im_end|>")) fail("chatml template needs <|im_start|> and EOS <|im_end|>");
  } else if (t.header.template === TEMPLATE.phi) {
    for (const m of PHI_MARKERS) if (!hasSpecial(m)) fail(`phi template needs the special token ${m}`);
    if (!same(t.pieces[257]!, "<|end|>") || !same(t.pieces[256]!, "<|endoftext|>"))
      fail("phi template needs <|endoftext|> at 256 and <|end|> at 257");
  } else if (t.header.template === TEMPLATE.gemma) {
    if (!hasSpecial("<start_of_turn>")) fail("gemma template needs the special token <start_of_turn>");
    if (!same(t.pieces[257]!, "<end_of_turn>") || !same(t.pieces[256]!, "<bos>"))
      fail("gemma template needs <bos> at 256 and <end_of_turn> at 257");
  }
}

/** Port of qwen_load: every structural check, then the v2 template rules. */
export function parseQbp(bytes: Uint8Array, vocab: number): QbpTokenizer {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const { header, size, merges: nmerge, total } = parseHeader(view, vocab);
  const pieces: Uint8Array[] = [];
  const special = new Uint8Array(vocab);
  let p = size, used = 0;
  for (let i = 0; i < vocab; i++) {
    if (p + 8 > bytes.length) fail("truncated Qwen tokenizer: token record");
    const n = view.getUint32(p, true), flag = view.getUint32(p + 4, true);
    if (!n || n > MAX_PIECE || n > total - used || flag > 1) fail("invalid Qwen piece");
    if (p + 8 + n > bytes.length) fail("truncated Qwen tokenizer: piece bytes");
    const piece = bytes.slice(p + 8, p + 8 + n);
    if (i < 256 && (n !== 1 || piece[0] !== i || flag)) fail("invalid Qwen byte-token permutation");
    pieces.push(piece);
    special[i] = flag;
    used += n;
    p += 8 + n;
  }
  if (used !== total) fail("Qwen piece byte count mismatch");
  const merges = new Uint32Array(nmerge * 3);
  const seen = new Set<number>();
  for (let i = 0; i < nmerge; i++, p += 12) {
    if (p + 12 > bytes.length) fail("truncated Qwen tokenizer: merges");
    const a = view.getUint32(p, true), b = view.getUint32(p + 4, true), out = view.getUint32(p + 8, true);
    checkMerge(pieces, special, a, b, out);
    const key = a * 0x80000 + b; /* ids < 2^19 after the vocab check */
    if (seen.has(key)) fail("duplicate Qwen merge");
    seen.add(key);
    merges.set([a, b, out], i * 3);
  }
  if (p !== bytes.length) fail("extra Qwen tokenizer payload");
  const t: QbpTokenizer = { header, pieces, special, merges };
  checkTemplate(t);
  return t;
}
