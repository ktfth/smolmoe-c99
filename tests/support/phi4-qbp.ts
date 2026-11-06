/**
 * Test-side builder of the SMQBP1 v2 tokenizer of Phi-4-mini-instruct from its
 * Hugging Face tokenizer.json (docs/FORMAT.md, "Byte-level BPE format"). It is
 * not the official exporter: it fails fast unless the input is the pinned file.
 *
 * The source directory is SMOLMOE_PHI4_SRC, default downloads/phi-4-mini-instruct
 * in the repository; tests skip when tokenizer.json is absent.
 */
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { encodeQbp, NORMALIZER, PRETOKENIZER, type QbpMerge, type QbpPiece, TEMPLATE } from "./qbp";
import { ROOT } from "./run";

/** microsoft/Phi-4-mini-instruct at cfbefacb99257ffa30c83adab238a50856ac3083. */
export const PHI4_TOKENIZER_SHA256 = "382cc235b56c725945e149cc25f191da667c836655efd0857b004320e90e91ea";
export const PHI4_VOCAB = 200064;
const BASE_ENTRIES = 200019;
const MERGES = 199742;
const ENDOFTEXT = 199999;
const END = 200020;

export const O200K_REGEX = String.raw`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`;

const MODEL = { type: "BPE", dropout: null, unk_token: null, continuing_subword_prefix: "", end_of_word_suffix: "",
  fuse_unk: false, byte_fallback: false, ignore_merges: false };
const PRE_TOKENIZER = { type: "Sequence", pretokenizers: [
  { type: "Split", pattern: { Regex: O200K_REGEX }, behavior: "Removed", invert: true },
  { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: false }] };

/** [id, content, special, rstrip] of every added token (all lstrip=false, normalized=false). */
const ADDED: readonly (readonly [number, string, boolean, boolean])[] = [
  [199999, "<|endoftext|>", true, false], [200018, "<|endofprompt|>", true, false],
  [200019, "<|assistant|>", true, true], [200020, "<|end|>", true, true], [200021, "<|user|>", true, true],
  [200022, "<|system|>", true, true], [200023, "<|tool|>", false, true], [200024, "<|/tool|>", false, true],
  [200025, "<|tool_call|>", false, true], [200026, "<|/tool_call|>", false, true],
  [200027, "<|tool_response|>", false, true], [200028, "<|tag|>", true, true],
];

/** Directory holding tokenizer.json (and config.json), or null when absent. */
export function phi4Source(): string | null {
  const dir = resolve(ROOT, process.env.SMOLMOE_PHI4_SRC ?? join("downloads", "phi-4-mini-instruct"));
  return existsSync(join(dir, "tokenizer.json")) ? dir : null;
}

/** GPT-2 bytes_to_unicode: the byte order of IDs 0..255 and each byte's character. */
export const BYTE_ORDER: readonly number[] = (() => {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) bs.push(b);
  return bs;
})();
const BYTE_CHAR: readonly string[] = (() => {
  const chars: string[] = [];
  /* The first 188 bytes of BYTE_ORDER are printable and map to themselves; the rest to U+0100.. in order. */
  BYTE_ORDER.forEach((b, i) => { chars[b] = String.fromCodePoint(i < 188 ? b : 256 + i - 188); });
  return chars;
})();

/** Hugging Face ID -> runtime ID (bytes first, then <|endoftext|> = BOS 256 and <|end|> = EOS 257). */
export function hfToRuntime(id: number): number {
  if (!Number.isInteger(id) || id < 0 || id >= PHI4_VOCAB) throw new Error(`invalid Phi-4 token ID ${id}`);
  if (id < 256) return BYTE_ORDER[id];
  if (id === ENDOFTEXT) return 256;
  if (id === END) return 257;
  if (id < ENDOFTEXT) return id + 2;
  return id < END ? id + 1 : id;
}

function fail(message: string): never {
  throw new Error(`Phi-4 tokenizer.json: ${message}`);
}

const same = (a: unknown, b: unknown) => Bun.deepEquals(a, b, true);

interface TokenizerJson {
  readonly normalizer: unknown;
  readonly pre_tokenizer: unknown;
  readonly post_processor: unknown;
  readonly decoder: { readonly type?: string } | null;
  readonly added_tokens: readonly { id: number; content: string; special: boolean; rstrip: boolean; lstrip: boolean;
    normalized: boolean; single_word: boolean }[];
  readonly model: { readonly vocab: Record<string, number>; readonly merges: readonly (readonly [string, string])[] };
}

function validate(t: TokenizerJson): void {
  const { vocab: _v, merges: _m, ...model } = t.model;
  if (t.normalizer !== null || t.post_processor !== null) fail("expected no normalizer and no post-processor");
  if (!same(model, MODEL)) fail("unexpected model settings");
  if (!same(t.pre_tokenizer, PRE_TOKENIZER)) fail("unexpected pre-tokenizer");
  if (t.decoder?.type !== "ByteLevel") fail("expected a ByteLevel decoder");
  const added = [...t.added_tokens].sort((a, b) => a.id - b.id).map((a) => [a.id, a.content, a.special, a.rstrip]);
  if (!same(added, ADDED) || t.added_tokens.some((a) => a.lstrip || a.normalized || a.single_word))
    fail("unexpected added tokens");
}

/** Builds the v2 file: normalizer none, o200k pre-tokenizer, Phi template. */
export function buildPhi4Qbp(dir: string): Uint8Array {
  const raw = readFileSync(join(dir, "tokenizer.json"));
  if (createHash("sha256").update(raw).digest("hex") !== PHI4_TOKENIZER_SHA256) fail("SHA-256 is not the pinned one");
  const config = join(dir, "config.json");
  if (existsSync(config) && JSON.parse(readFileSync(config, "utf8")).vocab_size !== PHI4_VOCAB)
    fail("config.json vocab_size is not 200064");
  const t = JSON.parse(raw.toString("utf8")) as TokenizerJson;
  validate(t);
  const vocab = new Map(Object.entries(t.model.vocab));
  const pieceOf: string[] = [];
  for (const [piece, id] of vocab) pieceOf[id] = piece;
  if (vocab.size !== BASE_ENTRIES || pieceOf.length !== BASE_ENTRIES || pieceOf.includes(undefined as never))
    fail("base vocabulary is not IDs 0..200018");
  BYTE_ORDER.forEach((b, i) => { if (vocab.get(BYTE_CHAR[b]) !== i) fail(`byte ${b} is not ID ${i}`); });
  if (t.model.merges.length !== MERGES) fail("unexpected merge count");

  const byteOfChar = new Map(BYTE_CHAR.map((ch, b) => [ch, b]));
  const outputs = new Set<number>();
  const seen = new Set<string>();
  const merges: QbpMerge[] = t.model.merges.map(([a, b]) => {
    const ia = vocab.get(a), ib = vocab.get(b), io = vocab.get(a + b);
    if (ia === undefined || ib === undefined || io === undefined) fail(`merge ${a} ${b} is not in the vocabulary`);
    if (seen.has(`${ia},${ib}`)) fail(`duplicate merge ${a} ${b}`);
    seen.add(`${ia},${ib}`);
    outputs.add(io);
    return [hfToRuntime(ia), hfToRuntime(ib), hfToRuntime(io)] as const;
  });

  const added = new Map(ADDED.map(([id, content]) => [id, content]));
  const pieces: QbpPiece[] = new Array(PHI4_VOCAB);
  let placeholders = 0;
  for (let id = 0; id < PHI4_VOCAB; id++) {
    const content = added.get(id);
    let piece: QbpPiece;
    if (content !== undefined) {
      piece = { bytes: new TextEncoder().encode(content), flag: 1 };
    } else if (id < BASE_ENTRIES) {
      const bytes = Uint8Array.from([...pieceOf[id]], (ch) => byteOfChar.get(ch) ?? fail(`piece ${id} has ${ch}`));
      /* Neither a byte nor any merge's output: an untrained <|endoftextNN|> placeholder. */
      const flag = id < 256 || outputs.has(id) ? 0 : 1;
      if (flag && !pieceOf[id].startsWith("<|endof")) fail(`unexpected unreachable piece ${pieceOf[id]}`);
      placeholders += flag;
      piece = { bytes, flag };
    } else {
      piece = { bytes: new TextEncoder().encode(`<unused:${id}>`), flag: 1 };
    }
    pieces[hfToRuntime(id)] = piece;
  }
  if (placeholders !== 19) fail(`expected 19 placeholder entries, found ${placeholders}`);
  return encodeQbp({ version: 2, normalizer: NORMALIZER.none, pretokenizer: PRETOKENIZER.o200k, template: TEMPLATE.phi },
    pieces, merges);
}
