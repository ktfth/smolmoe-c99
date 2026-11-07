/**
 * Small byte-level BPE helper for the reference: builds prompt ids and decodes
 * printed tokens straight from tokenizer.json. It is not a full tokenizer:
 * added tokens are only emitted by encodeChat (never parsed out of text), and
 * the pre-tokenizer's case-insensitive contraction group `(?i:'s|'t|...)` is
 * expanded by hand because JS regexes lack inline flags.
 *
 * Sources: tokenizer.json (Split regex + ByteLevel pre-tokenizer, BPE model
 * with ranked merges, ByteLevel decoder) and GPT-2's bytes_to_unicode table.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

/** GPT-2 bytes_to_unicode: printable latin-1 bytes keep their code point, the rest map to 256 + rank. */
export function bytesToUnicode(): readonly string[] {
  const printable = (b: number) => (b >= 0x21 && b <= 0x7e) || (b >= 0xa1 && b <= 0xac) || (b >= 0xae && b <= 0xff);
  let next = 0;
  return Object.freeze(Array.from({ length: 256 }, (_, b) => String.fromCodePoint(printable(b) ? b : 256 + next++)));
}

const BYTE_CHARS = bytesToUnicode();
const CHAR_BYTES: ReadonlyMap<string, number> = new Map(BYTE_CHARS.map((c, b) => [c, b]));

const CONTRACTIONS = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?";
const PRETOKEN = new RegExp(
  [
    `[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+${CONTRACTIONS}`,
    `[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*${CONTRACTIONS}`,
    "\\p{N}{1,3}",
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
    "\\s*[\\r\\n]+",
    "\\s+(?!\\S)",
    "\\s+",
  ].join("|"),
  "gu",
);

/** The Split pre-tokenizer (behavior Removed, invert true: the matches are the pieces). */
export function pretokenize(text: string): string[] {
  return [...text.matchAll(PRETOKEN)].map((m) => m[0]);
}

/** Classic rank-ordered BPE over one pre-token already mapped to byte characters. */
export function bpe(word: string, ranks: ReadonlyMap<string, number>): string[] {
  let parts = [...word];
  for (;;) {
    let bestRank = Infinity;
    let bestAt = -1;
    for (let i = 0; i + 1 < parts.length; i++) {
      const r = ranks.get(`${parts[i]} ${parts[i + 1]}`);
      if (r !== undefined && r < bestRank) (bestRank = r), (bestAt = i);
    }
    if (bestAt < 0) return parts;
    const [a, b] = [parts[bestAt]!, parts[bestAt + 1]!];
    const merged: string[] = [];
    for (let i = 0; i < parts.length; i++) {
      if (i + 1 < parts.length && parts[i] === a && parts[i + 1] === b) merged.push(a + b), i++;
      else merged.push(parts[i]!);
    }
    parts = merged;
  }
}

export interface Tokenizer {
  readonly vocab: ReadonlyMap<string, number>;
  readonly idToToken: ReadonlyMap<number, string>;
  readonly added: ReadonlyMap<number, string>;
  readonly addedByContent: ReadonlyMap<string, number>;
  readonly ranks: ReadonlyMap<string, number>;
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

export function buildTokenizer(json: unknown): Tokenizer {
  if (!isRecord(json) || !isRecord(json.model) || !isRecord(json.model.vocab) || !Array.isArray(json.model.merges)) {
    throw new Error("tokenizer.json: expected model.vocab and model.merges");
  }
  const vocab = new Map<string, number>();
  for (const [token, id] of Object.entries(json.model.vocab)) {
    if (typeof id !== "number" || !Number.isSafeInteger(id)) throw new Error(`tokenizer.json: bad id for ${token}`);
    vocab.set(token, id);
  }
  const ranks = new Map<string, number>();
  json.model.merges.forEach((m: unknown, rank: number) => {
    const pair = typeof m === "string" ? m.split(" ") : m;
    if (!Array.isArray(pair) || pair.length !== 2) throw new Error(`tokenizer.json: bad merge #${rank}`);
    ranks.set(`${pair[0]} ${pair[1]}`, rank);
  });
  const added = new Map<number, string>();
  for (const t of Array.isArray(json.added_tokens) ? json.added_tokens : []) {
    if (!isRecord(t) || typeof t.id !== "number" || typeof t.content !== "string") throw new Error("tokenizer.json: bad added token");
    added.set(t.id, t.content);
  }
  return {
    vocab,
    idToToken: new Map([...vocab].map(([token, id]) => [id, token])),
    added,
    addedByContent: new Map([...added].map(([id, content]) => [content, id])),
    ranks,
  };
}

export function loadTokenizer(modelDir: string): Tokenizer {
  return buildTokenizer(JSON.parse(readFileSync(join(modelDir, "tokenizer.json"), "utf8")));
}

/** Encodes plain text (no special tokens inside) to ids. */
export function encodeText(tok: Tokenizer, text: string): number[] {
  const bytes = new TextEncoder();
  return pretokenize(text).flatMap((piece) => {
    const mapped = [...bytes.encode(piece)].map((b) => BYTE_CHARS[b]!).join("");
    return bpe(mapped, tok.ranks).map((part) => {
      const id = tok.vocab.get(part);
      if (id === undefined) throw new Error(`no vocabulary entry for BPE piece ${JSON.stringify(part)}`);
      return id;
    });
  });
}

function addedId(tok: Tokenizer, content: string): number {
  const id = tok.addedByContent.get(content);
  if (id === undefined) throw new Error(`tokenizer has no added token ${content}`);
  return id;
}

/** Phi-4-mini chat template: <|system|>s<|end|><|user|>u<|end|><|assistant|>, no newlines, no BOS. */
export function encodeChat(tok: Tokenizer, system: string, user: string): number[] {
  const end = addedId(tok, "<|end|>");
  return [
    addedId(tok, "<|system|>"),
    ...encodeText(tok, system),
    end,
    addedId(tok, "<|user|>"),
    ...encodeText(tok, user),
    end,
    addedId(tok, "<|assistant|>"),
  ];
}

/** ByteLevel decode: added tokens verbatim, other tokens via the inverse byte alphabet, then UTF-8. */
export function decodeIds(tok: Tokenizer, ids: readonly number[]): string {
  const out: string[] = [];
  let pending: number[] = [];
  const flush = () => {
    if (pending.length > 0) out.push(new TextDecoder().decode(new Uint8Array(pending)));
    pending = [];
  };
  for (const id of ids) {
    const special = tok.added.get(id);
    if (special !== undefined) (flush(), out.push(special));
    else {
      const token = tok.idToToken.get(id);
      if (token === undefined) (flush(), out.push(`<unk:${id}>`));
      else for (const ch of token) pending.push(CHAR_BYTES.get(ch) ?? 0x3f);
    }
  }
  flush();
  return out.join("");
}
