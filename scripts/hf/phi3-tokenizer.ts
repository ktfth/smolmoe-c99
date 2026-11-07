/**
 * Phi-4-mini tokenizer (GPT-4o o200k byte-level BPE) -> .qbp v2 with
 * normalizer none, pre-tokenizer o200k and the Phi chat template.
 *
 * Runtime ids: bytes 0..255, <|endoftext|> (199999) -> 256, <|end|> (200020)
 * -> 257, every other id ascending. Flag 1: all added tokens, the base
 * entries no merge produces (the <|endoftextNN|> placeholders) and the
 * padding ids past the last token (<unused:N>).
 */
import { encodeQbp, NORMALIZER, parseQbp, PRETOKENIZER, TEMPLATE } from "../smoe/qbp";
import { type BuiltTokenizer, buildTokenizer, readHfBpe } from "./bpe-tokenizer";
import { type Json, asObject, canonicalJson } from "./common";

export const O200K_REGEX = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+" +
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*" +
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

const PRE_TOKENIZER = {
  type: "Sequence", pretokenizers: [
    { type: "Split", pattern: { Regex: O200K_REGEX }, behavior: "Removed", invert: true },
    { type: "ByteLevel", add_prefix_space: false, trim_offsets: true, use_regex: false },
  ],
};
const MODEL_FLAGS = { type: "BPE", dropout: null, unk_token: null, continuing_subword_prefix: "", end_of_word_suffix: "",
  fuse_unk: false, byte_fallback: false, ignore_merges: false };
const REQUIRED = ["<|endoftext|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>"];
const PLACEHOLDER = /^<\|endoftext\d{2}\|>$/;

/** Exact structure of the pinned tokenizer (spec-phi4-tokenizer 1.5, 1.6, 3.6). */
export interface PhiTokenizerPins {
  readonly baseVocab: number;
  readonly merges: number;
  readonly added: readonly (readonly [number, string])[];
  readonly placeholders: number;
  readonly padding: number;
}

export const PHI4_MINI_TOKENIZER: PhiTokenizerPins = {
  baseVocab: 200019, merges: 199742, placeholders: 19, padding: 35,
  added: [[199999, "<|endoftext|>"], [200018, "<|endofprompt|>"], [200019, "<|assistant|>"], [200020, "<|end|>"],
    [200021, "<|user|>"], [200022, "<|system|>"], [200023, "<|tool|>"], [200024, "<|/tool|>"], [200025, "<|tool_call|>"],
    [200026, "<|/tool_call|>"], [200027, "<|tool_response|>"], [200028, "<|tag|>"]],
};

function gate(t: Json): void {
  const fail = (what: string) => new Error(`tokenizer.json: ${what}`);
  if (t.normalizer !== null) throw fail("normalizer must be null (no NFC)");
  if (t.post_processor !== null) throw fail("post_processor must be null (no BOS)");
  if (canonicalJson(t.pre_tokenizer) !== canonicalJson(PRE_TOKENIZER)) throw fail("pre_tokenizer must be the o200k Split + ByteLevel");
  if (asObject(t.decoder, "tokenizer.json decoder").type !== "ByteLevel") throw fail("decoder must be ByteLevel");
  const { vocab: _vocab, merges: _merges, ...flags } = asObject(t.model, "tokenizer.json model");
  if (canonicalJson(flags) !== canonicalJson(MODEL_FLAGS)) throw fail("model must be plain BPE (no dropout, unk, prefixes, byte fallback)");
}

function checkPins(built: BuiltTokenizer, hf: ReturnType<typeof readHfBpe>, pins: PhiTokenizerPins): void {
  const fail = (what: string) => new Error(`tokenizer.json does not match the pinned Phi-4-mini tokenizer: ${what}`);
  if (hf.vocab.size !== pins.baseVocab) throw fail(`${hf.vocab.size} base entries`);
  if (hf.merges.length !== pins.merges) throw fail(`${hf.merges.length} merges`);
  if (canonicalJson(hf.added.map((a) => [a.id, a.content])) !== canonicalJson(pins.added)) throw fail("added tokens");
  const unreachable = built.stats.unreachable.map((id) => built.pieces[built.toNew[id]!]!);
  const text = unreachable.map((p) => new TextDecoder().decode(p));
  if (text.length !== pins.placeholders || !text.every((s) => PLACEHOLDER.test(s))) throw fail(`placeholders ${text.join(" ")}`);
  if (built.stats.unused !== pins.padding) throw fail(`${built.stats.unused} padding ids`);
}

export interface PhiTokenizer {
  readonly bytes: Uint8Array;
  readonly oldIds: Uint32Array;
  readonly toNew: Uint32Array;
  readonly special: Uint8Array;
}

export function phi3Tokenizer(json: unknown, vocab: number, pins: PhiTokenizerPins | null): PhiTokenizer {
  const t = asObject(json, "tokenizer.json");
  gate(t);
  const hf = readHfBpe(t);
  for (const content of REQUIRED) if (!hf.added.some((a) => a.content === content)) throw new Error(`tokenizer.json: ${content} is missing`);
  const built = buildTokenizer(hf, { vocabSize: vocab, bos: "<|endoftext|>", eos: "<|end|>", flagUnreachable: true });
  if (pins) checkPins(built, hf, pins);
  const bytes = encodeQbp({
    header: { version: 2, normalizer: NORMALIZER.none, pretokenizer: PRETOKENIZER.o200k, template: TEMPLATE.phi },
    pieces: built.pieces, special: built.special, merges: built.merges,
  });
  parseQbp(bytes, vocab);
  return { bytes, oldIds: built.oldIds, toNew: built.toNew, special: built.special };
}
