/**
 * Tests for the reference's small byte-level BPE helper (used only to build the
 * prompt ids and to print decoded tokens). Toy vocabularies; expected values by
 * hand from the GPT-2 bytes_to_unicode construction.
 */
import { describe, expect, test } from "bun:test";

import { bpe, buildTokenizer, bytesToUnicode, decodeIds, encodeChat, encodeText, pretokenize } from "../scripts/ref/vocab";

describe("GPT-2 byte alphabet", () => {
  const table = bytesToUnicode();

  test("printable latin-1 bytes map to themselves", () => {
    expect(table[0x41]).toBe("A");
    expect(table[0x21]).toBe("!");
    expect(table[0xc3]).toBe("Ã");
    expect(table[0xa1]).toBe("¡");
  });

  test("other bytes map to 256 + their rank among the unprintable bytes", () => {
    expect(table[0x00]).toBe("Ā"); // rank 0
    expect(table[0x0a]).toBe("Ċ"); // newline, rank 10 -> "Ċ"
    expect(table[0x20]).toBe("Ġ"); // space, rank 32 -> "Ġ"
    expect(table[0x7f]).toBe("ġ"); // DEL, rank 33
    expect(table[0xad]).toBe("Ń"); // soft hyphen, the last unprintable (rank 67)
  });

  test("is a bijection over 256 distinct code points", () => {
    expect(new Set(table).size).toBe(256);
  });
});

describe("pre-tokenizer split", () => {
  test("Portuguese sentences split on word boundaries with a leading space", () => {
    expect(pretokenize("Qual é a capital do Brasil?")).toEqual(["Qual", " é", " a", " capital", " do", " Brasil", "?"]);
    expect(pretokenize("Você é um assistente.")).toEqual(["Você", " é", " um", " assistente", "."]);
  });

  test("numbers group in runs of at most three digits", () => {
    expect(pretokenize("12345")).toEqual(["123", "45"]);
  });
});

describe("BPE merges by rank", () => {
  const ranks = new Map([
    ["a b", 0],
    ["ab c", 1],
    ["b c", 2],
  ]);

  test("lowest rank first, repeated until no pair merges", () => {
    expect(bpe("abc", ranks)).toEqual(["abc"]); // ab(0) -> ab c(1)
    expect(bpe("abcbc", ranks)).toEqual(["abc", "bc"]); // ab, abc, then bc
    expect(bpe("cba", ranks)).toEqual(["c", "b", "a"]);
  });
});

describe("encode and decode with a toy tokenizer", () => {
  const tok = buildTokenizer({
    model: {
      vocab: { "Ġol": 0, "Ã¡": 1, "Ġ": 2, o: 3, l: 4, "Ã": 5, "¡": 6, "Ġo": 7 },
      merges: [
        ["Ġ", "o"],
        ["Ġo", "l"],
        ["Ã", "¡"],
      ],
    },
    added_tokens: [
      { id: 10, content: "<|user|>" },
      { id: 11, content: "<|end|>" },
      { id: 12, content: "<|assistant|>" },
      { id: 13, content: "<|system|>" },
    ],
  });

  test("decode maps token characters back to UTF-8 bytes (\" olá\" = Ġol + Ã¡)", () => {
    expect(decodeIds(tok, [0, 1])).toBe(" olá");
    expect(decodeIds(tok, [10])).toBe("<|user|>");
  });

  test("encode runs byte-level BPE per pre-token", () => {
    expect(encodeText(tok, " olá")).toEqual([0, 1]);
  });

  test("chat prompt: <|system|>s<|end|><|user|>u<|end|><|assistant|> with no newlines", () => {
    expect(encodeChat(tok, "á", " ol")).toEqual([13, 1, 11, 10, 0, 11, 12]);
  });

  test("throws on a piece with no vocabulary entry", () => {
    expect(() => encodeText(tok, "z")).toThrow();
  });
});
