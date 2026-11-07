/**
 * Byte-level BPE export (.qbp v1 and the v2 header) and its TS re-check, a
 * port of qwen_load (src/tokenizer_qwen.c) plus the v2 marker rules.
 */
import { describe, expect, test } from "bun:test";

import { buildTokenizer, byteAlphabet, readHfBpe } from "../scripts/hf/bpe-tokenizer";
import { type QbpTokenizer, encodeQbp, parseQbp } from "../scripts/smoe/qbp";

const alphabet = byteAlphabet();
/** GPT-2 order: bytes 33..126, 161..172, 174..255, then the rest; id = position. */
const gpt2Order = [...Array(256).keys()].sort((a, b) => alphabet[a]!.codePointAt(0)! - alphabet[b]!.codePointAt(0)!);

function hfJson(pairs: boolean): Record<string, unknown> {
  const vocab: Record<string, number> = {};
  gpt2Order.forEach((byte, id) => { vocab[alphabet[byte]!] = id; });
  Object.assign(vocab, { "Ġt": 256, he: 257, "Ġthe": 258, "<|endoftext07|>": 259 });
  const merges = [["Ġ", "t"], ["h", "e"], ["Ġt", "he"]];
  return {
    normalizer: { type: "NFC" },
    model: { type: "BPE", vocab, merges: pairs ? merges : merges.map((m) => m.join(" ")) },
    added_tokens: [
      { id: 260, content: "<|endoftext|>", special: true },
      { id: 261, content: "<|im_start|>", special: true },
      { id: 262, content: "<|im_end|>", special: true },
    ],
  };
}

const text = (b: Uint8Array) => new TextDecoder().decode(b);

function build(flagUnreachable: boolean, pairs = false) {
  return buildTokenizer(readHfBpe(hfJson(pairs)), {
    vocabSize: 265, bos: "<|endoftext|>", eos: "<|im_end|>", flagUnreachable,
  });
}

describe("token permutation and pieces", () => {
  test("bytes first, then BOS/EOS at 256/257, then every other id ascending", () => {
    const t = build(false);
    expect(t.oldIds.length).toBe(265);
    for (let b = 0; b < 256; b++) expect(t.pieces[b]).toEqual(new Uint8Array([b]));
    expect([...t.oldIds.subarray(256)]).toEqual([260, 262, 256, 257, 258, 259, 261, 263, 264]);
    expect(t.pieces.slice(256).map(text)).toEqual(["<|endoftext|>", "<|im_end|>", " t", "he", " the", "<|endoftext07|>",
      "<|im_start|>", "<unused:263>", "<unused:264>"]);
    for (let n = 0; n < 265; n++) expect(t.toNew[t.oldIds[n]!]).toBe(n);
    expect(t.toNew[gpt2Order.indexOf(0x20)]).toBe(0x20);
  });

  test("flags: added tokens and unused ids are special; unreachable base entries only on request", () => {
    expect([...build(false).special.subarray(256)]).toEqual([1, 1, 0, 0, 0, 0, 1, 1, 1]);
    const phi = build(true);
    expect([...phi.special.subarray(256)]).toEqual([1, 1, 0, 0, 0, 1, 1, 1, 1]);
    expect(phi.stats).toMatchObject({ base: 260, added: 3, unused: 2, unreachable: [259] });
  });

  test("merges map through the permutation; string and pair forms agree", () => {
    const t = build(false);
    expect([...t.merges]).toEqual([0x20, 0x74, 258, 0x68, 0x65, 259, 258, 259, 260]);
    expect([...build(false, true).merges]).toEqual([...t.merges]);
  });

  test.each([
    ["a merge whose output is not in the vocabulary", (j: Record<string, unknown>) => {
      (j.model as { merges: string[] }).merges.push("t h");
    }, /merge/],
    ["a vocabulary id at or above vocab_size", (j: Record<string, unknown>) => {
      (j.model as { vocab: Record<string, number> }).vocab.zz = 400;
    }, /vocab/],
    ["a missing EOS token", (j: Record<string, unknown>) => { j.added_tokens = []; }, /<\|im_end\|>/],
    ["a non-alphabet character", (j: Record<string, unknown>) => {
      (j.model as { vocab: Record<string, number> }).vocab[" x"] = 263;
    }, /byte alphabet/],
    ["a malformed merge string", (j: Record<string, unknown>) => {
      (j.model as { merges: string[] }).merges.push("a b c");
    }, /merge/],
    ["an added-token id used twice", (j: Record<string, unknown>) => {
      (j.added_tokens as unknown[]).push({ id: 260, content: "<|other|>", special: true });
    }, /repeats an id/],
  ])("rejects %s", (_what, mutate, error) => {
    const j = hfJson(false);
    mutate(j);
    expect(() => buildTokenizer(readHfBpe(j), { vocabSize: 265, bos: "<|endoftext|>", eos: "<|im_end|>",
      flagUnreachable: false })).toThrow(error);
  });
});

function qbpOf(version: 1 | 2): QbpTokenizer {
  const t = build(version === 2);
  const header = version === 1 ? { version: 1 as const, imStart: t.toNew[261]! }
    : { version: 2 as const, normalizer: 1 as const, pretokenizer: 1 as const, template: 1 as const };
  return { header, pieces: t.pieces, special: t.special, merges: t.merges };
}

describe(".qbp encode / parse", () => {
  test("v1 round-trips with the documented 24-byte header", () => {
    const bytes = encodeQbp(qbpOf(1));
    const view = new DataView(bytes.buffer);
    expect(String.fromCharCode(...bytes.subarray(0, 8))).toBe("SMQBP1\0\x01");
    expect([view.getUint32(8, true), view.getUint32(12, true), view.getUint32(20, true)]).toEqual([265, 3, 262]);
    const back = parseQbp(bytes, 265);
    expect(back).toEqual(qbpOf(1));
  });

  test("v2 carries normalizer, pre-tokenizer and template in a 32-byte header", () => {
    const bytes = encodeQbp(qbpOf(2));
    const view = new DataView(bytes.buffer);
    expect(bytes[7]).toBe(2);
    expect([view.getUint32(20, true), view.getUint32(24, true), view.getUint32(28, true)]).toEqual([1, 1, 1]);
    expect(view.getUint32(32, true)).toBe(1); /* first piece length */
    expect(parseQbp(bytes, 265)).toEqual(qbpOf(2));
  });

  const v1 = () => Buffer.from(encodeQbp(qbpOf(1)));
  const pieceAt = (id: number) => { /* byte offset of piece record `id` in a v1 file */
    let off = 24;
    const t = qbpOf(1);
    for (let i = 0; i < id; i++) off += 8 + t.pieces[i]!.length;
    return off;
  };
  test.each([
    ["a wrong magic", (b: Buffer) => { b[0] = 0x58; }, 265, /header/],
    ["a vocabulary mismatch", (b: Buffer) => b, 266, /header/],
    ["an out-of-range im_start", (b: Buffer) => { b.writeUInt32LE(265, 20); }, 265, /dimensions/],
    ["a broken byte token", (b: Buffer) => { b[pieceAt(65) + 8] = 66; }, 265, /byte-token/],
    ["a flag of 2", (b: Buffer) => { b.writeUInt32LE(2, pieceAt(300 - 40) + 4); }, 265, /piece/],
    ["a total mismatch", (b: Buffer) => { b.writeUInt32LE(b.readUInt32LE(16) + 1, 16); }, 265, /count mismatch/],
    ["trailing bytes", (b: Buffer) => Buffer.concat([b, Buffer.from([0])]), 265, /extra/],
    ["a truncated merge table", (b: Buffer) => b.subarray(0, b.length - 1), 265, /truncated/],
    ["a merge using a special token", (b: Buffer) => { b.writeUInt32LE(256, b.length - 12); }, 265, /merge/],
    ["a merge whose output is not the concatenation", (b: Buffer) => { b.writeUInt32LE(259, b.length - 4); }, 265, /merge/],
    ["a duplicate merge", (b: Buffer) => { b.copy(b, b.length - 12, b.length - 36, b.length - 24); }, 265, /merge/],
  ])("rejects %s", (_what, mutate, vocab, error) => {
    const b = v1();
    const mutated = mutate(b) ?? b;
    expect(() => parseQbp(mutated, vocab)).toThrow(error);
  });

  test("v2 rejects unknown enum values and missing template markers", () => {
    const base = Buffer.from(encodeQbp(qbpOf(2)));
    for (const [at, value] of [[20, 2], [24, 0], [24, 3], [28, 3]] as const) {
      const b = Buffer.from(base);
      b.writeUInt32LE(value, at);
      expect(() => parseQbp(b, 265)).toThrow(/header|normalizer|pre-tokenizer|template/);
    }
    const phi = Buffer.from(base);
    phi.writeUInt32LE(2, 28);
    expect(() => parseQbp(phi, 265)).toThrow(/<\|system\|>|<\|end\|>/);
    const noFlag = { ...qbpOf(2), special: Uint8Array.from(qbpOf(2).special, (f, i) => (i === 257 ? 0 : f)) };
    expect(() => parseQbp(encodeQbp(noFlag), 265)).toThrow(/256 and 257/);
  });

  test("encode refuses inconsistent input", () => {
    const t = qbpOf(1);
    expect(() => encodeQbp({ ...t, special: t.special.subarray(1) })).toThrow(/flags/);
    expect(() => encodeQbp({ ...t, merges: t.merges.subarray(1) })).toThrow(/triples/);
  });
});
