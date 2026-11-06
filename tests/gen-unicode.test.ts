/**
 * scripts/gen-unicode.ts: the UnicodeData.txt parser, the class bitmask, the
 * NFC table derivation and, when the pinned UCD files are cached locally
 * (SMOLMOE_UCD_DIR or the generator's default cache), that the committed
 * headers are exactly what the generator writes.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  BITS, CLASS_HEADER, classBits, classRanges, generalCategories, NFC_HEADER, nfcTables, readCachedUcd, renderHeaders,
} from "../scripts/gen-unicode";
import { ROOT } from "./support/run";

const line = (cp: number, name: string, gc: string, ccc = 0, decomposition = "") =>
  `${cp.toString(16).toUpperCase().padStart(4, "0")};${name};${gc};${ccc};L;${decomposition};;;;N;;;;;`;

describe("General_Category to class bits", () => {
  test("letters, marks and numbers", () => {
    const { L, N, UPPER, LOWER } = BITS;
    expect([classBits("Lu"), classBits("Lt")]).toEqual([L | UPPER, L | UPPER]);
    expect(classBits("Ll")).toBe(L | LOWER);
    expect([classBits("Lm"), classBits("Lo")]).toEqual([L | UPPER | LOWER, L | UPPER | LOWER]);
    expect([classBits("Mn"), classBits("Mc"), classBits("Me")]).toEqual([UPPER | LOWER, UPPER | LOWER, UPPER | LOWER]);
    expect([classBits("Nd"), classBits("Nl"), classBits("No")]).toEqual([N, N, N]);
    for (const other of ["Zs", "Cn", "Cf", "Po", "Sm", "Pd", "Co"]) expect(classBits(other)).toBe(0);
  });
});

describe("UnicodeData.txt", () => {
  const text = [
    line(0x30, "DIGIT ZERO", "Nd"), line(0x31, "DIGIT ONE", "Nd"), line(0x41, "LATIN CAPITAL LETTER A", "Lu"),
    line(0x42, "LATIN CAPITAL LETTER B", "Lu"), line(0x61, "LATIN SMALL LETTER A", "Ll"),
    line(0x300, "COMBINING GRAVE ACCENT", "Mn", 230), line(0x4e00, "<CJK Ideograph, First>", "Lo"),
    line(0x4e03, "<CJK Ideograph, Last>", "Lo"),
  ].join("\n") + "\n";

  test("expands First/Last ranges and leaves unlisted code points unassigned", () => {
    const gc = generalCategories(text);
    expect(gc.length).toBe(0x110000);
    expect([gc[0x4e00], gc[0x4e02], gc[0x4e03], gc[0x4e04]]).toEqual(["Lo", "Lo", "Lo", "Cn"]);
    expect([gc[0x2f], gc[0x30], gc[0x300]]).toEqual(["Cn", "Nd", "Mn"]);
  });

  test("class ranges merge equal neighbours and omit zero", () => {
    expect(classRanges(generalCategories(text))).toEqual([
      [0x30, 0x31, BITS.N], [0x41, 0x42, BITS.L | BITS.UPPER], [0x61, 0x61, BITS.L | BITS.LOWER],
      [0x300, 0x300, BITS.UPPER | BITS.LOWER], [0x4e00, 0x4e03, BITS.L | BITS.UPPER | BITS.LOWER],
    ]);
  });

  test("NFC tables: canonical decompositions, and composition without exclusions", () => {
    const data = [
      line(0xc0, "LATIN CAPITAL LETTER A WITH GRAVE", "Lu", 0, "0041 0300"),
      line(0xa0, "NO-BREAK SPACE", "Zs", 0, "<noBreak> 0020"),
      line(0x300, "COMBINING GRAVE ACCENT", "Mn", 230),
      line(0x344, "COMBINING GREEK DIALYTIKA TONOS", "Mn", 230, "0308 0301"),
      line(0x958, "DEVANAGARI LETTER QA", "Lo", 0, "0915 093C"),
      line(0xf71, "TIBETAN VOWEL SIGN AA", "Mn", 129),
      line(0xf73, "TIBETAN VOWEL SIGN II", "Mn", 0, "0F71 0F72"),
      line(0x212b, "ANGSTROM SIGN", "Lu", 0, "00C5"),
    ].join("\n") + "\n";
    const t = nfcTables(data, "# comment\n0958    #  DEVANAGARI LETTER QA\n\n");
    expect(t.ccc).toEqual([[0x300, 230], [0x344, 230], [0xf71, 129]]);
    expect(t.decomp).toEqual([[0xc0, 0x41, 0x300], [0x344, 0x308, 0x301], [0x958, 0x915, 0x93c], [0xf73, 0xf71, 0xf72],
      [0x212b, 0xc5, 0]]);
    /* Excluded: listed (0958), non-starter decompositions (0344, 0F73) and singletons (212B). */
    expect(t.compose).toEqual([[0x41, 0x300, 0xc0]]);
  });

  test("a canonical decomposition longer than two code points is refused", () => {
    expect(() => nfcTables(line(0x1, "X", "Lo", 0, "0041 0042 0043") + "\n", "")).toThrow(/decomposition/);
  });
});

const ucd = readCachedUcd();
describe.skipIf(!ucd)("committed Unicode headers", () => {
  test("src/unicode_class.h and src/unicode_nfc.h are what the generator writes", () => {
    const headers = renderHeaders(ucd!);
    expect(readFileSync(join(ROOT, "src", CLASS_HEADER), "utf8")).toBe(headers.classHeader);
    expect(readFileSync(join(ROOT, "src", NFC_HEADER), "utf8")).toBe(headers.nfcHeader);
  });
});
