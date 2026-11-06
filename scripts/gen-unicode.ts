/**
 * Regenerates the runtime's Unicode tables from pinned Unicode Character Database files:
 *   src/unicode_class.h  General_Category as pre-tokenizer class bits, Unicode 16.0.0
 *                        (the version of Oniguruma 6.9.10, the regex engine of HF tokenizers)
 *   src/unicode_nfc.h    canonical combining classes, decompositions and primary composites
 *                        for NFC, Unicode 15.0.0 (unchanged since the Python generator)
 *
 *   bun scripts/gen-unicode.ts [--ucd DIR] [--check]
 *
 * The UCD files are downloaded once into DIR (default $SMOLMOE_UCD_DIR, else <tmp>/smolmoe-ucd) and
 * every file is SHA-256 verified before use. --check compares instead of writing (exit 1 on drift).
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

export interface UcdFile {
  readonly name: string;
  readonly version: string;
  readonly sha256: string;
}

export const UCD_FILES = {
  classData: { name: "UnicodeData", version: "16.0.0",
    sha256: "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f" },
  nfcData: { name: "UnicodeData", version: "15.0.0",
    sha256: "806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73" },
  exclusions: { name: "CompositionExclusions", version: "15.0.0",
    sha256: "3b019c0a33c3140cbc920c078f4f9af2680ba4f71869c8d4de5190667c70b6a3" },
} as const satisfies Record<string, UcdFile>;

export const CLASS_HEADER = "unicode_class.h";
export const NFC_HEADER = "unicode_nfc.h";

/** Class bits; UPPER = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}], LOWER = [\p{Ll}\p{Lm}\p{Lo}\p{M}] (o200k word classes). */
export const BITS = { L: 1, N: 2, UPPER: 4, LOWER: 8 } as const;

const MAX_CP = 0x110000;

export const ucdUrl = (f: UcdFile) => `https://www.unicode.org/Public/${f.version}/ucd/${f.name}.txt`;
export const ucdFileName = (f: UcdFile) => `${f.name}-${f.version}.txt`;
export const defaultUcdDir = () => resolve(process.env.SMOLMOE_UCD_DIR ?? join(tmpdir(), "smolmoe-ucd"));

export function classBits(category: string): number {
  const { L, N, UPPER, LOWER } = BITS;
  switch (category) {
    case "Lu": case "Lt": return L | UPPER;
    case "Ll": return L | LOWER;
    case "Lm": case "Lo": return L | UPPER | LOWER;
    case "Mn": case "Mc": case "Me": return UPPER | LOWER;
    case "Nd": case "Nl": case "No": return N;
    default: return 0;
  }
}

interface Row {
  readonly cp: number;
  readonly name: string;
  readonly category: string;
  readonly ccc: number;
  readonly decomposition: string;
}

function rows(text: string): Row[] {
  return text.split("\n").filter((l) => l.trim()).map((l) => {
    const f = l.split(";");
    const cp = parseInt(f[0], 16), ccc = Number(f[3]);
    if (f.length < 6 || !Number.isInteger(cp) || cp < 0 || cp >= MAX_CP || !Number.isInteger(ccc))
      throw new Error(`malformed UnicodeData line: ${l}`);
    return { cp, name: f[1], category: f[2], ccc, decomposition: f[5] };
  });
}

/** General_Category of every code point; <..., First>/<..., Last> pairs expanded, unlisted = Cn. */
export function generalCategories(text: string): string[] {
  const gc = new Array<string>(MAX_CP).fill("Cn");
  let first = -1;
  for (const r of rows(text)) {
    if (r.name.endsWith(", First>")) { first = r.cp; continue; }
    if (r.name.endsWith(", Last>")) {
      if (first < 0) throw new Error(`range end without start: ${r.name}`);
      gc.fill(r.category, first, r.cp + 1);
      first = -1;
      continue;
    }
    gc[r.cp] = r.category;
  }
  return gc;
}

/** Maximal runs of equal nonzero bits as [first, last, bits]. */
export function classRanges(categories: readonly string[]): [number, number, number][] {
  const out: [number, number, number][] = [];
  categories.forEach((g, cp) => {
    const bits = classBits(g), last = out.at(-1);
    if (!bits) return;
    if (last && last[1] === cp - 1 && last[2] === bits) last[1] = cp;
    else out.push([cp, cp, bits]);
  });
  return out;
}

export interface NfcTables {
  readonly ccc: [number, number][];
  readonly decomp: [number, number, number][];
  readonly compose: [number, number, number][];
}

/** Composition pairs exclude the listed exclusions, singletons and non-starter decompositions
 * (Full_Composition_Exclusion of UAX #15). */
export function nfcTables(unicodeData: string, exclusions: string): NfcTables {
  const list = rows(unicodeData);
  const cccOf = new Map(list.map((r) => [r.cp, r.ccc]));
  const excluded = new Set(exclusions.split("\n").map((l) => l.split("#")[0].trim()).filter(Boolean)
    .map((h) => parseInt(h, 16)));
  const ccc: [number, number][] = [];
  const decomp: [number, number, number][] = [];
  const compose: [number, number, number][] = [];
  for (const r of list.sort((a, b) => a.cp - b.cp)) {
    if (r.ccc) ccc.push([r.cp, r.ccc]);
    if (!r.decomposition || r.decomposition.startsWith("<")) continue;
    const d = r.decomposition.split(" ").map((h) => parseInt(h, 16));
    if (d.length > 2) throw new Error(`unexpected canonical decomposition of ${r.cp.toString(16)}`);
    decomp.push([r.cp, d[0], d[1] ?? 0]);
    if (d.length === 2 && !excluded.has(r.cp) && !r.ccc && !cccOf.get(d[0])) compose.push([d[0], d[1], r.cp]);
  }
  compose.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  return { ccc, decomp, compose };
}

export interface UcdTexts {
  readonly classData: string;
  readonly nfcData: string;
  readonly exclusions: string;
}

const pin = (f: UcdFile) => `${ucdFileName(f)} (sha256 ${f.sha256})`;
const rowsOf = (items: readonly (readonly number[])[]) => items.map((x) => `{${x.join(",")}},`);

export function renderHeaders(ucd: UcdTexts): { classHeader: string; nfcHeader: string } {
  const { classData, nfcData, exclusions } = UCD_FILES;
  const t = nfcTables(ucd.nfcData, ucd.exclusions);
  const classHeader = [
    `/* Generated by scripts/gen-unicode.ts from ${pin(classData)}. Do not edit. */`,
    "/* Pre-tokenizer classes: UC_L = \\p{L}, UC_N = \\p{N}, UC_UPPER = [\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}],",
    " * UC_LOWER = [\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]. Marks are UC_UPPER|UC_LOWER but not UC_L. */",
    `enum { UC_L = ${BITS.L}, UC_N = ${BITS.N}, UC_UPPER = ${BITS.UPPER}, UC_LOWER = ${BITS.LOWER} };`,
    "typedef struct {uint32_t a,b,bits;} UnicodeClassRange;",
    "static const UnicodeClassRange unicode_class[]={",
    ...rowsOf(classRanges(generalCategories(ucd.classData))), "};", "",
  ].join("\n");
  const nfcHeader = [
    `/* Generated by scripts/gen-unicode.ts from ${pin(nfcData)} and ${pin(exclusions)}. Do not edit. */`,
    "typedef struct {uint32_t a,b,value;} UnicodeRange;",
    "typedef struct {uint32_t cp,value;} UnicodeCCC;",
    "static const UnicodeCCC unicode_ccc[]={", ...rowsOf(t.ccc), "};",
    "static const UnicodeRange unicode_decomp[]={", ...rowsOf(t.decomp), "};",
    "static const UnicodeRange unicode_compose[]={", ...rowsOf(t.compose), "};", "",
  ].join("\n");
  return { classHeader, nfcHeader };
}

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

function verified(f: UcdFile, bytes: Uint8Array): string {
  const got = sha256(bytes);
  if (got !== f.sha256) throw new Error(`${ucdFileName(f)}: SHA-256 ${got}, expected ${f.sha256}`);
  return new TextDecoder().decode(bytes);
}

/** The pinned files from `dir` when all are present (and verified), else null. Never downloads. */
export function readCachedUcd(dir: string = defaultUcdDir()): UcdTexts | null {
  const paths = Object.values(UCD_FILES).map((f) => join(dir, ucdFileName(f)));
  if (!paths.every(existsSync)) return null;
  const read = (f: UcdFile) => verified(f, readFileSync(join(dir, ucdFileName(f))));
  return { classData: read(UCD_FILES.classData), nfcData: read(UCD_FILES.nfcData), exclusions: read(UCD_FILES.exclusions) };
}

async function fetchUcd(dir: string): Promise<UcdTexts> {
  mkdirSync(dir, { recursive: true });
  for (const f of Object.values(UCD_FILES)) {
    const path = join(dir, ucdFileName(f));
    if (existsSync(path)) continue;
    const response = await fetch(ucdUrl(f));
    if (!response.ok) throw new Error(`${ucdUrl(f)}: HTTP ${response.status}`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    verified(f, bytes);
    writeFileSync(`${path}.partial`, bytes);
    renameSync(`${path}.partial`, path);
  }
  const ucd = readCachedUcd(dir);
  if (!ucd) throw new Error(`UCD files missing from ${dir}`);
  return ucd;
}

async function main(): Promise<number> {
  const { values } = parseArgs({ options: { ucd: { type: "string" }, check: { type: "boolean", default: false } } });
  const src = resolve(import.meta.dir, "..", "src");
  const headers = renderHeaders(await fetchUcd(resolve(values.ucd ?? defaultUcdDir())));
  const outputs: [string, string][] = [[CLASS_HEADER, headers.classHeader], [NFC_HEADER, headers.nfcHeader]];
  let drift = 0;
  for (const [name, text] of outputs) {
    const path = join(src, name);
    if (!values.check) writeFileSync(path, text);
    else if (!existsSync(path) || readFileSync(path, "utf8") !== text) {
      console.error(`${name} differs from the generator output`);
      ++drift;
    }
  }
  console.log(values.check ? `checked ${outputs.length} headers` : `wrote ${outputs.map(([n]) => n).join(", ")}`);
  return drift ? 1 : 0;
}

if (import.meta.main) process.exit(await main());
