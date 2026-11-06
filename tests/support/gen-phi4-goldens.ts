/**
 * Writes tests/goldens/phi4-tokenizer.json: Hugging Face token IDs of Phi-4-mini (o200k_base) for texts
 * and hand-rendered chat prompts. Not part of the test run; it needs js-tiktoken, which the repository
 * does not depend on:
 *
 *   bun add js-tiktoken@1.0.21          (in any scratch directory DIR)
 *   bun tests/support/gen-phi4-goldens.ts --tiktoken DIR
 *
 * Reference: js-tiktoken o200k_base, the GPT-4o encoding Phi-4-mini uses. Its JavaScript regex differs
 * from HF tokenizers' Oniguruma in three places: \s has U+FEFF and lacks U+0085, and (?i) is ASCII-only
 * (Oniguruma also folds U+017F to s). Cases on those points ("source": "spec") take their IDs from the
 * o200k matcher of src/tokenizer_pretok.h, ported below with the Unicode 16.0.0 classes of
 * scripts/gen-unicode.ts; every js-tiktoken case must also match that port, or nothing is written.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";

import { classBits, generalCategories, readCachedUcd } from "../../scripts/gen-unicode";
import { BYTE_ORDER, PHI4_TOKENIZER_SHA256, phi4Source } from "./phi4-qbp";
import { ROOT } from "./run";

const SYSTEM = 200022, USER = 200021, ASSISTANT = 200019, END = 200020, TOOL = 200023, TOOL_END = 200024;
const L = 1, N = 2, UP = 4, LO = 8;
const space = (c: number) => (c >= 9 && c <= 13) || c === 32 || c === 0x85 || c === 0xa0 || c === 0x1680 ||
  (c >= 0x2000 && c <= 0x200a) || c === 0x2028 || c === 0x2029 || c === 0x202f || c === 0x205f || c === 0x3000;

type Encoder = (text: string) => number[];

/** The o200k pre-tokenizer of src/tokenizer_pretok.h, over code points and class bits. */
export function o200kPieces(text: string, bits: (cp: number) => number): string[] {
  const chars = [...text], c = chars.map((ch) => ch.codePointAt(0)!), k = c.map(bits), n = c.length;
  const run = (j: number, bit: number) => { while (j < n && (k[j] & bit)) ++j; return j; };
  const fold = (x: number) => (x === 0x17f ? 115 : x >= 65 && x <= 90 ? x + 32 : x);
  const contraction = (j: number) => {
    if (j + 1 < n && c[j] === 39) {
      const a = fold(c[j + 1]);
      if (a === 115 || a === 116 || a === 109 || a === 100) return j + 2;
      if (j + 2 < n) {
        const b = fold(c[j + 2]);
        if ((a === 114 && b === 101) || (a === 118 && b === 101) || (a === 108 && b === 108)) return j + 3;
      }
    }
    return j;
  };
  const lower = (s: number) => {
    const e = run(s, UP);
    if (e < n && (k[e] & LO)) return run(e, LO);
    for (let q = e; q > s; --q) if (k[q - 1] & LO) return q;
    return -1;
  };
  const upper = (s: number) => { const e = run(s, UP); return e === s ? -1 : run(e, LO); };
  const prefix = (j: number) => !(k[j] & (L | N)) && c[j] !== 13 && c[j] !== 10;
  const word = (i: number, w: (s: number) => number) => {
    let e = i + 1 < n && prefix(i) ? w(i + 1) : -1;
    if (e < 0) e = w(i);
    return e < 0 ? -1 : contraction(e);
  };
  const punct = (j: number) => !space(c[j]) && !(k[j] & (L | N));
  const boundary = (i: number) => {
    let j = word(i, lower);
    if (j >= 0) return j;
    if ((j = word(i, upper)) >= 0) return j;
    if (k[i] & N) { for (j = i + 1; j < n && j < i + 3 && (k[j] & N); ++j); return j; }
    j = i;
    if (c[j] === 32 && j + 1 < n && punct(j + 1)) ++j;
    if (punct(j)) {
      do ++j; while (j < n && punct(j));
      while (j < n && (c[j] === 13 || c[j] === 10 || c[j] === 47)) ++j;
      return j;
    }
    for (j = i; j < n && space(c[j]); ++j);
    let last = -1;
    for (let q = i; q < j; ++q) if (c[q] === 13 || c[q] === 10) last = q;
    if (last >= 0) return last + 1;
    return j === n || j - i === 1 ? j : j - 1;
  };
  const out: string[] = [];
  for (let i = 0; i < n;) { const j = boundary(i); out.push(chars.slice(i, j).join("")); i = j; }
  return out;
}

export function specEncoder(tokenizerJson: string): Encoder {
  const ucd = readCachedUcd();
  if (!ucd) throw new Error("run bun scripts/gen-unicode.ts first: the pinned UCD files are not cached");
  const gc = generalCategories(ucd.classData);
  const t = JSON.parse(tokenizerJson) as { model: { vocab: Record<string, number>; merges: [string, string][] } };
  const vocab = t.model.vocab;
  const rank = new Map<string, number>(), out: number[] = [];
  t.model.merges.forEach(([a, b], r) => { rank.set(`${vocab[a]},${vocab[b]}`, r); out[r] = vocab[a + b]; });
  const byteId = new Map(BYTE_ORDER.map((b, id) => [b, id]));
  const bpe = (bytes: Uint8Array) => {
    let sym = [...bytes].map((b) => byteId.get(b)!);
    for (;;) {
      let best = Infinity, pos = -1;
      for (let i = 0; i + 1 < sym.length; i++) {
        const r = rank.get(`${sym[i]},${sym[i + 1]}`);
        if (r !== undefined && r < best) { best = r; pos = i; }
      }
      if (pos < 0) return sym;
      sym = [...sym.slice(0, pos), out[best], ...sym.slice(pos + 2)];
    }
  };
  const te = new TextEncoder();
  return (text) => o200kPieces(text, (cp) => classBits(gc[cp])).flatMap((p) => bpe(te.encode(p)));
}

const TEXTS = [
  "Olá, você sabe programar?", "ação, coração, português e memória.",
  "Não sei se é possível: São Paulo, Brasília e Florianópolis às 9h30.",
  "function soma(a, b) {\n  return a + b; // retorna\n}\n\nconst x = soma(1, 2);\n",
  "def f(x):\n    if x > 0:\n        return x ** 2\n    return -x\n",
  "1234567 + 89 = 1234656; 3.14159; 1,000,000; R$ 2.500,00; 2026-11-06",
  "HelloWorld iPhone HELLOworld getHTTPResponseCode XMLHttpRequest camelCaseString",
  "DON'T don't I'm WE'LL she's they've I'd you're It's 3:00pm\u2014isn't it?", "'s alone and 'S",
  "Veja https://example.com/path/to/file.html?x=1&y=2#frag e http://a.b/c//d/", "a/b//\n/c\n",
  "C:\\Users\\Usuario\\war /usr/local/bin/", "中文和日本語と한국어 🌳🧠", "中文测试，日本語のテスト。",
  "emoji 😀👍🏽 ok 👨\u200d👩\u200d👧\u200d👦 🇧🇷", "  Uma frase com espaços.\n\nNova linha.", "\tA\r\n B C D",
  "trailing   ", "  leading", "a  \t  b\n\n\n   c\r\n\r\n", "   \n", "नमस्ते दुनिया", "Ⅻ²٣ ٤٥٦٧",
  "éclair vs e\u0301clair", "!!! ??? ... --- *** ((x)) [[y]] {{z}}", "S", "U", "a", "A",
  /* (?i) also folds U+017F into the contraction; no merge spans ' and ſ, so the IDs agree anyway. */
  "x'ſ, DON'ſ e 'ſa",
  "مرحبا بالعالم ١٢٣", "Καλημέρα κόσμε, Привет мир", "สวัสดีครับ", "\u{2EBF0}\u{2EBF1} e \u1C89\u1C8A",
];
/* Where JavaScript's \s and Oniguruma's disagree, and so do the IDs. */
const SPEC_TEXTS = ["a\u0085\u0085b \u0085x", "a\uFEFF\uFEFFb \uFEFFy"];
const CHATS: readonly { system: string; user: string; note?: string }[] = [
  { system: "S", user: "U" },
  { system: "Responda em português.", user: "\n  Olá!\n", note: "rstrip drops the user's leading whitespace" },
  { system: "", user: "Oi" },
  { system: "S<|tool|> [x]<|/tool|> ", user: "U", note: "system tool declarations, rstrip after each marker" },
  { system: "S", user: "Diga <|end|> agora",
    note: "deliberate divergence: HF would make <|end|> a control token; the runtime keeps user text plain" },
];

/** HF rendering with rstrip markers: whitespace after <|role|>, <|tool|> and <|/tool|> is absorbed. */
function renderChat(encode: Encoder, system: string, user: string): number[] {
  const strip = (s: string) => {
    const cps = [...s];
    let i = 0;
    while (i < cps.length && space(cps[i].codePointAt(0)!)) ++i;
    return cps.slice(i).join("");
  };
  const sys: number[] = [];
  let rest = strip(system);
  for (;;) {
    const at = [rest.indexOf("<|tool|>"), rest.indexOf("<|/tool|>")].filter((x) => x >= 0);
    if (!at.length) { sys.push(...encode(rest)); break; }
    const pos = Math.min(...at), open = rest.startsWith("<|tool|>", pos);
    sys.push(...encode(rest.slice(0, pos)), open ? TOOL : TOOL_END);
    rest = strip(rest.slice(pos + (open ? 8 : 9)));
  }
  return [SYSTEM, ...sys, END, USER, ...encode(strip(user)), END, ASSISTANT];
}

async function main(): Promise<void> {
  const { values } = parseArgs({ options: { tiktoken: { type: "string" } } });
  const source = phi4Source();
  if (!values.tiktoken || !source) throw new Error("usage: --tiktoken DIR, with SMOLMOE_PHI4_SRC holding tokenizer.json");
  const pkg = join(values.tiktoken, "node_modules", "js-tiktoken", "dist");
  const { Tiktoken } = await import(join(pkg, "lite.js"));
  const tiktoken = new Tiktoken((await import(join(pkg, "ranks", "o200k_base.js"))).default);
  const version = JSON.parse(readFileSync(join(pkg, "..", "package.json"), "utf8")).version;
  const reference: Encoder = (s) => [...tiktoken.encode(s)];
  const spec = specEncoder(readFileSync(join(source, "tokenizer.json"), "utf8"));
  const prose = readFileSync(join(ROOT, "tests", "goldens", "eval-pt.txt"), "utf8");
  const cases: Record<string, unknown>[] = [];
  for (const text of [...TEXTS, prose]) {
    const ids = reference(text);
    if (spec(text).join() !== ids.join()) throw new Error(`spec encoder differs from js-tiktoken on ${JSON.stringify(text)}`);
    cases.push({ text, reference_ids: ids, source: "js-tiktoken" });
  }
  for (const text of SPEC_TEXTS) {
    const ids = spec(text);
    if (reference(text).join() === ids.join()) throw new Error(`not an Oniguruma edge case: ${JSON.stringify(text)}`);
    cases.push({ text, reference_ids: ids, source: "spec" });
  }
  for (const c of CHATS) {
    const ids = renderChat(reference, c.system, c.user);
    if (renderChat(spec, c.system, c.user).join() !== ids.join()) throw new Error(`chat differs: ${JSON.stringify(c)}`);
    cases.push({ system: c.system, user: c.user, reference_ids: ids, source: "js-tiktoken", ...(c.note ? { note: c.note } : {}) });
  }
  const header = {
    reference: `js-tiktoken@${version} o200k_base, as Hugging Face IDs of microsoft/Phi-4-mini-instruct; "spec" cases ` +
      "are where JavaScript's regex differs from Oniguruma (see tests/support/gen-phi4-goldens.ts). Chat cases render " +
      "<|system|>S<|end|><|user|>U<|end|><|assistant|> with rstrip after every marker. Tests map IDs to the runtime with " +
      "hfToRuntime (tests/support/phi4-qbp.ts).",
    generator: "bun tests/support/gen-phi4-goldens.ts --tiktoken DIR",
    tokenizer_sha256: PHI4_TOKENIZER_SHA256,
    unicode: "16.0.0",
  };
  /* One case per line keeps the file reviewable. */
  const head = JSON.stringify(header, null, 1).replace(/\n}$/, ",\n \"cases\": [\n");
  const text = head + cases.map((c) => `  ${JSON.stringify(c)}`).join(",\n") + "\n ]\n}\n";
  JSON.parse(text);
  writeFileSync(join(ROOT, "tests", "goldens", "phi4-tokenizer.json"), text);
  console.log(`wrote ${cases.length} cases`);
}

if (import.meta.main) await main();
