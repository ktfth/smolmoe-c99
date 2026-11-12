/**
 * SMQBP1 byte-level BPE tokenizers built in the test: the v1 (Qwen) and v2
 * headers, the Phi chat template through the CLI, and every load-time rejection.
 * Real-vocabulary goldens are in models.test.ts (Qwen) and phi4-tokenizer.test.ts.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { BASE_CONFIG, configText } from "./support/fixture";
import {
  bytePieces, encodeQbp, NORMALIZER, PRETOKENIZER, type QbpHeader, type QbpPiece, TEMPLATE, textPiece,
} from "./support/qbp";
import { run, tempDir } from "./support/run";

const work = tempDir("tokenizer");
afterAll(() => work.dispose());

const special = (...texts: string[]): QbpPiece[] => texts.map((t) => textPiece(t, 1));
/* IDs 256.. ; "ab" (the only merge's output) comes last. */
const PHI = [...special("<|endoftext|>", "<|end|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|/tool|>")];
const CHATML = [...special("<|endoftext|>", "<|im_end|>", "<|im_start|>", "<tool_call>", "</tool_call>")];
const PHI_V2: QbpHeader = { version: 2, normalizer: NORMALIZER.none, pretokenizer: PRETOKENIZER.o200k, template: TEMPLATE.phi };
const S = 83, U = 85;

let count = 0;
interface Files { readonly config: string; readonly tokenizer: string }

/** Writes bytes + `extra` + "ab" and a config whose vocab matches. */
function files(header: QbpHeader, extra: readonly QbpPiece[], mutate: (b: Uint8Array) => Uint8Array = (b) => b): Files {
  const pieces = [...bytePieces(), ...extra, textPiece("ab", 0)];
  const ab = pieces.length - 1;
  const name = `t${count++}`;
  const tokenizer = join(work.path, `${name}.qbp`);
  const config = join(work.path, `${name}.ini`);
  writeFileSync(tokenizer, mutate(encodeQbp(header, pieces, [[97, 98, ab]])));
  writeFileSync(config, configText({ ...BASE_CONFIG, vocab_size: pieces.length }));
  return { config, tokenizer };
}

function tokenize(f: Files, prompt: string, extra: readonly string[] = []) {
  const input = join(work.path, `p${count}.txt`);
  const ids = join(work.path, `ids${count++}.txt`);
  writeFileSync(input, prompt);
  const r = run(["--config", f.config, "--tokenizer", f.tokenizer, "--prompt-file", input, "--tokenize-only",
    "--token-ids", ids, ...extra]);
  const list = r.code === 0 ? readFileSync(ids, "utf8").trim().split("\n").filter(Boolean).map(Number) : [];
  return { ...r, ids: list };
}

describe("SMQBP1 v2 with the Phi template", () => {
  const phi = () => files(PHI_V2, PHI);

  test("--chat renders <|system|>S<|end|><|user|>U<|end|><|assistant|>", () => {
    const r = tokenize(phi(), "U", ["--chat", "--system", "S"]);
    expect(r.code).toBe(0);
    expect(r.ids).toEqual([258, S, 257, 259, U, 257, 260]);
    expect(r.stdout.toString()).toBe("SU");
  });

  test("whitespace right after a role marker is dropped, as HF rstrip does", () => {
    const r = tokenize(phi(), "\n \t\u0085U ", ["--chat", "--system", "S"]);
    expect(r.code).toBe(0);
    expect(r.ids).toEqual([258, S, 257, 259, U, 32, 257, 260]);
  });

  test("a user's literal <|end|> stays text", () => {
    const r = tokenize(phi(), "<|end|>", ["--chat", "--system", "S"]);
    expect(r.code).toBe(0);
    expect(r.ids.filter((id) => id === 257)).toHaveLength(2);
    expect(r.stdout.toString()).toBe("S<|end|>");
  });

  test("no normalization: decomposed accents stay decomposed", () => {
    const r = tokenize(phi(), "éab");
    expect(r.code).toBe(0);
    expect(r.ids).toEqual([101, 0xcc, 0x81, 263]);
    expect(r.stdout.toString()).toBe("éab");
  });

  test("an empty plain prompt is just BOS (<|endoftext|>)", () => {
    const r = tokenize(phi(), "");
    expect(r.code).toBe(0);
    expect(r.ids).toEqual([256]);
    expect(r.stdout.toString()).toBe("");
  });
});

describe("SMQBP1 v1 and v2 ChatML", () => {
  test("v2 ChatML encodes like the v1 header it replaces", () => {
    const v1 = tokenize(files({ version: 1, imStart: 258 }, CHATML), "éab", ["--chat", "--system", "S"]);
    const v2 = tokenize(files({ version: 2, normalizer: NORMALIZER.nfc, pretokenizer: PRETOKENIZER.qwen2,
      template: TEMPLATE.chatml }, CHATML), "éab", ["--chat", "--system", "S"]);
    expect(v1.code).toBe(0);
    expect(v2.code).toBe(0);
    expect(v2.ids).toEqual(v1.ids);
    expect(v1.ids.slice(0, 2)).toEqual([258, 115]); /* <|im_start|>s(ystem) */
    expect(v1.stdout.toString()).toBe("system\nS\nuser\néab\nassistant\n");
  });

  test("a tokenizer without a chat template refuses --chat", () => {
    const f = files({ ...PHI_V2, template: TEMPLATE.none }, PHI);
    expect(tokenize(f, "U").code).toBe(0);
    const r = tokenize(f, "U", ["--chat"]);
    expect(r.code).toBe(1);
    expect(r.stderr).toMatch(/chat needs a chat model/);
  });
});

describe("SMQBP1 load-time rejections", () => {
  const rejects = (f: Files, pattern: RegExp) => {
    const r = tokenize(f, "x");
    expect(r.code).toBe(1);
    expect(r.stderr).toStartWith("smolmoe: ");
    expect(r.stderr).toMatch(pattern);
  };
  const header = (fields: Partial<{ normalizer: number; pretokenizer: number; template: number }>): QbpHeader =>
    ({ ...PHI_V2, ...fields });

  test("unknown version", () => {
    rejects(files(PHI_V2, PHI, (b) => b.map((x, i) => (i === 7 ? 3 : x))), /invalid SMQBP1 tokenizer header/);
  });
  test("truncated v2 header", () => {
    rejects(files(PHI_V2, PHI, (b) => b.subarray(0, 28)), /invalid SMQBP1 tokenizer header/);
  });
  test("unknown normalizer, pre-tokenizer or chat template", () => {
    rejects(files(header({ normalizer: 2 }), PHI), /unsupported SMQBP1 normalizer 2/);
    rejects(files(header({ pretokenizer: 0 }), PHI), /unsupported SMQBP1 pre-tokenizer 0/);
    rejects(files(header({ pretokenizer: 5 }), PHI), /unsupported SMQBP1 pre-tokenizer 5/);
    rejects(files(header({ template: 5 }), PHI), /unsupported SMQBP1 chat template 5/);
  });
  test("a Phi template without its role markers", () => {
    const noUser = PHI.map((p) => (new TextDecoder().decode(p.bytes) === "<|user|>" ? textPiece("<|usr|>", 1) : p));
    rejects(files(PHI_V2, noUser), /chat template needs a <\|user\|> token/);
  });
  test("EOS and BOS must be the template's end and end-of-text tokens", () => {
    const swapped = [PHI[1], PHI[0], ...PHI.slice(2)];
    rejects(files(PHI_V2, swapped), /token 256 must be <\|endoftext\|>|token 257 must be <\|end\|>/);
    rejects(files(PHI_V2, [PHI[0], textPiece("<|im_end|>", 1), ...PHI.slice(2)]), /token 257 must be <\|end\|>/);
    const chatml: QbpHeader = { version: 2, normalizer: 1, pretokenizer: 1, template: TEMPLATE.chatml };
    rejects(files(chatml, PHI), /token 257 must be <\|im_end\|>/);
    rejects(files(chatml, [CHATML[0], CHATML[1], textPiece("<|start|>", 1), ...CHATML.slice(3)]),
      /chat template needs a <\|im_start\|> token/);
  });
  test("truncated records and trailing bytes after the merges", () => {
    rejects(files(PHI_V2, PHI, (b) => b.subarray(0, b.length - 5)), /truncated SMQBP1 tokenizer: merges/);
    rejects(files(PHI_V2, PHI, (b) => b.subarray(0, 40)), /truncated SMQBP1 tokenizer/);
    rejects(files(PHI_V2, PHI, (b) => Uint8Array.from([...b, 0])), /extra SMQBP1 tokenizer payload/);
  });
  test("a v1 header's im_start must be <|im_start|>", () => {
    rejects(files({ version: 1, imStart: 259 }, CHATML), /im_start must be the <\|im_start\|> token/);
  });
  test("BOS and EOS must be flagged special", () => {
    rejects(files(PHI_V2, [textPiece("<|endoftext|>", 0), ...PHI.slice(1)]), /tokens 256 and 257 must be flagged/);
    rejects(files(PHI_V2, [PHI[0], textPiece("<|end|>", 0), ...PHI.slice(2)]), /tokens 256 and 257 must be flagged/);
  });
});
