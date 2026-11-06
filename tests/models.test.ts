/**
 * End-to-end checks with the trained models in models/ (skipped when the
 * weights are not present: they are distributed separately from the source).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import golden from "./goldens/qwen-tokenizer.json";
import { bundledModel, modelArgs, run, runOk, runWithInput, SCALAR_BINARY, tempDir } from "./support/run";

const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
const stories = bundledModel("tinystories-15m", "tokenizer.spb");
const work = tempDir("models");
afterAll(() => work.dispose());

interface GoldenCase {
  readonly text?: string;
  readonly chatml_prompt?: string;
  readonly system?: string;
  readonly runtime_ids: readonly number[];
}

/* make sanitize raises it: under ASan the determinism test takes ~5x longer (160 s on CI). */
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);

describe.skipIf(!qwen)("Qwen2.5-0.5B-Instruct", () => {
  const tokenize = (bytes: Uint8Array, extra: readonly string[] = []) => {
    const input = join(work.path, "prompt.txt");
    const ids = join(work.path, "ids.txt");
    writeFileSync(input, bytes);
    const r = run([...modelArgs(qwen!).slice(2), "--prompt-file", input, "--context", 1024, "--tokenize-only",
      "--token-ids", ids, ...extra]);
    const list = r.code === 0 ? Bun.file(ids).text() : Promise.resolve("");
    return { ...r, ids: list };
  };

  test("tokenizer matches Hugging Face reference IDs (remapped) on every golden case", async () => {
    for (const c of (golden as { cases: GoldenCase[] }).cases) {
      const chat = c.chatml_prompt !== undefined;
      const text = c.text ?? c.chatml_prompt ?? "";
      const extra = chat ? ["--chat", ...(c.system !== undefined ? ["--system", c.system] : [])] : [];
      const r = tokenize(new TextEncoder().encode(text), extra);
      expect(r.code).toBe(0);
      expect((await r.ids).trim().split("\n").map(Number)).toEqual([...c.runtime_ids]);
      if (!chat) expect(r.stdout.toString()).toBe(text.normalize("NFC"));
    }
  });

  test("NFC normalization round-trips decomposed and multilingual text", () => {
    const cases = ["ac\u0327a\u0303o e cafe\u0301", "\u1100\u1161\u11a8", "a\u0315\u0300", "\u212b\u0344",
      "\u1e0a\u0323", "\x00Olá\x00", "🌳\u200d🌱", "中文 123\n\n ", "'S 'RE 'VE 'LL\t\u00a0"];
    for (const text of cases) {
      const r = tokenize(new TextEncoder().encode(text));
      expect(r.code).toBe(0);
      expect(r.stdout.toString()).toBe(text.normalize("NFC"));
    }
  });

  test("invalid UTF-8 is rejected", () => {
    for (const bytes of [[0xc0, 0x80], [0xed, 0xa0, 0x80], [0xf4, 0x90, 0x80, 0x80], [0xe2, 0x82], [0x80]]) {
      expect(tokenize(Uint8Array.from(bytes)).code).not.toBe(0);
    }
  });

  test("answers arithmetic in Portuguese and stops at <|im_end|>", () => {
    const r = runOk([...modelArgs(qwen!), "--chat", "--prompt", "Quanto é 7 mais 5? Responda apenas com o número.",
      "--tokens", 32, "--context", 256]);
    expect(r.stdout.toString().trim()).toBe("12");
    expect(r.stats.generated_tokens).toBeLessThan(32);
    expect(r.stats.tracked_peak_bytes).toBeLessThan(64 * 1048576);
  }, SLOW);

  test("raw logits are identical across threads, batch sizes and kernels", () => {
    const base = [...modelArgs(qwen!), "--chat", "--prompt", "Diga olá.", "--tokens", 4, "--context", 128, "--hash-logits"];
    const hashes = [
      runOk([...base, "--threads", 1, "--batch", 1]),
      runOk([...base, "--threads", 4, "--batch", 16]),
      runOk([...base, "--threads", 2, "--batch", 7, "--scalar"]),
      runOk([...base, "--threads", 3, "--batch", 32], SCALAR_BINARY),
    ].map((r) => r.stats.logits_hash);
    expect(new Set(hashes).size).toBe(1);
  }, SLOW);
});

describe.skipIf(!stories)("TinyStories 15M", () => {
  test("continues a story with the known greedy output", () => {
    const r = runOk([...modelArgs(stories!), "--prompt", "Once upon a time, there was a little robot", "--tokens", 24,
      "--context", 256]);
    expect(r.stdout.toString()).toStartWith(". The robot was very sad because it was all alone.");
  }, SLOW);

  test("prompt round-trips through the score-BPE tokenizer", () => {
    const r = run([...modelArgs(stories!).slice(2), "--prompt", "Lily saw a café, ünïcödé.", "--tokenize-only"]);
    expect(r.code).toBe(0);
    expect(r.stdout.toString()).toBe("Lily saw a café, ünïcödé.");
  });
});

describe.skipIf(!qwen)("interactive chat", () => {
  test("keeps conversation history across turns and forgets it after /limpar", () => {
    const input = "Meu nome é Ana e eu moro em Recife.\nQual é o meu nome?\n/limpar\nQual é o meu nome?\n/sair\n";
    const r = runWithInput([...modelArgs(qwen!), "--tokens", 48, "--context", 512], input);
    expect(r.code).toBe(0);
    const replies = r.stdout.toString().trim().split("\n");
    expect(replies.length).toBe(3);
    expect(replies[1]).toContain("Ana");
    expect(replies[2]).not.toContain("Ana");
    expect(r.stderr).toContain("(conversa reiniciada)");
  }, SLOW);
});

describe.skipIf(!qwen)("interactive chat input limits", () => {
  test("an overlong line is cut at a character boundary and stays one message", () => {
    const long = "ação ".repeat(120); /* ~840 bytes, the cap at --context 64 is 256 */
    const r = runWithInput([...modelArgs(qwen!), "--tokens", 8, "--context", 64, "--system", "Seja breve."],
      `${long}\n/sair\n`);
    expect(r.code).toBe(0);
    /* Exactly two prompts: the long line and /sair. Before the fix its tail became extra turns. */
    expect(r.stderr.split("\n> ").length - 1).toBe(2);
    expect((r.stderr.match(/mensagem cortada em \d+ bytes/g) ?? []).length).toBe(1);
  }, SLOW);
});
