/**
 * Dolphin 3.0 Llama 3.2 1B (make dolphin-weights), skipped when the converted
 * weights are not in models/: the tokenizer against Hugging Face reference
 * IDs, the chat template, determinism across kernels and batching, and a
 * greedy answer.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import golden from "./goldens/llama3-tokenizer.json";
import { bundledModel, modelArgs, run, runOk, runWithInput, tempDir } from "./support/run";

const dolphin = bundledModel("dolphin3.0-llama3.2-1b", "tokenizer.qbp");
const work = tempDir("llama-model");
afterAll(() => work.dispose());

interface GoldenCase {
  readonly text?: string;
  readonly chatml_prompt?: string;
  readonly system?: string;
  readonly runtime_ids: readonly number[];
}

const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);

describe.skipIf(!dolphin)("Dolphin 3.0 Llama 3.2 1B", () => {
  test("tokenizer matches Hugging Face reference IDs (remapped) on every golden case", () => {
    for (const c of (golden as { cases: GoldenCase[] }).cases) {
      const chat = c.chatml_prompt !== undefined;
      const input = join(work.path, "prompt.txt"), ids = join(work.path, "ids.txt");
      writeFileSync(input, c.text ?? c.chatml_prompt ?? "");
      const r = run([...modelArgs(dolphin!).slice(2), "--prompt-file", input, "--context", 1024, "--tokenize-only",
        "--token-ids", ids, ...(chat ? ["--chat", "--system", c.system ?? ""] : [])]);
      expect(r.code).toBe(0);
      const got = readFileSync(ids, "utf8").trim().split("\n").map(Number);
      expect(got).toEqual([...c.runtime_ids]);
      if (!chat) expect(r.stdout.toString()).toBe(c.text);
    }
  });

  test("logits are identical across threads, batch sizes and the scalar kernels", () => {
    const base = [...modelArgs(dolphin!), "--prompt", "Write a haiku about the sea.", "--chat", "--system", "You are Dolphin.",
      "--tokens", 8, "--context", 256, "--hash-logits"];
    const a = runOk([...base, "--threads", 1, "--batch", 1]).stats.logits_hash;
    expect(runOk([...base, "--threads", 3, "--batch", 7]).stats.logits_hash).toBe(a);
    expect(runOk([...base, "--scalar", "--threads", 2]).stats.logits_hash).toBe(a);
  }, SLOW);

  test("greedy completion and a multi-turn chat answer coherently", () => {
    const r = runOk([...modelArgs(dolphin!), "--prompt", "The capital of France is", "--tokens", 6, "--context", 128]);
    expect(r.stdout.toString()).toContain("Paris");
    const chat = runWithInput([...modelArgs(dolphin!), "--tokens", 40, "--context", 512, "--no-tools"],
      "Qual é a capital do Brasil?\n");
    expect(chat.code).toBe(0);
    expect(chat.stdout.toString()).toContain("Brasília");
  }, SLOW);
});
