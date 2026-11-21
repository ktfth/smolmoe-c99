/**
 * The models converted on request (make qwen-1.5b-weights, gemma-weights,
 * phi-weights), each skipped when its weights are not in models/: the runtime
 * gives the same logits across threads, batch sizes and kernels, and the chat
 * answers coherently. Without these, a runtime change that breaks only one of
 * them reaches the user before any test.
 */
import { describe, expect, test } from "bun:test";

import { BINARY, bundledModel, modelArgs, ROOT, runOk, runWithInput, type BundledModel } from "./support/run";

const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);

const qwen15 = bundledModel("qwen2.5-1.5b-instruct", "tokenizer.qbp");
const gemma = bundledModel("gemma-3-1b-it", "tokenizer.qbp");
const phi = bundledModel("phi-4-mini-instruct", "tokenizer.qbp");

/* The same logits with one thread and one token per batch, with odd thread and
 * batch counts, and (unless skipped: Phi takes minutes) with the scalar kernels. */
function sameLogits(m: BundledModel, scalar: boolean): void {
  const base = [...modelArgs(m), "--prompt", "Escreva um haicai sobre o mar.", "--chat", "--tokens", 8, "--context", 256,
    "--hash-logits"];
  const a = runOk([...base, "--threads", 1, "--batch", 1]).stats.logits_hash;
  expect(runOk([...base, "--threads", 3, "--batch", 7]).stats.logits_hash).toBe(a);
  if (scalar) expect(runOk([...base, "--scalar", "--threads", 2]).stats.logits_hash).toBe(a);
}

function answersCapital(m: BundledModel): void {
  const chat = runWithInput([...modelArgs(m), "--tokens", 40, "--context", 512, "--no-tools", "--no-book"],
    "Qual é a capital do Brasil?\n");
  expect(chat.code).toBe(0);
  expect(chat.stdout.toString()).toContain("Brasília");
}

describe.skipIf(!qwen15)("Qwen2.5-1.5B-Instruct", () => {
  test("logits are identical across threads, batch sizes and the scalar kernels", () => sameLogits(qwen15!, true), SLOW);

  test("answers in Portuguese, and the qwen-1.5b alias finds its config and tokenizer", () => {
    answersCapital(qwen15!);
    /* An alias is a path relative to the working directory, as in `make chat`. */
    const r = runWithInput(["qwen-1.5b", "--chat", "--prompt", "Qual é a capital do Brasil?", "--tokens", 24, "--no-book"],
      "", BINARY, ROOT);
    expect(r.code).toBe(0);
    expect(r.stdout.toString()).toContain("Brasília");
  }, SLOW);

  test("keeps the tools with the command book on (only the 0.5B drops them)", () => {
    const r = runWithInput([...modelArgs(qwen15!), "--tokens", 8], "/sair\n");
    expect(r.code).toBe(0);
    expect(r.stderr).toContain("Ferramentas: data_atual");
    expect(r.stderr).not.toContain("ferramentas desligadas");
  }, SLOW);

  test("does the arithmetic in calcular", () => {
    const r = runWithInput([...modelArgs(qwen15!), "--now", "2026-11-06T10:30", "--tokens", 48, "--no-book"],
      "Quanto é 15% de 230?\n");
    expect(r.code).toBe(0);
    expect(r.stderr).toContain("[ferramenta] calcular");
    expect(r.stdout.toString()).toContain("34,5");
  }, SLOW);
});

describe.skipIf(!gemma)("Gemma 3 1B", () => {
  test("logits are identical across threads, batch sizes and the scalar kernels", () => sameLogits(gemma!, true), SLOW);
  test("answers in Portuguese", () => answersCapital(gemma!), SLOW);
});

describe.skipIf(!phi)("Phi-4-mini", () => {
  test("logits are identical across threads and batch sizes", () => sameLogits(phi!, false), SLOW);
  test("answers in Portuguese", () => answersCapital(phi!), SLOW);
});
