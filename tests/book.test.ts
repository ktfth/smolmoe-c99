/**
 * The command book in the runtime (src/book.c): the generated files are current
 * with scripts/cmdcheck/livro.json and livro.ts, the binary builds the very
 * prompts the eval measured (scripts/cmdcheck/eval-docs.ts, arm D), and the chat
 * names the tasks it used. The prompt checks need only the Dolphin tokenizer;
 * the chat check runs the model (make dolphin-weights), and both are skipped
 * without it.
 */
import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { goldenText, incText } from "../scripts/cmdcheck/gen-book";
import { buscar, carregar, prefixo, type Shell } from "../scripts/cmdcheck/livro";
import { bundledModel, modelArgs, ROOT, run, runWithInput } from "./support/run";

const dolphin = bundledModel("dolphin3.0-llama3.2-1b", "tokenizer.qbp");
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);
const SETS = ["tests/evals/livro/perguntas.json", "tests/evals/livro-2/perguntas.json"];
const SHELL_ARG: Record<Shell, string> = { powershell: "pwsh", posix: "bash" };

interface Pergunta { readonly id: string; readonly shell: Shell; readonly pergunta: string }

function perguntas(): Pergunta[] {
  return SETS.flatMap((s) => (JSON.parse(readFileSync(join(ROOT, s), "utf8")) as { perguntas: Pergunta[] }).perguntas);
}

describe("livro de comandos no runtime", () => {
  test("src/book_data.inc and the search golden are current (bun scripts/cmdcheck/gen-book.ts)", () => {
    expect(readFileSync(join(ROOT, "src", "book_data.inc"), "utf8")).toBe(incText());
    expect(readFileSync(join(ROOT, "tests", "goldens", "livro-busca.tsv"), "utf8")).toBe(goldenText());
  });

  describe.skipIf(!dolphin)("with the Dolphin tokenizer", () => {
    const tok = () => [...modelArgs(dolphin!).slice(2), "--context", 1024, "--tokenize-only", "--chat"];

    test("--chat --book builds the eval's arm D prompt for every question of both held-out sets", () => {
      const livro = carregar();
      for (const p of perguntas()) {
        const achados = buscar(livro, p.pergunta, p.shell);
        const want = run([...tok(), "--prompt", prefixo(achados, p.shell) + p.pergunta]);
        const got = run([...tok(), "--book", "--shell", SHELL_ARG[p.shell], "--prompt", p.pergunta]);
        expect(want.code).toBe(0);
        expect(got.code).toBe(0);
        expect({ id: p.id, prompt: got.stdout.toString() }).toEqual({ id: p.id, prompt: want.stdout.toString() });
        const named = achados.length ? `(livro de comandos: ${achados.map((a) => a.tarefa.id).join(", ")})` : "";
        expect({ id: p.id, notice: got.stderr.includes("(livro de comandos:") ? got.stderr.trim() : "" })
          .toEqual({ id: p.id, notice: named });
      }
    }, SLOW);

    test("without --book a --chat prompt goes as written", () => {
      const q = "Como vejo o espaço livre do disco?";
      const plain = run([...tok(), "--prompt", q]).stdout.toString();
      expect(plain).not.toContain("Comandos reais");
      expect(run([...tok(), "--no-book", "--prompt", q]).stdout.toString()).toBe(plain);
      expect(run([...modelArgs(dolphin!).slice(2), "--tokenize-only", "--book", "--prompt", q]).stdout.toString())
        .not.toContain("Comandos reais");
    });
  });

  describe.skipIf(!dolphin)("chat (Dolphin)", () => {
    const args = () => [...modelArgs(dolphin!), "--tokens", 1, "--context", 1024, "--no-tools", "--no-command-check",
      "--shell", "bash"];

    test("a maintenance question gets the book and the chat names its tasks; --no-book turns it off", () => {
      const q = "Como vejo o espaço livre do disco?\n";
      const on = runWithInput(args(), q);
      expect(on.code).toBe(0);
      expect(on.stderr).toContain("Livro de comandos: perguntas de manutenção levam ao modelo exemplos reais de bash.");
      expect(on.stderr).toMatch(/\(livro de comandos: espaco-disco[,)]/);
      const off = runWithInput([...args(), "--no-book"], q);
      expect(off.code).toBe(0);
      expect(off.stderr).not.toContain("ivro de comandos");
      /* The stats come last, after the chat's "> " prompt. */
      const tokens = (r: { stderr: string }) =>
        (JSON.parse(r.stderr.slice(r.stderr.lastIndexOf('{"'))) as { prompt_tokens: number }).prompt_tokens;
      expect(tokens(on)).toBeGreaterThan(tokens(off) + 20);
    }, SLOW);

    test("an off-topic message goes without the book", () => {
      const r = runWithInput(args(), "Escreva um poema sobre o mar.\n");
      expect(r.code).toBe(0);
      expect(r.stderr).not.toContain("(livro de comandos:");
    }, SLOW);
  });
});
