/**
 * The command book (scripts/cmdcheck/livro.json) and its search
 * (scripts/cmdcheck/livro.ts): the book's shape, what the search finds for
 * everyday phrasing and for questions with nothing to do with maintenance,
 * and that the command check finds nothing invented in the book's examples
 * (PowerShell on Windows, sh where there is one). Needs no weights.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { writeFileSync } from "node:fs";
import { join } from "node:path";

import { buscar, carregar, palavras, prefixo, type Shell, stem } from "../scripts/cmdcheck/livro";
import { run, tempDir } from "./support/run";

const livro = carregar();
const WINDOWS = process.platform === "win32";
const SLOW = 120_000;

const work = tempDir("livro");
afterAll(() => work.dispose());

describe("livro de comandos", () => {
  test("every task is well formed and ids are unique", () => {
    expect(new Set(livro.tarefas.map((t) => t.id)).size).toBe(livro.tarefas.length);
    expect(livro.tarefas.length).toBeGreaterThanOrEqual(50);
    for (const t of livro.tarefas) {
      expect(t.frases.length).toBeGreaterThanOrEqual(3);
      expect(t.powershell.length + t.posix.length).toBeGreaterThan(0);
      /* Every example says what it does, after "  # ". */
      for (const line of [...t.powershell, ...t.posix]) expect(line).toMatch(/\S {2}# \S/);
    }
  });

  test("the stem joins the forms of a word and keeps different words apart", () => {
    expect(stem("reiniciar")).toBe(stem("reinicie"));
    expect(stem("ocupam")).toBe(stem("ocupa"));
    expect(stem("subir")).toBe(stem("subo"));
    expect(stem("servicos")).toBe(stem("servico"));
    expect(stem("servico")).not.toBe(stem("servidor"));
    expect(palavras("Como vejo, no PowerShell, o espaço do disco?")).toEqual([stem("espaco"), stem("disco")]);
  });

  const casos: readonly [string, Shell, string][] = [
    ["meu HD tá lotado, como descubro o que ocupa espaço?", "posix", "pasta-grande"],
    ["o nginx caiu, como subo ele de novo?", "posix", "servico-reiniciar"],
    ["tem como saber qual app tá prendendo a porta 3000?", "powershell", "porta-em-uso"],
    ["esqueci a senha do usuário joao, como redefino?", "posix", "senha-usuario"],
    ["quero que o script rode toda segunda às 8h", "posix", "tarefas-agendadas"],
    ["o pc tá muito lento", "powershell", "cpu-processos"],
    ["como removo o pacote apache2 e as configurações?", "posix", "remover-programa"],
    ["preciso criar um zip da pasta Documentos", "powershell", "compactar"],
    ["como sei se o firewall está ligado?", "powershell", "firewall"],
  ];
  for (const [pergunta, shell, id] of casos) {
    test(`acha ${id}: ${pergunta}`, () => {
      expect(buscar(livro, pergunta, shell).map((a) => a.tarefa.id)).toContain(id);
    });
  }

  for (const pergunta of ["Qual a capital da França?", "Me conta uma piada", "Como faço um bolo de cenoura?",
    "Escreva um poema sobre o mar", "Quanto é 2 mais 2?", "Oi, tudo bem?"]) {
    test(`não acha nada: ${pergunta}`, () => {
      expect(buscar(livro, pergunta, "posix")).toEqual([]);
      expect(buscar(livro, pergunta, "powershell")).toEqual([]);
      expect(prefixo([], "posix")).toBe("");
    });
  }

  test("at most two tasks, and only the shell's commands go in the prompt", () => {
    const achados = buscar(livro, "Como vejo o espaço livre de cada disco?", "posix");
    expect(achados.length).toBeGreaterThan(0);
    expect(achados.length).toBeLessThanOrEqual(2);
    const p = prefixo(achados, "posix");
    expect(p).toContain("```bash\ndf -h");
    expect(p).not.toContain("Get-Volume");
  });

  /* The book's own examples through the command check, a few tasks at a time
   * (one PowerShell answer holds a reply, not the whole book). */
  const grupos = (shell: Shell): string[][] => {
    const out: string[][] = [];
    const tarefas = livro.tarefas.filter((t) => t[shell].length);
    for (let i = 0; i < tarefas.length; i += 8) out.push(tarefas.slice(i, i + 8).flatMap((t) => t[shell]));
    return out;
  };
  const conferir = (shell: Shell, lines: readonly string[], i: number, flag: string): string => {
    const file = join(work.path, `${shell}-${i}.md`);
    writeFileSync(file, `\`\`\`${shell === "powershell" ? "powershell" : "bash"}\n${lines.join("\n")}\n\`\`\`\n`);
    const r = run(["--check-commands", file, "--shell", flag]);
    expect(r.code).toBe(0);
    return r.stdout.toString();
  };

  const ps = Bun.which("pwsh") ? "pwsh" : Bun.which("powershell") ? "powershell" : null;
  /* The cmdlets are Windows's (Get-NetAdapter, Get-Volume): elsewhere they are missing for real. */
  test.skipIf(!WINDOWS || ps === null)("nothing invented in the PowerShell examples", () => {
    grupos("powershell").forEach((lines, i) => {
      const out = conferir("powershell", lines, i, ps!);
      expect(out).not.toMatch(/não consegui conferir/);
      expect(out).not.toMatch(/não achei aqui|não tem o parâmetro/);
    });
  }, SLOW);

  const sh = WINDOWS ? (Bun.which("bash") && Bun.which("sh") ? "bash" : null) : Bun.which("sh") ? "sh" : null;
  /* A program that is not installed here may be missing; an option or subcommand never is invented. */
  test.skipIf(sh === null)("nothing invented in the bash examples", () => {
    grupos("posix").forEach((lines, i) => {
      const out = conferir("posix", lines, i, sh!);
      expect(out).not.toMatch(/não consegui conferir/);
      expect(out).not.toMatch(/não cita (a opção|o subcomando)/);
    });
  }, SLOW);
});
