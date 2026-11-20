/**
 * The machine role (plan, proposal 4a): every case of the adversary's frozen
 * ambiente-casos.json, read through the real environment by build/test-papel
 * (the same code as the chat). A case gives SMOLMOE_ENV (null: unset), the
 * role, the notice and, for an invalid value, what the notice shows.
 */
import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { BINARY, BUILD, ROOT, bundledModel, modelArgs } from "./support/run";

const EXE = process.platform === "win32" ? ".exe" : "";
const PAPEL = join(BUILD, `test-papel${EXE}`);
const CASOS = process.env.SMOLMOE_AMBIENTE_CASOS ?? join(ROOT, "tests", "evals", "manutencao", "ambiente-casos.json");

interface Caso {
  readonly id: string;
  readonly SMOLMOE_ENV: string | null;
  readonly esperado: "dev" | "producao";
  readonly aviso: "nenhum" | "nao_declarado" | "invalido";
  readonly mostra?: string;
}

const SET = join(ROOT, "tests", "evals", "manutencao");
const MANIFESTO_AMBIENTE = "54800cb6b61fb62d5334b0032430f3dcb5f5f5a27ea57c0dc295c120b42ac928";
const sha = (path: string) => new Bun.CryptoHasher("sha256").update(readFileSync(path)).digest("hex");

test("the role cases are the table frozen in DECISIONS", () => {
  const manifesto = join(SET, "MANIFEST-ambiente.sha256");
  expect(sha(manifesto)).toBe(MANIFESTO_AMBIENTE);
  expect(readFileSync(join(ROOT, "docs", "DECISIONS.md"), "utf8")).toContain(MANIFESTO_AMBIENTE);
  for (const linha of readFileSync(manifesto, "utf8").trim().split(/\r?\n/)) {
    const [hash, nome] = linha.split(/\s+\*?/);
    expect(sha(join(SET, nome!))).toBe(hash!);
  }
});

describe.skipIf(!existsSync(PAPEL) || !existsSync(CASOS))("SMOLMOE_ENV, case by case", () => {
  const casos = existsSync(CASOS) ? (JSON.parse(readFileSync(CASOS, "utf8")) as { casos: Caso[] }).casos : [];

  test.each(casos.map((c) => [c.id, c] as const))("%s", (_id, c) => {
    const env: Record<string, string> = { ...process.env } as Record<string, string>;
    delete env.SMOLMOE_ENV;
    if (c.SMOLMOE_ENV !== null) env.SMOLMOE_ENV = c.SMOLMOE_ENV;
    const r = Bun.spawnSync([PAPEL], { env, stdout: "pipe", stderr: "pipe" });
    expect(r.exitCode).toBe(0);
    const want = `${c.esperado} ${c.aviso}${c.aviso === "invalido" ? ` ${c.mostra}` : ""}`;
    expect(r.stdout.toString().replace(/\r?\n$/, "")).toBe(want);
  });
});

describe.skipIf(!existsSync(PAPEL))("byte sequences and lengths an ordinary case cannot carry", () => {
  const hex = (bytes: string) => Bun.spawnSync([PAPEL, "--hex", bytes], { stdout: "pipe" }).stdout.toString().trim();

  test("UTF-8 that is not well formed is escaped byte by byte, so a disguised ESC never reaches the terminal", () => {
    expect(hex("E0809B")).toBe("producao invalido \\xe0\\x80\\x9b"); /* ESC as an overlong form */
    expect(hex("C19B")).toBe("producao invalido \\xc1\\x9b");
    expect(hex("EDA080")).toBe("producao invalido \\xed\\xa0\\x80"); /* a surrogate */
    expect(hex("F4908080")).toBe("producao invalido \\xf4\\x90\\x80\\x80"); /* above U+10FFFF */
    expect(hex("C3A7")).toBe("producao invalido ç");
    expect(hex("F09F9880")).toBe("producao invalido 😀");
  });

  test("a value longer than what is read is invalid, even when its start is 'dev' and the rest blanks (R9-2)", () => {
    const env: Record<string, string> = { ...process.env, SMOLMOE_ENV: `dev${" ".repeat(2045)}x` } as Record<string, string>;
    const out = Bun.spawnSync([PAPEL], { env, stdout: "pipe" }).stdout.toString().replace(/\r?\n$/, "");
    expect(out.startsWith("producao invalido dev")).toBe(true);
    expect(out.endsWith("…")).toBe(true);
    /* Just under the limit, the blanks are trimmed and it is dev. */
    const curto: Record<string, string> = { ...process.env, SMOLMOE_ENV: `dev${" ".repeat(2040)}` } as Record<string, string>;
    expect(Bun.spawnSync([PAPEL], { env: curto, stdout: "pipe" }).stdout.toString().trim()).toBe("dev nenhum");
  });

  test("a value of 5000 characters is read whole from the environment and shown cut at 64", () => {
    const env: Record<string, string> = { ...process.env, SMOLMOE_ENV: "x".repeat(5000) } as Record<string, string>;
    const r = Bun.spawnSync([PAPEL], { env, stdout: "pipe" });
    expect(r.stdout.toString().trim()).toBe(`producao invalido ${"x".repeat(64)}…`);
  });
});

const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);

describe.skipIf(!qwen || !existsSync(BINARY))("the chat shows the role", () => {
  /* Piped, so without colors: the banner is the text [PRODUÇÃO]. */
  const abrir = (valor: string | null) => {
    const env: Record<string, string> = { ...process.env } as Record<string, string>;
    delete env.SMOLMOE_ENV;
    delete env.NO_COLOR;
    if (valor !== null) env.SMOLMOE_ENV = valor;
    const r = Bun.spawnSync([BINARY, ...modelArgs(qwen!), "--no-tools", "--context", "256"],
      { env, stdin: new TextEncoder().encode("/sair\n"), stdout: "pipe", stderr: "pipe" });
    expect(r.exitCode).toBe(0);
    return r.stderr.toString();
  };

  test("production: the [PRODUÇÃO] banner without colors, and every prompt starts with [PRODUÇÃO]", () => {
    const err = abrir("production");
    expect(err).toContain("[PRODUÇÃO] Esta máquina é de produção (SMOLMOE_ENV=production)");
    expect(err).toContain("[PRODUÇÃO] > ");
    expect(err).not.toContain("\u001b[");
  }, SLOW);

  test("unset or invalid: the conservative mode, each with its own notice; dev: only the [dev] prompt", () => {
    const ausente = abrir(null);
    expect(ausente).toContain("SMOLMOE_ENV não está definida: modo conservador");
    expect(ausente).toContain("[conservador] > ");
    const invalida = abrir("staging");
    expect(invalida).toContain("SMOLMOE_ENV=staging não é válido; use dev, development, prod ou production.");
    expect(invalida).toContain("[conservador] > ");
    const dev = abrir("dev");
    expect(dev).toContain("[dev] > ");
    expect(dev).not.toMatch(/SMOLMOE_ENV|PRODUÇÃO|conservador/);
  }, SLOW);
});
