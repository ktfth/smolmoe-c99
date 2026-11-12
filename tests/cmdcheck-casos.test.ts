/**
 * The synthetic replies of tests/goldens/cmdcheck-casos.json, each checked
 * with --check-commands in its shell where this machine has one: PowerShell
 * (Windows, or pwsh elsewhere) and POSIX sh (Linux and macOS, Git Bash on
 * Windows). Needs no weights.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { ROOT, run, tempDir } from "./support/run";

interface Caso {
  readonly id: string;
  readonly shell: "powershell" | "posix";
  readonly so_windows?: boolean;
  readonly resposta: string;
  readonly deve: readonly string[];
  readonly nao_deve: readonly string[];
  readonly arquivo_que_nao_pode_existir?: string;
}

const casos = (JSON.parse(readFileSync(join(ROOT, "tests", "goldens", "cmdcheck-casos.json"), "utf8")) as {
  casos: Caso[];
}).casos;

const WINDOWS = process.platform === "win32";
/* --shell for each kind, or null where this machine has none. */
const SHELLS: Record<Caso["shell"], string | null> = {
  powershell: WINDOWS ? (Bun.which("pwsh") ? "pwsh" : Bun.which("powershell") ? "powershell" : null)
    : Bun.which("pwsh") ? "pwsh" : null,
  posix: WINDOWS ? (Bun.which("bash") && Bun.which("sh") ? "bash" : null) : Bun.which("sh") ? "sh" : null,
};
/* PowerShell starts in about a second; a cold CI runner can take several. */
const SLOW = 60_000;

const work = tempDir("cmdcheck-casos");
afterAll(() => work.dispose());

describe("casos sintéticos da conferência de comandos", () => {
  test("every case is well formed and ids are unique", () => {
    expect(new Set(casos.map((c) => c.id)).size).toBe(casos.length);
    for (const c of casos) {
      expect(["powershell", "posix"]).toContain(c.shell);
      for (const re of [...c.deve, ...c.nao_deve]) expect(() => new RegExp(re)).not.toThrow();
    }
    expect(casos.filter((c) => c.shell === "posix").length).toBeGreaterThanOrEqual(10);
    expect(casos.filter((c) => c.shell === "powershell").length).toBeGreaterThanOrEqual(10);
  });

  for (const c of casos) {
    const shell = SHELLS[c.shell];
    const skip = shell === null || (c.so_windows === true && !WINDOWS);
    test.skipIf(skip)(`${c.id} (${c.shell})`, () => {
      const file = join(work.path, `${c.id}.md`);
      writeFileSync(file, c.resposta);
      const r = run(["--check-commands", file, "--shell", shell!]);
      const out = r.stdout.toString();
      expect(r.code).toBe(0);
      for (const re of c.deve) expect(out).toMatch(new RegExp(re));
      for (const re of c.nao_deve) expect(out).not.toMatch(new RegExp(re));
      if (c.arquivo_que_nao_pode_existir) {
        expect(existsSync(join(work.path, c.arquivo_que_nao_pode_existir))).toBe(false);
        expect(existsSync(join(ROOT, c.arquivo_que_nao_pode_existir))).toBe(false);
      }
    }, SLOW);
  }
});
