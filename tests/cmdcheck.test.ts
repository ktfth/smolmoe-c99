/**
 * The command check against this machine's real PowerShell: --check-commands
 * reads a file as if it were a reply and prints what the chat would print
 * after it. Needs no weights; skipped where there is no PowerShell.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { run, tempDir } from "./support/run";

const work = tempDir("cmdcheck");
afterAll(() => work.dispose());

const hasPowerShell = ["pwsh", "powershell"].some((name) => Bun.which(name) !== null);
/* PowerShell starts in about a second here; a cold CI runner can take several. */
const SLOW = 60_000;

function check(reply: string, extra: readonly string[] = []): { code: number; out: string; err: string } {
  const file = join(work.path, `reply-${Math.random().toString(36).slice(2)}.md`);
  writeFileSync(file, reply);
  const r = run(["--check-commands", file, ...extra]);
  return { code: r.code, out: r.stdout.toString(), err: r.stderr };
}

describe.skipIf(!hasPowerShell)("--check-commands", () => {
  test("flags an invented cmdlet and an invented parameter, with real neighbours", () => {
    const r = check("Use:\n```powershell\nGet-NetAdapterInfo\nGet-ChildItem -Recursive C:\\Temp\n```\n");
    expect(r.code).toBe(0);
    expect(r.out).toContain("Conferência no PowerShell");
    expect(r.out).toMatch(/Get-NetAdapterInfo: não achei aqui \(pode ser de um módulo que esta máquina não tem\)/);
    expect(r.out).toMatch(/Get-ChildItem não tem o parâmetro -Recursive\. Parecidos: [^\n]*-Recurse/);
    expect(r.out).toContain("não diz se o comando faz o que você pediu");
  }, SLOW);

  test("a correct read-only command passes, and the notice still says what was not checked", () => {
    const r = check("Rode `Get-ChildItem -Recurse -File C:\\Temp` e depois `Get-Process -Name pwsh`.");
    expect(r.code).toBe(0);
    expect(r.out).toContain("os nomes e parâmetros existem");
    expect(r.out).toContain("Isso não diz se o comando faz o que você pediu");
    expect(r.out).not.toContain("não tem o parâmetro");
  }, SLOW);

  test("a real command that changes the system is flagged even when it is spelled right", () => {
    const r = check("```powershell\nRemove-Item -Recurse C:\\Temp\\velho\nStop-Process -Id 4\n```");
    expect(r.out).toContain("Remove-Item muda o sistema: confira o alvo e teste antes com -WhatIf.");
    expect(r.out).toContain("Stop-Process muda o sistema");
    expect(r.out).not.toContain("os nomes e parâmetros existem");
  }, SLOW);

  test("cmdlets that take -WhatIf but only touch the PowerShell session are not system changes", () => {
    const r = check("```powershell\nGet-ChildItem | ForEach-Object { $_.Length }\nNew-Variable -Name x -Value 1\n" +
      "Set-Alias -Name ll -Value Get-ChildItem\nConvert-Path .\n```");
    expect(r.out).toContain("os nomes e parâmetros existem");
    expect(r.out).not.toContain("muda o sistema");
  }, SLOW);

  test("a real hallucinated parameter from Dolphin is caught", () => {
    const r = check("```powershell\nGet-ChildItem -Path C:\\Temp | ForEach-Object { (Get-ItemProperty -Path " +
      "$_.FullName -Property Length).Count }\n```");
    expect(r.out).toContain("Get-ItemProperty não tem o parâmetro -Property.");
    expect(r.out).not.toContain("ForEach-Object");
  }, SLOW);

  test("aliases, abbreviated parameters and common parameters are accepted", () => {
    const r = check("```\nls -Rec C:\\Temp\ngci -Force -ErrorAction SilentlyContinue\nGet-Item x -Verbose\n```");
    expect(r.out).toContain("os nomes e parâmetros existem");
  }, SLOW);

  /* Windows only: elsewhere PowerShell runs /usr/bin/ls, which takes -la. */
  test.skipIf(process.platform !== "win32")("PowerShell's view of a Unix flag is reported, since the command would fail here", () => {
    const r = check("```\nls -la\n```");
    expect(r.out).toContain("Get-ChildItem não tem o parâmetro -la.");
  }, SLOW);

  test("a block for another shell is named, never reported as missing", () => {
    const r = check("No Debian:\n```bash\nsudo apt update && sudo apt upgrade -y\n```\n");
    expect(r.code).toBe(0);
    expect(r.out).toContain("bloco bash não conferido");
    expect(r.out).not.toContain("não achei");
  }, SLOW);

  test("prose without backticks is read; ordinary dashed words are not", () => {
    const r = check("Use Get-Proccess para listar. O Wi-Fi e o e-mail ficam como estão.");
    expect(r.out).toContain("Get-Proccess: não achei aqui");
    expect(r.out).not.toContain("Wi-Fi");
  }, SLOW);

  test("nothing to check prints nothing", () => {
    const r = check("A memória RAM guarda os dados dos programas abertos.");
    expect(r.code).toBe(0);
    expect(r.out).toBe("");
  });

  test("the reply text is only parsed: a command in it never runs", () => {
    const marker = join(work.path, "nao-devia-existir.txt");
    const r = check("```powershell\nNew-Item -ItemType File -Path '" + marker + "'\nSet-Content '" + marker +
      "' x\n```");
    expect(r.code).toBe(0);
    expect(existsSync(marker)).toBe(false);
    expect(r.out).toContain("New-Item muda o sistema");
  }, SLOW);
});

describe("--check-commands errors", () => {
  test("a missing file is an error", () => {
    const r = run(["--check-commands", join(work.path, "nao-existe.md")]);
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("--check-commands");
  });

  test("--shell picks the PowerShell, and a non-PowerShell is refused", () => {
    const file = join(work.path, "x.md");
    writeFileSync(file, "`Get-Item x`");
    const r = run(["--check-commands", file, "--shell", process.platform === "win32" ? "cmd" : "sh"]);
    expect(r.code).not.toBe(0);
    expect(r.stderr).toContain("PowerShell");
  });
});
