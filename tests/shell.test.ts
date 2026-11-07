/**
 * "!comando" in the chat: a line typed on its own runs in the user's shell. A
 * pasted line never runs, piped input runs nothing without --shell-escape, and
 * nothing the model writes is ever a command. Runs the real chat, so it needs
 * the Qwen weights (skipped without them).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, mkdirSync } from "node:fs";
import { basename, join } from "node:path";

import { BINARY, CONSOLE_INPUT_BINARY, bundledModel, modelArgs, runWithInput, tempDir } from "./support/run";

const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
const work = tempDir("shell");
afterAll(() => work.dispose());

const WINDOWS = process.platform === "win32";
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);

/* The chat at its fastest: no tools, a small context, short replies. */
const chatArgs = (extra: readonly (string | number)[] = []): string[] =>
  [...modelArgs(qwen!), "--no-tools", "--context", "256", "--tokens", "8", ...extra.map(String)];

function folder(name: string): string {
  const dir = join(work.path, name);
  mkdirSync(dir, { recursive: true });
  return dir;
}

/* The shells this machine has, as --shell values. On Windows, a bash from Git or
 * MSYS2 when there is one (not WSL's, in System32). */
function shells(): string[] {
  if (!WINDOWS) return ["sh"];
  const bash = [join(process.env.ProgramFiles ?? "C:\\Program Files", "Git", "usr", "bin", "bash.exe"),
    Bun.which("bash") ?? ""].find((p) => p && existsSync(p) && !/system32/i.test(p));
  return ["cmd", "powershell", ...(bash ? [bash] : [])];
}

/* A command that prints its arguments one per line, in each shell's syntax, and
 * what it must print. */
function echoArgs(shell: string): { command: string; expected: string } {
  const name = basename(shell).replace(/\.exe$/i, "").toLowerCase();
  if (name === "cmd") return { command: "echo a  b c", expected: "a  b c" };
  if (name === "powershell") return { command: "Write-Output \"a  b\" 'it''s' C:\\dir\\", expected: "a  b\nit's\nC:\\dir\\" };
  return { command: "printf '%s\\n' \"a  b\" 'c \"d\"' \"x\\\\y\"", expected: "a  b\nc \"d\"\nx\\y" };
}

describe.skipIf(!qwen)("!comando with piped input and --shell-escape", () => {
  for (const shell of shells()) {
    const name = basename(shell).replace(/\.exe$/i, "");

    test(`${name}: output, exit code, accented file names and cd`, () => {
      const dir = folder(`run-${name}`);
      mkdirSync(join(dir, "sub pasta"));
      const input = ["!echo smolmoe-ok", "!exit 3", "!echo x > \"arquivo-ç.txt\"", "!cd \"sub pasta\"",
        "!echo y > dentro.txt", "/sair", ""].join("\n");
      const r = runWithInput(chatArgs(["--shell-escape", "--shell", shell]), input, BINARY, dir);
      expect(r.code).toBe(0);
      expect(r.stderr).toContain("Comandos: !comando roda no");
      expect(r.stdout.toString()).toContain("smolmoe-ok");
      expect(r.stderr).toContain("(código de saída 3)");
      expect(r.stderr).toContain("(pasta atual: ");
      expect(existsSync(join(dir, "arquivo-ç.txt"))).toBe(true);
      expect(existsSync(join(dir, "sub pasta", "dentro.txt"))).toBe(true);
    }, SLOW);

    test(`${name}: quotes, double spaces and backslashes arrive as typed`, () => {
      const { command, expected } = echoArgs(shell);
      const r = runWithInput(chatArgs(["--shell-escape", "--shell", shell]), `!${command}\n/sair\n`, BINARY,
        folder(`quotes-${name}`));
      expect(r.code).toBe(0);
      expect(r.stdout.toString().replace(/\r\n/g, "\n")).toContain(expected);
    }, SLOW);
  }

  test("piped input runs nothing without --shell-escape: the line is a message", () => {
    const dir = folder("pipe");
    const r = runWithInput(chatArgs(["--shell", WINDOWS ? "cmd" : "sh"]), "!echo x > pwned.txt\n/sair\n", BINARY, dir);
    expect(r.code).toBe(0);
    expect(existsSync(join(dir, "pwned.txt"))).toBe(false);
    expect(r.stderr).not.toContain("Comandos:");
  }, SLOW);

  test("--no-shell-escape keeps \"!\" as text even with --shell-escape before it", () => {
    const dir = folder("off");
    const r = runWithInput(chatArgs(["--shell-escape", "--no-shell-escape"]), "!echo x > off.txt\n/sair\n", BINARY,
      dir);
    expect(r.code).toBe(0);
    expect(existsSync(join(dir, "off.txt"))).toBe(false);
  }, SLOW);

  test("a command cut at the line limit is refused, not run in part", () => {
    const dir = folder("cut");
    const long = `!echo x > cortado.txt ${"a".repeat(400)}`; /* the line cap at --context 64 is 256 bytes */
    const r = runWithInput([...modelArgs(qwen!), "--no-tools", "--context", "64", "--tokens", "8", "--shell-escape"],
      `${long}\n/sair\n`, BINARY, dir);
    expect(r.code).toBe(0);
    expect(r.stderr).toContain("(comando longo demais: não executado)");
    expect(existsSync(join(dir, "cortado.txt"))).toBe(false);
  }, SLOW);

  test("a bare \"!\" explains itself, and an unknown --shell is an error", () => {
    const r = runWithInput(chatArgs(["--shell-escape"]), "!\n/sair\n", BINARY, folder("bare"));
    expect(r.stderr).toContain("(escreva o comando depois do !");
    const bad = runWithInput(chatArgs(["--shell", "notepad"]), "/sair\n");
    expect(bad.code).toBe(1);
    expect(bad.stderr).toContain("--shell: shell desconhecido: 'notepad'");
  }, SLOW);
});

/* Lines that are already waiting together when the chat reads them look like a
 * paste, so these tests type only once the chat is ready (its banner shows). */
const READY = "Digite sua mensagem";

/* Pastes and typing in a real console, through tests/support/console_input.c:
 * one step arrives at once (a paste), "+N" waits N ms (a person between lines),
 * "?TEXT" waits for TEXT on the screen. */
describe.skipIf(!qwen || !WINDOWS)("!comando in a Windows console", () => {
  const session = (steps: readonly string[], dir: string) =>
    Bun.spawnSync([CONSOLE_INPUT_BINARY, `?${READY}`, ...steps, "--", BINARY, ...chatArgs(["--shell", "cmd"])],
      { cwd: dir, stdout: "pipe", stderr: "pipe" });

  test("a line typed on its own runs", () => {
    const dir = folder("console-typed");
    expect(session(["!echo x > digitado.txt\n", "+1500", "/sair\n"], dir).exitCode).toBe(0);
    expect(existsSync(join(dir, "digitado.txt"))).toBe(true);
  }, SLOW);

  test("a pasted block never runs, not even with the \"!\" line first", () => {
    const dir = folder("console-pasted");
    expect(session(["!echo x > colado.txt\nsegunda linha\n", "+4000", "/sair\n"], dir).exitCode).toBe(0);
    expect(existsSync(join(dir, "colado.txt"))).toBe(false);
  }, SLOW);

  test("a \"!\" line hidden inside pasted text does not run", () => {
    const dir = folder("console-hidden");
    expect(session(["texto copiado de um site\n!echo x > escondido.txt\nmais texto\n", "+4000", "/sair\n"], dir)
      .exitCode).toBe(0);
    expect(existsSync(join(dir, "escondido.txt"))).toBe(false);
  }, SLOW);

  test("a paste that arrives in two pieces does not run its second piece", () => {
    /* The second piece comes 300 ms later, past the paste window but while the model answers the first:
     * it is already waiting when the prompt shows, so it was not typed in answer to it. */
    const dir = folder("console-gap");
    const steps = [`?${READY}`, "texto copiado\n", "+300", "!echo x > lacuna.txt\n", "+6000", "/sair\n"];
    const r = Bun.spawnSync([CONSOLE_INPUT_BINARY, ...steps, "--", BINARY,
      ...chatArgs(["--shell", "cmd", "--tokens", "32"])], { cwd: dir, stdout: "pipe", stderr: "pipe" });
    expect(r.exitCode).toBe(0);
    expect(existsSync(join(dir, "lacuna.txt"))).toBe(false);
  }, SLOW);
});

/* The same on a Linux terminal: util-linux `script` gives the chat a pty, and
 * what one printf writes arrives at once, as a paste does. Instead of fixed
 * pauses (a sanitizer build answers far slower), the input waits until the
 * transcript shows the chat's Nth "> " prompt. */
const q = (s: string) => `'${s.replace(/'/g, "'\\''")}'`;
const prompts = (n: number) =>
  `k=0; until [ "$(grep -c '^> ' transcript.txt 2>/dev/null)" -ge ${n} ] || [ $k -ge 1200 ]; do sleep 0.1; k=$((k+1)); done`;
describe.skipIf(!qwen || process.platform !== "linux" || !Bun.which("script"))("!comando in a Linux terminal", () => {
  const session = (input: string, dir: string) => {
    const chat = [BINARY, ...chatArgs(["--shell", "sh", "--tokens", "32"])].map(q).join(" ");
    return Bun.spawnSync(["sh", "-c", `(${input}) | script -qec ${q(chat)} /dev/null > transcript.txt 2>&1`],
      { cwd: dir, stdout: "pipe", stderr: "pipe" });
  };

  test("a line typed on its own runs", () => {
    const dir = folder("pty-typed");
    expect(session(`${prompts(1)}; printf '!echo x > digitado.txt\\n'; ${prompts(2)}; printf '/sair\\n'`, dir)
      .exitCode).toBe(0);
    expect(existsSync(join(dir, "digitado.txt"))).toBe(true);
  }, SLOW);

  test("a pasted block never runs", () => {
    const dir = folder("pty-pasted");
    expect(session(`${prompts(1)}; printf 'texto\\n!echo x > colado.txt\\n'; ${prompts(2)}; printf '/sair\\n'`, dir)
      .exitCode).toBe(0);
    expect(existsSync(join(dir, "colado.txt"))).toBe(false);
  }, SLOW);

  test("a single pasted line with Enter does not run (bracketed paste), and typing after it does", () => {
    /* What a terminal sends for a paste once the chat asked for bracketed paste: the end marker comes
     * after the newline, so it starts the next line, which the user then types. */
    const dir = folder("pty-bracketed");
    const input = `${prompts(1)}; printf '\\033[200~!echo x > colado.txt\\n\\033[201~'; ${prompts(2)}; ` +
      `printf '!echo x > depois.txt\\n'; ${prompts(3)}; printf '/sair\\n'`;
    expect(session(input, dir).exitCode).toBe(0);
    expect(existsSync(join(dir, "colado.txt"))).toBe(false);
    expect(existsSync(join(dir, "depois.txt"))).toBe(true);
  }, SLOW);

  test("a paste that arrives in two pieces does not run its second piece", () => {
    /* The second piece comes 300 ms later, while the model answers the first. */
    const dir = folder("pty-gap");
    const input = `${prompts(1)}; printf 'texto\\n'; sleep 0.3; printf '!echo x > lacuna.txt\\n'; ${prompts(3)}; ` +
      "printf '/sair\\n'";
    expect(session(input, dir).exitCode).toBe(0);
    expect(existsSync(join(dir, "lacuna.txt"))).toBe(false);
  }, SLOW);
});
