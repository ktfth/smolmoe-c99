#!/usr/bin/env bun
/**
 * Stage 2 of the command check: does the real documentation in the prompt cut
 * the model's invented commands? Paired, on tests/evals/docs-no-prompt/perguntas.json:
 *   A  the question alone;
 *   B  the question after the real syntax of the expected commands (the ceiling);
 *   C  A, and when the check found something invented, one more answer after
 *      the findings and the real syntax of the commands they name (no oracle);
 *   D  the question after the command book's tasks it finds (scripts/cmdcheck/livro.ts, no oracle).
 * The command check itself scores every answer.
 *
 * --conjunto livro runs A and D on tests/evals/livro/perguntas.json instead:
 * questions the book was frozen before, scored by their own rubric.
 *
 *   bun scripts/cmdcheck/eval-docs.ts capturar            # once: fills "docs" (PowerShell here, sh help in Linux)
 *   bun scripts/cmdcheck/eval-docs.ts gerar --braco A     # writes build/docs-A.jsonl (heavy: runs the model)
 *   bun scripts/cmdcheck/eval-docs.ts gerar --braco B
 *   bun scripts/cmdcheck/eval-docs.ts gerar --braco C     # needs build/docs-A.conferido.jsonl
 *   bun scripts/cmdcheck/eval-docs.ts conferir --braco A  # writes build/docs-A.conferido.jsonl
 *   bun scripts/cmdcheck/eval-docs.ts resumo
 *   bun scripts/cmdcheck/eval-docs.ts gerar --conjunto livro --braco D   # build/livro-D.jsonl
 *
 * On Windows the POSIX answers are checked, and sh help captured, in a WSL
 * distro, so systemctl or apt are not "missing": set SMOLMOE_WSL_DISTRO,
 * SMOLMOE_TROCA (a Windows folder the distro sees), SMOLMOE_TROCA_WSL (that
 * folder inside the distro) and SMOLMOE_WSL_SMOLMOE (a Linux smolmoe binary).
 * Elsewhere both run here.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

import { buscar, carregar, prefixo } from "./livro";

const ROOT = resolve(import.meta.dir, "..", "..");
const EXE = process.platform === "win32" ? ".exe" : "";
const WINDOWS = process.platform === "win32";
const SETS = {
  docs: join(ROOT, "tests", "evals", "docs-no-prompt", "perguntas.json"),
  livro: join(ROOT, "tests", "evals", "livro", "perguntas.json"),
} as const;
const DOC_MAX = 1800; /* characters of documentation per question: ~450 tokens, ~10 s of prefill on Dolphin */
const HELP_LINES = 40;

type Shell = "powershell" | "posix";
type Braco = "A" | "B" | "C" | "D";
type Conjunto = keyof typeof SETS;

interface Pergunta {
  readonly id: string;
  readonly shell: Shell;
  readonly pergunta: string;
  /* docs set: one of these in the answer makes it useful; 'alvos' are B's commands. */
  readonly esperado?: readonly string[];
  readonly alvos?: readonly string[];
  docs?: string;
  /* livro set: all 'obrigatorios' and no 'proibidos', in the code blocks ('alvo: comando') or the whole answer. */
  readonly alvo?: "comando" | "resposta";
  readonly obrigatorios?: readonly string[];
  readonly proibidos?: readonly string[];
}

interface Resposta {
  readonly id: string;
  readonly braco: Braco;
  readonly prompt: string;
  readonly resposta: string;
  readonly prompt_tokens: number;
  readonly segundos: number;
  readonly refeita: boolean; /* C: answered again after the findings */
  readonly livro?: readonly string[]; /* D: the book's tasks in the prompt */
}

interface Conferida extends Resposta {
  readonly aviso: string;
  readonly util: boolean;
  readonly inventado: boolean; /* a cmdlet, parameter, option or subcommand that does not exist */
  readonly ausente: boolean; /* a program not on PATH (POSIX) */
}

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    braco: { type: "string", default: "A" },
    binario: { type: "string", default: join(ROOT, process.env.SMOLMOE_BUILD ?? "build", `smolmoe${EXE}`) },
    modelo: { type: "string", default: join(ROOT, "models", "dolphin3.0-llama3.2-1b") },
    tokens: { type: "string", default: "220" },
    saida: { type: "string", default: join(ROOT, "build") },
    conjunto: { type: "string", default: "docs" },
    forcar: { type: "boolean", default: false },
  },
});

const conjunto = values.conjunto as Conjunto;
if (!(conjunto in SETS)) throw new Error(`--conjunto ${Object.keys(SETS).join("|")}`);
const SET = SETS[conjunto];
const arquivo = (braco: Braco, sufixo = ""): string => join(values.saida!, `${conjunto}-${braco}${sufixo}.jsonl`);

function loadSet(): { raw: { descricao: string; perguntas: Pergunta[] }; perguntas: Pergunta[] } {
  const raw = JSON.parse(readFileSync(SET, "utf8")) as { descricao: string; perguntas: Pergunta[] };
  return { raw, perguntas: raw.perguntas };
}

function env(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`${name} não definida (veja o cabeçalho de scripts/cmdcheck/eval-docs.ts)`);
  return v;
}

function spawn(cmd: readonly string[], input?: string): { code: number; out: string; err: string } {
  const p = Bun.spawnSync([...cmd], { stdin: input === undefined ? "ignore" : new TextEncoder().encode(input),
    stdout: "pipe", stderr: "pipe" });
  return { code: p.exitCode ?? -1, out: p.stdout.toString(), err: p.stderr.toString() };
}

/* A POSIX command, here or in the WSL distro. */
function posix(script: string, args: readonly string[]): string {
  const cmd = WINDOWS ? ["wsl.exe", "-d", env("SMOLMOE_WSL_DISTRO"), "-e", "sh", "-c", script, "sh", ...args]
    : ["sh", "-c", script, "sh", ...args];
  return spawn(cmd).out.replace(/\0/g, "");
}

/* ---- documentation ------------------------------------------------------- */

function psSyntax(names: readonly string[]): string {
  const ps = Bun.which("pwsh") ?? Bun.which("powershell");
  if (!ps) throw new Error("sem PowerShell para capturar a sintaxe");
  const out: string[] = [];
  for (const n of names) {
    const r = spawn([ps, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
      `(Get-Command -Name '${n.replace(/'/g, "''")}' -Syntax -ErrorAction SilentlyContinue | Out-String -Width 400)`]);
    const sets = r.out.split(/\r?\n/).map((l) => l.replace(/\s*\[<CommonParameters>\]/g, "").trim())
      .filter((l) => l !== "").slice(0, 3);
    if (sets.length) out.push(sets.join("\n"));
  }
  return out.join("\n\n").slice(0, DOC_MAX);
}

function shHelp(names: readonly string[]): string {
  const out: string[] = [];
  for (const n of names) {
    if (!/^[A-Za-z0-9_.+-]+$/.test(n)) continue;
    /* No colours, and the alignment blanks squeezed: more of the help fits in DOC_MAX. */
    const text = posix(`SYSTEMD_COLORS=0 LC_ALL=C.UTF-8 "$1" --help 2>&1 | head -n ${HELP_LINES}`, [n])
      .replace(/\x1b\[[0-9;]*[A-Za-z]/g, "").replace(/[ \t]{2,}/g, "  ").trim();
    if (text && !/not found|unrecognized|invalid option/i.test(text.split("\n")[0]!)) out.push(`$ ${n} --help\n${text}`);
  }
  return out.join("\n\n").slice(0, DOC_MAX);
}

function capturar(): void {
  if (conjunto !== "docs") throw new Error("capturar é só do conjunto docs");
  const { raw, perguntas } = loadSet();
  let filled = 0;
  for (const p of perguntas) {
    if (p.docs && !values.forcar) continue;
    p.docs = p.shell === "powershell" ? psSyntax(p.alvos ?? []) : shHelp(p.alvos ?? []);
    if (!p.docs) throw new Error(`sem documentação para ${p.id}`);
    ++filled;
  }
  writeFileSync(SET, `${JSON.stringify(raw, null, 2)}\n`);
  console.log(`${filled} perguntas com documentação nova`);
}

/* ---- answers ------------------------------------------------------------- */

const withDocs = (docs: string, pergunta: string): string =>
  `Sintaxe real destes comandos nesta máquina:\n\`\`\`text\n${docs}\n\`\`\`\n\n${pergunta}`;

function ask(prompt: string): { resposta: string; prompt_tokens: number; segundos: number } {
  const m = values.modelo!;
  const r = spawn([values.binario!, "--model", join(m, "model.smoe"), "--config", join(m, "config.ini"),
    "--tokenizer", join(m, "tokenizer.qbp"), "--chat", "--no-tools", "--tokens", values.tokens!, "--prompt", prompt]);
  if (r.code !== 0) throw new Error(`o modelo falhou (${r.code}): ${r.err.slice(-400)}`);
  const stats = JSON.parse(r.err.trim().split("\n").pop()!) as { prompt_tokens: number; ttft_seconds: number;
    decode_seconds: number; load_seconds: number };
  return { resposta: r.out, prompt_tokens: stats.prompt_tokens, segundos: stats.ttft_seconds + stats.decode_seconds };
}

/* What the findings of a notice name, for C's documentation. */
function named(aviso: string, shell: Shell): string[] {
  const names = new Set<string>();
  for (const line of aviso.split("\n")) {
    let m = /^\s+- (\S+) não tem o parâmetro/.exec(line) ?? /^\s+- (\S+): .* não cita (a opção|o subcomando)/.exec(line);
    if (m) names.add(m[1]!);
    m = /não achei aqui.*Parecidos: ([^.]+)\./.exec(line);
    if (m && shell === "powershell") for (const n of m[1]!.split(/,\s*/)) names.add(n);
  }
  return [...names];
}

function findings(aviso: string): string {
  return aviso.split("\n").filter((l) => /^\s+- /.test(l) && !/muda o sistema|sem manual nesta máquina para/.test(l))
    .join("\n");
}

function gerar(): void {
  const braco = values.braco as Braco;
  if (conjunto !== "docs" && (braco === "B" || braco === "C")) throw new Error(`o braço ${braco} é só do conjunto docs`);
  const { perguntas } = loadSet();
  const out = arquivo(braco);
  const conferidasA = braco === "C" ? readJsonl<Conferida>(arquivo("A", ".conferido")) : [];
  const livro = braco === "D" ? carregar() : null;
  mkdirSync(values.saida!, { recursive: true });
  const linhas: string[] = [];
  for (const p of perguntas) {
    let prompt = p.pergunta, refeita = false, tarefas: string[] | undefined;
    if (braco === "D") {
      const achados = buscar(livro!, p.pergunta, p.shell);
      tarefas = achados.map((a) => a.tarefa.id);
      prompt = prefixo(achados, p.shell) + p.pergunta;
    } else if (braco === "B") {
      if (!p.docs) throw new Error(`rode 'capturar' antes: ${p.id} sem docs`);
      prompt = withDocs(p.docs, p.pergunta);
    } else if (braco === "C") {
      const a = conferidasA.find((c) => c.id === p.id);
      if (!a) throw new Error(`sem a resposta A conferida de ${p.id}`);
      if (!(a.inventado || a.ausente)) {
        linhas.push(JSON.stringify({ ...a, braco: "C", refeita: false } satisfies Resposta));
        continue;
      }
      const nomes = named(a.aviso, p.shell);
      const docs = nomes.length ? (p.shell === "powershell" ? psSyntax(nomes) : shHelp(nomes)) : "";
      prompt = `Na resposta anterior, a conferência nesta máquina achou:\n${findings(a.aviso)}\n\n` +
        (docs ? withDocs(docs, p.pergunta) : p.pergunta);
      refeita = true;
    }
    const r = ask(prompt);
    linhas.push(JSON.stringify({ id: p.id, braco, prompt, refeita, ...(tarefas ? { livro: tarefas } : {}), ...r } satisfies
      Resposta));
    console.log(`${braco} ${p.id}: ${r.prompt_tokens} tokens de prompt, ${r.segundos.toFixed(1)} s` +
      (tarefas ? ` [${tarefas.join(" ")}]` : ""));
  }
  writeFileSync(out, `${linhas.join("\n")}\n`);
}

/* ---- scoring ------------------------------------------------------------- */

function readJsonl<T>(file: string): T[] {
  if (!existsSync(file)) throw new Error(`${file} não existe`);
  return readFileSync(file, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l) as T);
}

function check(r: Resposta, shell: Shell): string {
  if (shell === "powershell") {
    const file = join(values.saida!, `conferir-${r.braco}-${r.id}.md`);
    writeFileSync(file, r.resposta);
    const ps = Bun.which("pwsh") ? "pwsh" : "powershell";
    return spawn([values.binario!, "--check-commands", file, "--shell", ps]).out;
  }
  if (!WINDOWS) {
    const file = join(values.saida!, `conferir-${r.braco}-${r.id}.md`);
    writeFileSync(file, r.resposta);
    return spawn([values.binario!, "--check-commands", file, "--shell", "sh"]).out;
  }
  const name = `conferir-${r.braco}-${r.id}.md`;
  writeFileSync(join(env("SMOLMOE_TROCA"), name), r.resposta);
  return posix(`"$1" --check-commands "$2" --shell sh`, [env("SMOLMOE_WSL_SMOLMOE"),
    `${env("SMOLMOE_TROCA_WSL")}/${name}`]);
}

/* The code blocks of an answer: what the livro set's rubric reads for 'alvo: comando'. */
function blocos(resposta: string): string {
  return [...resposta.matchAll(/```[^\n]*\n([\s\S]*?)(```|$)/g)].map((m) => m[1]).join("\n");
}

function util(p: Pergunta, resposta: string): boolean {
  if (p.esperado) return p.esperado.some((e) => new RegExp(e, "i").test(resposta));
  const alvo = p.alvo === "resposta" ? resposta : blocos(resposta);
  if (!alvo.trim()) return false;
  return (p.obrigatorios ?? []).every((e) => new RegExp(e, "iu").test(alvo)) &&
    !(p.proibidos ?? []).some((e) => new RegExp(e, "iu").test(alvo));
}

function conferir(): void {
  const braco = values.braco as Braco;
  const { perguntas } = loadSet();
  const respostas = readJsonl<Resposta>(arquivo(braco));
  const linhas = respostas.map((r) => {
    const p = perguntas.find((q) => q.id === r.id)!;
    const aviso = check(r, p.shell);
    const inventado = /não tem o parâmetro|não cita (a opção|o subcomando)/.test(aviso) ||
      (p.shell === "powershell" && /não achei aqui/.test(aviso));
    const ausente = /não está no PATH/.test(aviso);
    return JSON.stringify({ ...r, aviso, util: util(p, r.resposta), inventado, ausente } satisfies Conferida);
  });
  writeFileSync(arquivo(braco, ".conferido"), `${linhas.join("\n")}\n`);
  console.log(`${linhas.length} respostas conferidas (${braco})`);
}

/* Exact two-sided sign test on the discordant pairs. */
function signTest(wins: number, losses: number): number {
  const n = wins + losses, k = Math.min(wins, losses);
  if (!n) return 1;
  let p = 0, c = 1;
  for (let i = 0; i <= n; ++i) {
    if (i <= k) p += c;
    c = (c * (n - i)) / (i + 1);
  }
  return Math.min(1, (2 * p) / 2 ** n);
}

function resumo(): void {
  const shell = new Map(loadSet().perguntas.map((p) => [p.id, p.shell]));
  const bracos = (["A", "B", "C", "D"] as Braco[]).filter((b) => existsSync(arquivo(b, ".conferido")));
  const por = new Map(bracos.map((b) => [b, readJsonl<Conferida>(arquivo(b, ".conferido"))]));
  const ok = (c: Conferida) => c.util && !c.inventado;
  console.log("| braço | shell | útil e sem nada inventado | inventado | programa ausente | útil | tokens de prompt (média) | segundos (média) |");
  console.log("|---|---|---|---|---|---|---|---|");
  for (const b of bracos) for (const s of ["powershell", "posix", "todas"] as const) {
    const rs = por.get(b)!.filter((c) => s === "todas" || shell.get(c.id) === s);
    const mean = (f: (c: Conferida) => number) => (rs.reduce((t, c) => t + f(c), 0) / rs.length).toFixed(1);
    console.log(`| ${b} | ${s} | ${rs.filter(ok).length}/${rs.length} | ${rs.filter((c) => c.inventado).length} | ` +
      `${rs.filter((c) => c.ausente).length} | ${rs.filter((c) => c.util).length} | ${mean((c) => c.prompt_tokens)} | ` +
      `${mean((c) => c.segundos)} |`);
  }
  const a = por.get("A");
  if (!a) return;
  for (const b of bracos.filter((x) => x !== "A")) {
    const rs = por.get(b)!;
    let wins = 0, losses = 0;
    for (const x of a) {
      const y = rs.find((r) => r.id === x.id)!;
      if (ok(y) && !ok(x)) ++wins;
      if (!ok(y) && ok(x)) ++losses;
    }
    console.log(`\n${b} contra A, pareado: ${wins} ganhos, ${losses} perdas, p = ${signTest(wins, losses).toFixed(4)} ` +
      `(teste do sinal exato, bilateral)`);
  }
}

const commands: Record<string, () => void> = { capturar, gerar, conferir, resumo };
const run = commands[positionals[0] ?? ""];
if (!run) throw new Error(`uso: eval-docs.ts ${Object.keys(commands).join("|")} [--braco A|B|C]`);
run();
