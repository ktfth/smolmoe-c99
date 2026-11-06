#!/usr/bin/env bun
/**
 * Measures chat function calling: for each question, a fresh chat with a fixed
 * clock; checks which tool ran (stderr) and whether the answer (stdout) holds
 * the expected fact. Questions that need no tool must not call one.
 *
 * usage: bun scripts/eval-tools.ts [--set dev|holdout|holdout2] [--session] [--verbose] [--model-dir DIR] [--binary PATH]
 */
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const ROOT = resolve(import.meta.dir, "..");
const EXE = process.platform === "win32" ? ".exe" : "";
const NOW = "2026-11-06T10:30"; /* a Friday; Christmas is 49 days away */

interface Case {
  readonly question: string;
  readonly tool: "data_atual" | "dias_ate" | null;
  readonly answer?: string; /* expected substring, case-insensitive */
}

/* A future date described as past ("49 dias atrás") is a wrong answer even with the right number. */
const PAST = /atr[aá]s|passou/i;

const CASES: readonly Case[] = [
  { question: "Quanto tempo falta para o natal?", tool: "dias_ate", answer: "49" },
  { question: "Quantos dias faltam para o Natal?", tool: "dias_ate", answer: "49" },
  { question: "Quantos dias faltam para o ano novo?", tool: "dias_ate", answer: "56" },
  { question: "Quantos dias faltam para o dia 15 de novembro?", tool: "dias_ate", answer: "9" },
  { question: "Falta quanto para o meu aniversário, que é 3 de março?", tool: "dias_ate", answer: "117" },
  { question: "Quantos dias até o dia das crianças, 12 de outubro?", tool: "dias_ate", answer: "340" },
  { question: "Quanto falta para o carnaval?", tool: "dias_ate", answer: "95" },
  { question: "Que dia é hoje?", tool: "data_atual", answer: "6 de novembro" },
  { question: "Qual é a data de hoje?", tool: "data_atual", answer: "6 de novembro" },
  { question: "Em que dia da semana estamos?", tool: "data_atual", answer: "sexta" },
  { question: "Que horas são?", tool: "data_atual", answer: "10:30" },
  { question: "Estamos em que mês?", tool: "data_atual", answer: "novembro" },
  { question: "Qual é a capital da França?", tool: null, answer: "Paris" },
  { question: "Quanto é 7 mais 5?", tool: null, answer: "12" },
  { question: "Quem escreveu Dom Casmurro?", tool: null, answer: "Machado" },
  { question: "Me dê uma dica curta para estudar melhor.", tool: null },
  { question: "Olá, tudo bem?", tool: null },
];

/* Held out: written after the design was chosen on CASES, run once, never
 * tuned on. This is the honest score. Includes a known gap (weekday names). */
const HOLDOUT: readonly Case[] = [
  { question: "Quantos dias faltam para a Páscoa?", tool: "dias_ate", answer: "142" },
  { question: "Quanto tempo até o Dia das Mães?", tool: "dias_ate", answer: "184" },
  { question: "Quantos dias faltam para Tiradentes?", tool: "dias_ate", answer: "166" },
  { question: "Daqui a quantos dias é o dia 1º de dezembro?", tool: "dias_ate", answer: "25" },
  { question: "Quantos dias faltam para 31/12?", tool: "dias_ate", answer: "55" },
  { question: "Quanto falta para o dia 20 de novembro?", tool: "dias_ate", answer: "14" },
  { question: "Quantos dias até sexta?", tool: "dias_ate", answer: "7" },
  { question: "Minha viagem é em 10 de janeiro. Quantos dias faltam?", tool: "dias_ate", answer: "65" },
  { question: "Hoje é que dia?", tool: "data_atual", answer: "6 de novembro" },
  { question: "Me diga a data de hoje, por favor.", tool: "data_atual", answer: "6 de novembro" },
  { question: "Agora são que horas?", tool: "data_atual", answer: "10:30" },
  { question: "Hoje é segunda-feira?", tool: "data_atual", answer: "sexta" },
  { question: "Qual é a melhor hora do dia para estudar?", tool: null },
  { question: "Quantas horas tem um dia?", tool: null, answer: "24" },
  { question: "O que é um feriado?", tool: null },
  { question: "Qual é o maior planeta do sistema solar?", tool: null, answer: "Júpiter" },
  { question: "Traduza 'bom dia' para o inglês.", tool: null, answer: "good morning" },
];

/* Second held-out set: written before the call validation was implemented
 * (after HOLDOUT exposed false calls on messages with time words), run once. */
const HOLDOUT2: readonly Case[] = [
  { question: "Quantos dias faltam para o Dia dos Pais?", tool: "dias_ate", answer: "275" },
  { question: "Falta muito para o réveillon?", tool: "dias_ate", answer: "55" },
  { question: "Quantos dias até 25/01?", tool: "dias_ate", answer: "80" },
  { question: "Em quantos dias chega o dia 30 de novembro?", tool: "dias_ate", answer: "24" },
  { question: "Quanto tempo falta para a Consciência Negra?", tool: "dias_ate", answer: "14" },
  { question: "Meu casamento é dia 14 de fevereiro. Faltam quantos dias?", tool: "dias_ate", answer: "100" },
  { question: "Que dia é hoje mesmo?", tool: "data_atual", answer: "6 de novembro" },
  { question: "Você sabe que horas são agora?", tool: "data_atual", answer: "10:30" },
  { question: "Estamos em que ano?", tool: "data_atual", answer: "2026" },
  { question: "Hoje é fim de semana?", tool: "data_atual", answer: "sexta" },
  { question: "Quantos minutos tem uma hora?", tool: null, answer: "60" },
  { question: "Me conte uma curiosidade sobre o mês de dezembro.", tool: null },
  { question: "Qual é a capital do Japão?", tool: null },
  { question: "Como faço para acordar mais cedo?", tool: null },
  { question: "Quantos dias tem uma semana?", tool: null, answer: "7" },
];

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    "model-dir": { type: "string", default: join(ROOT, "models", "qwen2.5-0.5b-instruct") },
    binary: { type: "string", default: join(ROOT, "build", `smolmoe${EXE}`) },
    verbose: { type: "boolean", default: false },
    session: { type: "boolean", default: false },
    set: { type: "string", default: "dev" }, /* dev: design set; holdout, holdout2: new questions */
  },
});

const SETS: Record<string, readonly Case[]> = { dev: CASES, holdout: HOLDOUT, holdout2: HOLDOUT2 };
if (!SETS[values.set!]) throw new Error("--set must be dev, holdout or holdout2");
const SET = SETS[values.set!]!;

function chat(questions: readonly string[]): { stdout: string; stderr: string } {
  const dir = values["model-dir"]!;
  const proc = Bun.spawnSync([values.binary!, "--model", join(dir, "model.smoe"), "--config", join(dir, "config.ini"),
    "--tokenizer", join(dir, "tokenizer.qbp"), "--now", NOW, "--tokens", "96"], {
    stdin: new TextEncoder().encode(`${questions.join("\n")}\n/sair\n`), stdout: "pipe", stderr: "pipe",
  });
  if (proc.exitCode !== 0) throw new Error(`exit ${proc.exitCode}\n${proc.stderr.toString()}`);
  return { stdout: proc.stdout.toString(), stderr: proc.stderr.toString() };
}

/* Not anchored: with piped input the notice follows the "> " prompt on the same line. */
const toolsIn = (stderr: string) => [...stderr.matchAll(/\[ferramenta\] ([\w?]+)/g)].map((m) => m[1]!);

function ask(c: Case): { tools: string[]; answer: string } {
  const { stdout, stderr } = chat([c.question]);
  return { tools: toolsIn(stderr), answer: stdout.trim() };
}

/* Every question in one conversation: after earlier turns (and their tool
 * results) the model behaves differently. Decoding is greedy, so the run with
 * questions 0..i extends the run with 0..i-1: the answer to question i is the
 * new stdout, and its tools are in the stderr segment after the last prompt. */
function sessionTurns(): { tools: string[]; answer: string }[] {
  const turns: { tools: string[]; answer: string }[] = [];
  let before = "";
  for (let i = 0; i < SET.length; ++i) {
    const { stdout, stderr } = chat(SET.slice(0, i + 1).map((c) => c.question));
    if (!stdout.startsWith(before)) throw new Error(`turn ${i}: the transcript is not deterministic`);
    turns.push({ tools: toolsIn(stderr.split("\n> ")[i + 1] ?? ""), answer: stdout.slice(before.length).trim() });
    before = stdout;
  }
  return turns;
}

const turns = values.session ? sessionTurns() : SET.map(ask);
let toolOk = 0, answerOk = 0, bothOk = 0, answerCases = 0;
const rows: string[] = [];
for (const [i, c] of SET.entries()) {
  const { tools, answer } = turns[i]!;
  const rightTool = c.tool ? tools.includes(c.tool) : tools.length === 0;
  const rightAnswer = (c.answer === undefined || answer.toLowerCase().includes(c.answer.toLowerCase())) &&
    !(c.tool === "dias_ate" && PAST.test(answer));
  toolOk += Number(rightTool);
  if (c.answer !== undefined) {
    answerCases += 1;
    answerOk += Number(rightAnswer);
  }
  bothOk += Number(rightTool && rightAnswer);
  rows.push(`| ${c.question} | ${c.tool ?? "—"} | ${tools.join(", ") || "—"} | ${rightTool && rightAnswer ? "ok" : "FALHA"} |`);
  if (values.verbose) console.error(`${rightTool && rightAnswer ? "ok   " : "FALHA"} ${c.question}\n      ${answer.replace(/\n/g, " ")}`);
}

console.log("| Pergunta | Ferramenta esperada | Chamada | Resultado |\n|---|---|---|---|");
console.log(rows.join("\n"));
console.log(`\n${values.session ? "sessão única" : "chats novos"}: ferramenta certa: ${toolOk}/${SET.length}; ` +
  `resposta certa: ${answerOk}/${answerCases}; ambos: ${bothOk}/${SET.length}`);
