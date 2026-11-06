#!/usr/bin/env bun
/**
 * Performance pipeline: runs fixed scenarios several times with --profile,
 * reports medians (throughput, latency, memory, time per forward phase), saves
 * the report and compares it with a baseline. Exits 1 when a key metric
 * regresses beyond the threshold.
 *
 * On a laptop, heat shifts the numbers between sessions; to judge a change,
 * use --ab with the previous build: both binaries run alternately in the same
 * session and the old one is the base.
 *
 * usage: bun scripts/bench.ts [--model-dir DIR] [--binary PATH] [--runs N] [--threads N] [--batch N]
 *                             [--save FILE] [--compare FILE | --ab OLD_BINARY] [--threshold PCT]
 */
import { existsSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { cpus, platform, release } from "node:os";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const ROOT = resolve(import.meta.dir, "..");
const EXE = process.platform === "win32" ? ".exe" : "";
const CHAT_PROMPT =
  "Explique em um parágrafo como funciona a fotossíntese, citando a luz, a água, o gás carbônico e a glicose, " +
  "e por que ela é importante para a vida na Terra.";
const LONG_PROMPT_BYTES = 1400; /* about 390 Qwen tokens of tests/goldens/eval-pt.txt */

export interface RunStats {
  readonly backend: string;
  readonly threads: number;
  readonly batch: number;
  readonly prompt_tokens: number;
  readonly generated_tokens: number;
  readonly ttft_seconds: number;
  readonly prefill_tokens_per_second: number;
  readonly decode_tokens_per_second: number;
  readonly rss_peak_bytes: number;
  readonly output_hash: string; /* of the generated text, added by the runner */
  readonly profile?: Readonly<Record<string, number>>;
}

export interface ScenarioReport {
  readonly prompt_tokens: number;
  readonly generated_tokens: number;
  readonly prefill_tokens_per_second: number;
  readonly decode_tokens_per_second: number;
  readonly ttft_seconds: number;
  readonly rss_mib: number;
  /* Median seconds per forward phase over the whole run (prefill + decode). */
  readonly phases: Readonly<Record<string, number>>;
}

export interface BenchReport {
  readonly date: string;
  readonly commit: string;
  readonly machine: { readonly cpu: string; readonly logical_cores: number; readonly os: string };
  readonly backend: string;
  readonly threads: number;
  readonly batch: number;
  readonly runs: number;
  readonly scenarios: Readonly<Record<string, ScenarioReport>>;
}

export type Verdict = "melhor" | "pior" | "igual";

export interface Delta {
  readonly scenario: string;
  readonly metric: string;
  readonly base: number;
  readonly now: number;
  readonly percent: number; /* positive = better */
  readonly verdict: Verdict;
}

/* Key metrics and whether a higher value is better. */
const KEY_METRICS: readonly (readonly [keyof ScenarioReport, boolean])[] = [
  ["prefill_tokens_per_second", true],
  ["decode_tokens_per_second", true],
  ["ttft_seconds", false],
  ["rss_mib", false],
];

export function median(xs: readonly number[]): number {
  if (!xs.length) throw new Error("median of an empty list");
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function summarize(runs: readonly RunStats[]): ScenarioReport {
  if (!runs.length) throw new Error("no runs to summarize");
  const hashes = new Set(runs.map((r) => r.output_hash));
  if (hashes.size > 1) throw new Error("runs of the same scenario produced different text");
  const names = Object.keys(runs[0].profile ?? {});
  return {
    prompt_tokens: runs[0].prompt_tokens,
    generated_tokens: median(runs.map((r) => r.generated_tokens)),
    prefill_tokens_per_second: median(runs.map((r) => r.prefill_tokens_per_second)),
    decode_tokens_per_second: median(runs.map((r) => r.decode_tokens_per_second)),
    ttft_seconds: median(runs.map((r) => r.ttft_seconds)),
    rss_mib: Math.max(...runs.map((r) => r.rss_peak_bytes)) / 1048576,
    phases: Object.fromEntries(names.map((n) => [n, median(runs.map((r) => r.profile?.[n] ?? 0))])),
  };
}

/* Percent change where positive means better; within ±threshold is "igual".
 * A metric the base has but the new report lacks (a missing scenario, or a
 * value that dropped to 0) is "pior". */
export function compare(base: BenchReport, now: BenchReport, thresholdPercent: number): Delta[] {
  if (!Number.isFinite(thresholdPercent) || thresholdPercent < 0) throw new Error("threshold must be a percentage >= 0");
  const deltas: Delta[] = [];
  for (const [scenario, b] of Object.entries(base.scenarios)) {
    const n = now.scenarios[scenario];
    for (const [metric, higherIsBetter] of KEY_METRICS) {
      const bv = b[metric] as number, nv = (n?.[metric] as number | undefined) ?? 0;
      if (!(bv > 0)) continue; /* e.g. no decode in a prefill-only scenario */
      if (!(nv > 0)) {
        deltas.push({ scenario, metric, base: bv, now: nv, percent: -100, verdict: "pior" });
        continue;
      }
      const raw = ((nv - bv) / bv) * 100;
      const percent = higherIsBetter ? raw : -raw;
      const verdict: Verdict = percent > thresholdPercent ? "melhor" : percent < -thresholdPercent ? "pior" : "igual";
      deltas.push({ scenario, metric, base: bv, now: nv, percent, verdict });
    }
  }
  return deltas;
}

function tokenizerIn(dir: string): string {
  const name = readdirSync(dir).find((f) => /^tokenizer\.(qbp|spb|bpe)$/.test(f));
  if (!name) throw new Error(`no tokenizer in ${dir}`);
  return join(dir, name);
}

function runOnce(binary: string, args: readonly string[]): RunStats {
  const proc = Bun.spawnSync([binary, ...args], { stdout: "pipe", stderr: "pipe" });
  const stderr = proc.stderr.toString();
  if (proc.exitCode !== 0) throw new Error(`benchmark run failed: ${stderr}`);
  const stats = JSON.parse(stderr.trim().split("\n").at(-1) ?? "{}") as Omit<RunStats, "output_hash">;
  return { ...stats, output_hash: Bun.hash(proc.stdout).toString(16) };
}

/* Runs every scenario `runs` times per binary, alternating binaries (and their
 * order on each round) so slow drifts such as heat hit all of them alike. */
function measure(binaries: readonly string[], scenarios: Readonly<Record<string, string[]>>, runs: number):
    Record<string, RunStats[]>[] {
  const extra = binaries.map((b) => (supportsProfile(b) ? ["--profile"] : []));
  const results = binaries.map(() => Object.fromEntries(Object.keys(scenarios).map((s) => [s, [] as RunStats[]])));
  binaries.forEach((b, i) => runOnce(b, [...scenarios.chat, ...extra[i]])); /* warm the page cache */
  for (let round = 0; round < runs; ++round)
    for (const [name, args] of Object.entries(scenarios))
      for (let k = 0; k < binaries.length; ++k) {
        const b = round % 2 ? binaries.length - 1 - k : k;
        results[b][name].push(runOnce(binaries[b], [...args, ...extra[b]]));
      }
  return results;
}

/* The first LONG_PROMPT_BYTES of the Portuguese eval text, cut at a line end. */
function longPrompt(): string {
  const text = readFileSync(join(ROOT, "tests", "goldens", "eval-pt.txt"), "utf8");
  const cut = text.lastIndexOf("\n", LONG_PROMPT_BYTES);
  return text.slice(0, cut > 0 ? cut : LONG_PROMPT_BYTES);
}

function gitCommit(): string {
  try {
    const p = Bun.spawnSync(["git", "rev-parse", "--short", "HEAD"], { cwd: ROOT, stdout: "pipe", stderr: "ignore" });
    const dirty = Bun.spawnSync(["git", "status", "--porcelain", "--untracked-files=no"], { cwd: ROOT, stdout: "pipe" });
    const sha = p.exitCode === 0 ? p.stdout.toString().trim() : "unknown";
    return dirty.stdout.toString().trim() ? `${sha}+dirty` : sha;
  } catch {
    return "unknown"; /* git not installed */
  }
}

/* Builds older than the profiler reject --profile; they are measured without it. */
function supportsProfile(binary: string): boolean {
  const help = Bun.spawnSync([binary, "--help"], { stdout: "pipe", stderr: "ignore" });
  return help.stdout.toString().includes("--profile");
}

function printReport(r: BenchReport): void {
  console.log(`${r.machine.cpu} · ${r.backend} · ${r.threads} threads · lote ${r.batch} · ${r.runs} rodadas · ${r.commit}`);
  for (const [name, s] of Object.entries(r.scenarios)) {
    const total = Object.values(s.phases).reduce((a, b) => a + b, 0);
    console.log(`\n${name}: ${s.prompt_tokens} tokens de entrada, ${s.generated_tokens} gerados`);
    console.log(`  leitura ${s.prefill_tokens_per_second.toFixed(1)} tok/s · escrita ${s.decode_tokens_per_second.toFixed(1)} tok/s` +
      ` · 1º token ${s.ttft_seconds.toFixed(2)} s · RSS ${s.rss_mib.toFixed(0)} MiB`);
    for (const [phase, sec] of Object.entries(s.phases).sort((a, b) => b[1] - a[1]))
      if (total > 0 && sec / total >= 0.005)
        console.log(`  ${phase.padEnd(11)} ${(sec * 1000).toFixed(0).padStart(6)} ms ${((100 * sec) / total).toFixed(1).padStart(5)}%`);
  }
}

function printDeltas(deltas: readonly Delta[]): void {
  console.log("\ncomparação com a base (positivo = melhor):");
  for (const d of deltas)
    console.log(`  ${d.scenario.padEnd(7)} ${d.metric.padEnd(26)} ${d.base.toFixed(2).padStart(9)} → ${d.now.toFixed(2).padStart(9)}` +
      `  ${(d.percent >= 0 ? "+" : "") + d.percent.toFixed(1)}%  ${d.verdict}`);
}

function main(): void {
  const { values } = parseArgs({
    args: Bun.argv.slice(2),
    options: {
      "model-dir": { type: "string", default: join(ROOT, "models", "qwen2.5-0.5b-instruct") },
      binary: { type: "string", default: join(ROOT, "build", `smolmoe${EXE}`) },
      runs: { type: "string", default: "5" },
      threads: { type: "string" },
      batch: { type: "string" },
      save: { type: "string" },
      compare: { type: "string" },
      ab: { type: "string" },
      threshold: { type: "string", default: "5" },
    },
  });
  const dir = resolve(values["model-dir"]!);
  if (!existsSync(join(dir, "model.smoe"))) {
    console.error(`model not found in ${dir} (bun scripts/fetch-models.ts; see docs/MODELS.md)`);
    process.exit(2);
  }
  const runs = Number.parseInt(values.runs!, 10);
  if (!(runs >= 1)) throw new Error("--runs must be a positive integer");
  const tokenizer = tokenizerIn(dir);
  const chat = tokenizer.endsWith(".qbp");
  const common = ["--model", join(dir, "model.smoe"), "--config", join(dir, "config.ini"), "--tokenizer", tokenizer,
    "--context", "1024",
    ...(values.threads ? ["--threads", values.threads] : []), ...(values.batch ? ["--batch", values.batch] : [])];
  const scenarios: Record<string, string[]> = {
    /* Reading a long prompt: what the chat pays at startup and for tool results. */
    leitura: [...common, "--prompt", longPrompt(), "--tokens", "1"],
    /* A short chat question and a 64-token answer: time to first token and writing speed. */
    chat: [...common, "--prompt", chat ? CHAT_PROMPT : "Once upon a time, there was a little robot",
      "--tokens", "64", ...(chat ? ["--chat"] : [])],
  };
  if (values.compare && values.ab) throw new Error("use either --compare or --ab");
  const threshold = Number(values.threshold);
  if (!Number.isFinite(threshold) || threshold < 0) throw new Error("--threshold must be a percentage >= 0");
  const binaries = [resolve(values.binary!), ...(values.ab ? [resolve(values.ab)] : [])];
  const machine = { cpu: cpus()[0]?.model.trim() ?? "unknown", logical_cores: cpus().length, os: `${platform()} ${release()}` };
  const reports = measure(binaries, scenarios, runs).map((results, i): BenchReport => {
    const first = results.leitura[0];
    return {
      date: new Date().toISOString(),
      commit: i === 0 ? gitCommit() : `base: ${binaries[i]}`,
      machine,
      backend: first.backend,
      threads: first.threads,
      batch: first.batch,
      runs,
      scenarios: Object.fromEntries(Object.entries(results).map(([name, rs]) => [name, summarize(rs)])),
    };
  });
  const report = reports[0];
  printReport(report);
  if (values.save) writeFileSync(values.save, JSON.stringify(report, null, 2) + "\n");
  let base: BenchReport | undefined = reports[1];
  if (values.compare) {
    base = JSON.parse(readFileSync(values.compare, "utf8")) as BenchReport;
    if (base.machine.cpu !== report.machine.cpu)
      console.warn(`\naviso: a base foi medida em outra máquina (${base.machine.cpu})`);
  }
  if (base) {
    const deltas = compare(base, report, threshold);
    printDeltas(deltas);
    if (deltas.some((d) => d.verdict === "pior")) process.exit(1);
  }
}

if (import.meta.main) main();
