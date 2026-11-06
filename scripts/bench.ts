#!/usr/bin/env bun
/**
 * Repeats a generation and reports median prefill/decode throughput and TTFT.
 *
 * usage: bun scripts/bench.ts [--model-dir DIR] [--runs N] [--threads N] [--batch N] [--tokens N] [--binary PATH]
 */
import { existsSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";

const ROOT = resolve(import.meta.dir, "..");
const EXE = process.platform === "win32" ? ".exe" : "";
const PROMPT =
  "Explique em um parágrafo como funciona a fotossíntese, citando a luz, a água, o gás carbônico e a glicose, " +
  "e por que ela é importante para a vida na Terra.";

interface RunStats {
  readonly backend: string;
  readonly threads: number;
  readonly batch: number;
  readonly prompt_tokens: number;
  readonly generated_tokens: number;
  readonly ttft_seconds: number;
  readonly prefill_tokens_per_second: number;
  readonly decode_tokens_per_second: number;
  readonly tracked_peak_bytes: number;
  readonly rss_peak_bytes: number;
}

const { values } = parseArgs({
  args: Bun.argv.slice(2),
  options: {
    "model-dir": { type: "string", default: join(ROOT, "models", "qwen2.5-0.5b-instruct") },
    runs: { type: "string", default: "5" },
    threads: { type: "string" },
    batch: { type: "string" },
    tokens: { type: "string", default: "48" },
    binary: { type: "string", default: join(ROOT, "build", `smolmoe${EXE}`) },
  },
});

function tokenizerIn(dir: string): string {
  const name = readdirSync(dir).find((f) => /^tokenizer\.(qbp|spb|bpe)$/.test(f));
  if (!name) throw new Error(`no tokenizer in ${dir}`);
  return join(dir, name);
}

function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function runOnce(args: readonly string[]): RunStats {
  const proc = Bun.spawnSync([values.binary!, ...args], { stdout: "ignore", stderr: "pipe" });
  const stderr = proc.stderr.toString();
  if (proc.exitCode !== 0) throw new Error(`benchmark run failed: ${stderr}`);
  return JSON.parse(stderr.trim().split("\n").at(-1) ?? "{}") as RunStats;
}

function main(): void {
  const dir = resolve(values["model-dir"]!);
  if (!existsSync(join(dir, "model.smoe"))) {
    console.error(`model not found in ${dir} (see docs/MODELS.md)`);
    process.exit(2);
  }
  const tokenizer = tokenizerIn(dir);
  const chat = tokenizer.endsWith(".qbp");
  const args = ["--model", join(dir, "model.smoe"), "--config", join(dir, "config.ini"), "--tokenizer", tokenizer,
    "--prompt", chat ? PROMPT : "Once upon a time, there was a little robot", "--tokens", values.tokens!,
    "--context", "512", ...(chat ? ["--chat"] : []),
    ...(values.threads ? ["--threads", values.threads] : []), ...(values.batch ? ["--batch", values.batch] : [])];
  const runs = Number.parseInt(values.runs!, 10);
  runOnce(args); /* warm the page cache */
  const results = Array.from({ length: runs }, () => runOnce(args));
  const first = results[0];
  const report = {
    model: dir,
    backend: first.backend,
    threads: first.threads,
    batch: first.batch,
    runs,
    prompt_tokens: first.prompt_tokens,
    generated_tokens: median(results.map((r) => r.generated_tokens)),
    ttft_seconds: median(results.map((r) => r.ttft_seconds)),
    prefill_tokens_per_second: median(results.map((r) => r.prefill_tokens_per_second)),
    decode_tokens_per_second: median(results.map((r) => r.decode_tokens_per_second)),
    heap_mib: first.tracked_peak_bytes / 1048576,
    rss_mib: Math.max(...results.map((r) => r.rss_peak_bytes)) / 1048576,
  };
  console.log(JSON.stringify(report, null, 2));
}

main();
