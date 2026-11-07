/** Helpers to execute the smolmoe binaries from tests. */
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

export const ROOT = resolve(import.meta.dir, "..", "..");
const BUILD = resolve(ROOT, process.env.SMOLMOE_BUILD ?? "build");
const EXE = process.platform === "win32" ? ".exe" : "";

export const BINARY = join(BUILD, `smolmoe${EXE}`);
export const SCALAR_BINARY = join(BUILD, `smolmoe-scalar${EXE}`);
export const QUANTIZE_BINARY = join(BUILD, `smolmoe-quantize${EXE}`);
export const BUNDLE_BINARY = join(BUILD, `smolmoe-bundle${EXE}`);
/** Windows only: types into a program in its own console (tests/support/console_input.c). */
export const CONSOLE_INPUT_BINARY = join(BUILD, `console-input${EXE}`);
export const EXE_SUFFIX = EXE;

export interface Stats {
  readonly version: string;
  readonly backend: string;
  readonly threads: number;
  readonly batch: number;
  readonly budget_bytes: number;
  readonly tracked_peak_bytes: number;
  readonly rss_peak_bytes: number;
  readonly mapped_bytes: number;
  readonly kv_bytes: number;
  readonly prompt_tokens: number;
  readonly generated_tokens: number;
  readonly processed_tokens: number;
  readonly prefill_tokens_per_second: number;
  readonly decode_tokens_per_second: number;
  readonly expert_calls: number;
  readonly logits_hash: string;
}

export interface RunResult {
  readonly code: number;
  readonly stdout: Buffer;
  readonly stderr: string;
}

export function run(args: readonly (string | number)[], binary: string = BINARY): RunResult {
  const proc = Bun.spawnSync([binary, ...args.map(String)], { stdout: "pipe", stderr: "pipe" });
  return {
    code: proc.exitCode ?? -1,
    stdout: Buffer.from(proc.stdout),
    stderr: proc.stderr.toString(),
  };
}

/** Runs with bytes piped to stdin (interactive chat), optionally in another directory. */
export function runWithInput(args: readonly (string | number)[], input: string, binary: string = BINARY,
  cwd?: string): RunResult {
  const proc = Bun.spawnSync([binary, ...args.map(String)], {
    stdin: new TextEncoder().encode(input),
    stdout: "pipe",
    stderr: "pipe",
    cwd,
  });
  return { code: proc.exitCode ?? -1, stdout: Buffer.from(proc.stdout), stderr: proc.stderr.toString() };
}

/** Runs and requires success; returns the parsed stats line. */
export function runOk(args: readonly (string | number)[], binary: string = BINARY): RunResult & { stats: Stats } {
  const result = run(args, binary);
  if (result.code !== 0) throw new Error(`smolmoe failed (${result.code}): ${result.stderr}`);
  const line = result.stderr.trim().split("\n").at(-1) ?? "";
  return { ...result, stats: JSON.parse(line) as Stats };
}

/** Runs and requires a clean, prefixed error. Returns stderr. */
export function runFails(args: readonly (string | number)[], binary: string = BINARY): string {
  const result = run(args, binary);
  if (result.code === 0) throw new Error(`expected failure for: ${args.join(" ")}`);
  return result.stderr;
}

export function tempDir(prefix: string): { path: string; dispose: () => void } {
  const path = mkdtempSync(join(tmpdir(), `smolmoe-${prefix}-`));
  return { path, dispose: () => rmSync(path, { recursive: true, force: true }) };
}

export interface BundledModel {
  readonly model: string;
  readonly config: string;
  readonly tokenizer: string;
}

export function bundledModel(dir: string, tokenizerFile: string): BundledModel | null {
  const base = join(ROOT, "models", dir);
  const model = join(base, "model.smoe");
  const tokenizer = join(base, tokenizerFile);
  return existsSync(model) && existsSync(tokenizer) ? { model, config: join(base, "config.ini"), tokenizer } : null;
}

export function modelArgs(m: BundledModel): string[] {
  return ["--model", m.model, "--config", m.config, "--tokenizer", m.tokenizer];
}
