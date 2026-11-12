#!/usr/bin/env bun
/**
 * Hugging Face checkpoint -> smolmoe converter (replaces tools/import_qwen2.py).
 *
 * usage: bun scripts/convert.ts COMMAND ...
 *   qwen2 SOURCE OUTDIR [--container v2|v3] [--threads N]
 *       Qwen2.5-0.5B-Instruct (pinned): byte-identical to the shipped files with --container v2.
 *   phi3 SOURCE OUTDIR [--container v2|v3] [--threads N]
 *       Phi-4-mini-instruct (pinned revision and tokenizer).
 *   llama SOURCE OUTDIR [--container v2|v3] [--threads N]
 *       Llama 3.x (LlamaForCausalLM): Dolphin 3.0 Llama 3.2 1B (pinned by fetch-llama), 3B, Llama 3.1 8B...
 *   fetch-qwen2 DIR     download the pinned Qwen2.5 source files into DIR (SHA-256 verified)
 *   fetch-phi3 DIR      download the pinned Phi-4-mini source files into DIR (SHA-256 verified, 7.7 GB)
 *   fetch-gemma3 DIR    download the pinned Gemma 3 1B source files into DIR (SHA-256 verified)
 *   fetch-llama DIR     download the pinned Dolphin 3.0 Llama 3.2 1B source files into DIR (SHA-256 verified, 2.5 GB)
 *   bundle MODEL CONFIG TOKENIZER OUT
 *       repackage a model into a single-file v3 container (same bytes as smolmoe-bundle)
 *   check MODEL [--config C] [--tokenizer T]
 *       re-check a model as the C loader would (directory, payload, config, tokenizer, tensor set)
 *
 * --container v2 (default): model.smoe + config.ini + tokenizer.qbp; v3: model.smoe embeds both.
 * --threads defaults to the CPU count (max 8); SMOLMOE_CONVERT_THREADS overrides it.
 */
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";

import { type Container, type ConvertOptions, type ConvertReport, defaultThreads } from "./hf/common";
import { fetchPinned } from "./hf/fetch";
import { GEMMA3_1B, GEMMA3_FILES, convertGemma3 } from "./hf/gemma3";
import { DOLPHIN3_1B, DOLPHIN3_1B_FILES, convertLlama } from "./hf/llama";
import { PHI4_MINI, PHI4_MINI_FILES, convertPhi3 } from "./hf/phi3";
import { QWEN2, QWEN2_FILES, convertQwen2 } from "./hf/qwen2";
import { bundleSingleFile } from "./smoe/bundle";
import { checkFiles } from "./smoe/check";
import { assertLittleEndian, message } from "./smoe/format";

const USAGE = "usage: bun scripts/convert.ts qwen2|phi3|gemma3|llama SOURCE OUTDIR [--container v2|v3] [--threads N]\n" +
  "       bun scripts/convert.ts fetch-qwen2|fetch-phi3|fetch-gemma3|fetch-llama DIR\n" +
  "       bun scripts/convert.ts bundle MODEL CONFIG TOKENIZER OUT\n" +
  "       bun scripts/convert.ts check MODEL [--config C] [--tokenizer T]";

class UsageError extends Error {}

const log = (line: string) => console.error(`convert: ${line}`);

function threadCount(value: string | undefined): number {
  const text = value ?? process.env.SMOLMOE_CONVERT_THREADS;
  if (text === undefined) return defaultThreads();
  const n = Number(text);
  if (!/^[0-9]+$/.test(text) || n < 1 || n > 64) throw new UsageError(`--threads must be 1..64, got ${text}`);
  return n;
}

function container(value: string | undefined): Container {
  if (value === undefined || value === "v2" || value === "v3") return value ?? "v2";
  throw new UsageError(`--container must be v2 or v3, got ${value}`);
}

function printReport(r: ConvertReport): void {
  for (const f of r.files) log(`${f.path} (${f.bytes} bytes)`);
  log(`done in ${r.seconds.toFixed(1)} s`);
}

function parse(argv: readonly string[]) {
  try {
    return parseArgs({
      args: [...argv], allowPositionals: true, strict: true,
      options: { container: { type: "string" }, threads: { type: "string" }, config: { type: "string" },
        tokenizer: { type: "string" }, help: { type: "boolean" } },
    });
  } catch (e) {
    throw new UsageError(`${message(e)}\n${USAGE}`);
  }
}

export async function main(argv: readonly string[]): Promise<void> {
  assertLittleEndian(); /* typed-array views over file bytes */
  const { values, positionals } = parse(argv);
  const [command, ...rest] = positionals;
  if (values.help || !command) throw new UsageError(USAGE);
  if (command === "qwen2" || command === "phi3" || command === "gemma3" || command === "llama") {
    if (rest.length !== 2) throw new UsageError(USAGE);
    const options: ConvertOptions = { source: rest[0]!, output: rest[1]!, container: container(values.container),
      threads: threadCount(values.threads), pinned: command === "qwen2" || command === "phi3", log };
    printReport(await (command === "qwen2" ? convertQwen2(options)
      : command === "phi3" ? convertPhi3(options) : command === "llama" ? convertLlama(options) : convertGemma3(options)));
  } else if (command === "fetch-qwen2" || command === "fetch-phi3" || command === "fetch-gemma3" || command === "fetch-llama") {
    if (rest.length !== 1) throw new UsageError(USAGE);
    const { fetched } = command === "fetch-qwen2" ? await fetchPinned(QWEN2.repo, QWEN2_FILES, rest[0]!, log)
      : command === "fetch-phi3" ? await fetchPinned(PHI4_MINI.repo, PHI4_MINI_FILES, rest[0]!, log)
      : command === "fetch-llama" ? await fetchPinned(DOLPHIN3_1B.repo, DOLPHIN3_1B_FILES, rest[0]!, log)
      : await fetchPinned(GEMMA3_1B.repo, GEMMA3_FILES, rest[0]!, log);
    log(`${fetched.length} file(s) downloaded, the rest already present and verified`);
  } else if (command === "bundle") {
    if (rest.length !== 4) throw new UsageError(USAGE);
    const [model, config, tokenizer, out] = rest as [string, string, string, string];
    const bytes = await bundleSingleFile(model, readFileSync(config), readFileSync(tokenizer), out);
    log(`${out} (${bytes} bytes)`);
  } else if (command === "check") {
    if (rest.length !== 1) throw new UsageError(USAGE);
    const r = checkFiles({ model: rest[0]!, config: values.config ? readFileSync(values.config) : undefined,
      tokenizer: values.tokenizer ? readFileSync(values.tokenizer) : undefined });
    process.stdout.write(`${JSON.stringify({ version: r.container.version, records: r.container.count,
      blobs: r.container.blobs, bytes: r.container.size, config: r.config, tokenizer: r.tokenizer })}\n`);
  } else throw new UsageError(USAGE);
}

if (import.meta.main) {
  try {
    await main(Bun.argv.slice(2));
  } catch (e) {
    console.error(e instanceof UsageError ? e.message : `convert: ${message(e)}`);
    process.exit(e instanceof UsageError ? 2 : 1);
  }
}
