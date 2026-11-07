#!/usr/bin/env bun
/**
 * Independent float64 reference forward pass of Phi-3 / Phi-4-mini-instruct,
 * computed straight from the original BF16 safetensors with Hugging Face
 * semantics (fused qkv_proj / gate_up_proj sliced as published, split-half
 * partial RoPE with attention_factor on cos/sin, tied lm_head). It shares no
 * code or conventions with the C runtime or the converter.
 *
 * usage:
 *   bun scripts/ref/phi3-reference.ts <model_dir> --ids 200022,50,... [--out logits.f32]
 *       [--hidden hidden.f32] [--top 5] [--generate N]
 *   bun scripts/ref/phi3-reference.ts <model_dir> --system "..." --user "..." [same options]
 *   bun scripts/ref/phi3-reference.ts <model_dir> --self-check [--ids <id>]
 *
 * --out     logits for every position, little-endian float32, positions x vocab_size.
 * --hidden  residual stream snapshots, little-endian float32, (num_layers + 2) x positions x hidden_size:
 *           [0] embeddings, [l + 1] output of layer l, [num_layers + 1] after the final norm.
 * --generate N  greedy continuation, re-running the full causal forward each step.
 * --system/--user  build ids with the chat template from tokenizer.json (instead of --ids).
 */
import { closeSync, existsSync, openSync, readFileSync, writeFileSync, writeSync } from "node:fs";
import { join } from "node:path";

import { loadPhi3Config, type Phi3Config } from "./config";
import { topK } from "./linear";
import { runForward, validateCheckpoint } from "./model";
import { WeightStore } from "./safetensors";
import { runSelfCheck } from "./self-check";
import { decodeIds, encodeChat, loadTokenizer, type Tokenizer } from "./vocab";

export interface CliArgs {
  readonly modelDir: string;
  readonly ids: readonly number[] | null;
  readonly system: string | null;
  readonly user: string | null;
  readonly out: string | null;
  readonly hidden: string | null;
  readonly top: number;
  readonly generate: number;
  readonly selfCheck: boolean;
}

const VALUE_FLAGS = new Set(["--ids", "--system", "--user", "--out", "--hidden", "--top", "--generate"]);
const MAX_TOKENS = 4096;

function count(flag: string, text: string, min: number): number {
  const n = Number(text);
  if (!Number.isSafeInteger(n) || n < min) throw new Error(`${flag} must be an integer >= ${min}, got ${text}`);
  return n;
}

export function parseIds(text: string): number[] {
  const ids = text.split(",").map((s) => s.trim()).filter((s) => s.length > 0).map((s) => count("--ids", s, 0));
  if (ids.length === 0 || ids.length > MAX_TOKENS) throw new Error(`--ids needs 1..${MAX_TOKENS} ids`);
  return ids;
}

export function parseArgs(argv: readonly string[]): CliArgs {
  const values = new Map<string, string>();
  const positional: string[] = [];
  let selfCheck = false;
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg === "--self-check") selfCheck = true;
    else if (VALUE_FLAGS.has(arg)) {
      if (i + 1 >= argv.length) throw new Error(`${arg} needs a value`);
      values.set(arg, argv[++i]!);
    } else if (arg.startsWith("--")) throw new Error(`unknown option ${arg}`);
    else positional.push(arg);
  }
  if (positional.length !== 1) throw new Error("expected exactly one <model_dir>");
  const ids = values.has("--ids") ? parseIds(values.get("--ids")!) : null;
  const system = values.get("--system") ?? null;
  const user = values.get("--user") ?? null;
  if (ids !== null && (system !== null || user !== null)) throw new Error("use either --ids or --system/--user");
  if ((system === null) !== (user === null)) throw new Error("--system and --user go together");
  if (!selfCheck && ids === null && system === null) throw new Error("need --ids or --system/--user");
  return {
    modelDir: positional[0]!,
    ids,
    system,
    user,
    out: values.get("--out") ?? null,
    hidden: values.get("--hidden") ?? null,
    top: values.has("--top") ? count("--top", values.get("--top")!, 1) : 5,
    generate: values.has("--generate") ? count("--generate", values.get("--generate")!, 0) : 0,
    selfCheck,
  };
}

/** Little-endian float32 bytes of a float64 buffer (rounded to nearest). */
export function toFloat32LE(values: Float64Array): Uint8Array {
  const bytes = new Uint8Array(values.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < values.length; i++) view.setFloat32(i * 4, values[i]!, true);
  return bytes;
}

function stopTokens(modelDir: string): ReadonlySet<number> {
  const path = join(modelDir, "generation_config.json");
  if (!existsSync(path)) return new Set();
  const eos: unknown = (JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>).eos_token_id;
  return new Set((Array.isArray(eos) ? eos : [eos]).filter((v): v is number => typeof v === "number"));
}

function tokenText(tok: Tokenizer | null, id: number): string {
  return tok === null ? "" : JSON.stringify(decodeIds(tok, [id]));
}

function reportPositions(logits: Float64Array, ids: readonly number[], cfg: Phi3Config, tok: Tokenizer | null, k: number): void {
  const V = cfg.vocabSize;
  ids.forEach((id, t) => {
    const best = topK(logits, t * V, V, Math.max(k, 2));
    const margin = best[0]!.logit - best[1]!.logit;
    console.log(`pos ${String(t).padStart(2)} in ${String(id).padStart(6)} ${tokenText(tok, id)}`);
    console.log(`   top${k}: ${best.slice(0, k).map((s) => `${s.id} ${tokenText(tok, s.id)} ${s.logit.toFixed(6)}`).join(" | ")}`);
    console.log(`   margin top1-top2: ${margin.toFixed(6)}`);
  });
}

function greedy(store: WeightStore, cfg: Phi3Config, ids: readonly number[], steps: number, tok: Tokenizer | null, stops: ReadonlySet<number>, first: number): number[] {
  const out = [first];
  console.log(`greedy step 1: ${first} ${tokenText(tok, first)}`);
  while (out.length < steps && !stops.has(out[out.length - 1]!)) {
    const started = performance.now();
    const seq = [...ids, ...out];
    const { logits } = runForward(store, cfg, seq);
    const [best] = topK(logits, (seq.length - 1) * cfg.vocabSize, cfg.vocabSize, 1);
    out.push(best!.id);
    console.log(`greedy step ${out.length}: ${best!.id} ${tokenText(tok, best!.id)} (${((performance.now() - started) / 1000).toFixed(1)} s)`);
  }
  return out;
}

function main(argv: readonly string[]): number {
  const args = parseArgs(argv);
  const started = performance.now();
  const cfg = loadPhi3Config(args.modelDir);
  const tok = existsSync(join(args.modelDir, "tokenizer.json")) ? loadTokenizer(args.modelDir) : null;
  const ids = args.ids ?? (args.system !== null && tok !== null ? encodeChat(tok, args.system, args.user!) : null);
  if (args.system !== null && tok === null) throw new Error("--system/--user need tokenizer.json in the model dir");
  const store = WeightStore.open(args.modelDir);
  try {
    validateCheckpoint(store, cfg);
    if (args.selfCheck) return runSelfCheck(store, cfg, ids?.[0] ?? 0, console.log) ? 0 : 1;
    if (ids === null) throw new Error("no ids");
    console.log(`ids (${ids.length}): ${ids.join(",")}`);
    if (tok !== null) console.log(`text: ${JSON.stringify(decodeIds(tok, ids))}`);

    const hiddenFd = args.hidden === null ? null : openSync(args.hidden, "w");
    const forwardStart = performance.now();
    let result;
    try {
      result = runForward(store, cfg, ids, {
        log: (m) => console.error(m),
        onHidden: hiddenFd === null ? undefined : (_, h) => void writeSync(hiddenFd, toFloat32LE(h)),
      });
    } finally {
      if (hiddenFd !== null) closeSync(hiddenFd);
    }
    const forwardSeconds = (performance.now() - forwardStart) / 1000;
    const { rope, logits } = result;
    console.log(`rope: ${rope.factorKind} factors, attention_factor ${rope.attentionFactor}, inv_freq[0..3] ${Array.from(rope.invFreq.subarray(0, 4)).join(", ")}`);
    if (args.out !== null) writeFileSync(args.out, toFloat32LE(logits));
    reportPositions(logits, ids, cfg, tok, args.top);
    console.log(`forward: ${ids.length} positions in ${forwardSeconds.toFixed(1)} s`);

    if (args.generate > 0) {
      const [first] = topK(logits, (ids.length - 1) * cfg.vocabSize, cfg.vocabSize, 1);
      const gen = greedy(store, cfg, ids, args.generate, tok, stopTokens(args.modelDir), first!.id);
      if (tok !== null) console.log(`greedy continuation: ${JSON.stringify(decodeIds(tok, gen))}`);
    }
    console.log(`total: ${((performance.now() - started) / 1000).toFixed(1)} s`);
    return 0;
  } finally {
    store.close();
  }
}

if (import.meta.main) {
  try {
    process.exit(main(process.argv.slice(2)));
  } catch (err: unknown) {
    console.error(`error: ${err instanceof Error ? err.message : String(err)}`);
    process.exit(2);
  }
}
