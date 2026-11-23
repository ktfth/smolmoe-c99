/**
 * Qwen3-MoE end to end on a tiny synthetic checkpoint (random weights, not a
 * trained model): the converter (scripts/hf/qwen3moe.ts) and the runtime's
 * routed experts, per-head Q/K norms and softmax routing, against the
 * independent float64 reference (scripts/ref/qwen3moe-reference.ts), position
 * by position. The runtime quantizes weights and activations to 8 bits and
 * the reference does not, so they agree closely, not exactly.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { parseHexFloat } from "../scripts/compare-traces";
import { convertQwen3Moe } from "../scripts/hf/qwen3moe";
import { qwen2Tokenizer } from "../scripts/hf/qwen2";
import { QWEN3MOE_SHAPE, writeQwen3MoeSynthetic } from "../scripts/hf/synthetic";
import { forward, loadConfig } from "../scripts/ref/qwen3moe-reference";
import { WeightStore } from "../scripts/ref/safetensors";
import { checkFiles } from "../scripts/smoe/check";
import { run, tempDir } from "./support/run";

const work = tempDir("qwen3moe");
afterAll(() => work.dispose());
const SRC = join(work.path, "src"), OUT = join(work.path, "out");
const TEXT = "the quick brown fox jumps over the lazy dog, again and again";

beforeAll(async () => {
  writeQwen3MoeSynthetic(SRC);
  await convertQwen3Moe({ source: SRC, output: OUT, container: "v2", threads: 2, pinned: false, log: () => {} });
});

function logSoftmax(row: ArrayLike<number>): Float64Array {
  let high = -Infinity;
  for (let i = 0; i < row.length; i++) high = Math.max(high, row[i]!);
  let sum = 0;
  for (let i = 0; i < row.length; i++) sum += Math.exp(row[i]! - high);
  return Float64Array.from({ length: row.length }, (_, i) => row[i]! - high - Math.log(sum));
}

describe("Qwen3-MoE synthetic checkpoint", () => {
  test("converts to the runtime's tensor set: router, 8 experts per layer, Q/K norms, untied head", () => {
    const r = checkFiles({ model: join(OUT, "model.smoe"), config: readFileSync(join(OUT, "config.ini")),
      tokenizer: readFileSync(join(OUT, "tokenizer.qbp")) });
    const s = QWEN3MOE_SHAPE;
    expect(r.container.count).toBe(3 + s.layers * (7 + 2 + 3 * s.experts));
    expect(readFileSync(join(OUT, "config.ini"), "utf8")).toContain("architecture=smolmoe_qwen3");
  });

  test("the runtime's logits follow the float64 reference at every position", () => {
    const trace = join(work.path, "c.trace"), idsFile = join(work.path, "ids.txt");
    const r = run(["--model", join(OUT, "model.smoe"), "--config", join(OUT, "config.ini"), "--tokenizer",
      join(OUT, "tokenizer.qbp"), "--prompt", TEXT, "--perplexity", "--trace", trace, "--token-ids", idsFile, "--threads", 2]);
    expect(r.code).toBe(0);
    const newIds = readFileSync(idsFile, "utf8").trim().split("\n").map(Number);
    const rows = readFileSync(trace, "utf8").trim().split("\n").map((line) => line.split(" ").slice(1).map(parseHexFloat));
    expect(rows.length).toBe(newIds.length);

    /* The converter renumbers the vocabulary; the reference speaks the checkpoint's own ids. */
    const tok = JSON.parse(readFileSync(join(SRC, "tokenizer.json"), "utf8"));
    const { oldIds } = qwen2Tokenizer(tok, QWEN3MOE_SHAPE.vocab, false);
    const store = WeightStore.open(SRC);
    const ref = forward(store, loadConfig(SRC), newIds.map((id) => oldIds[id]!));
    store.close();

    const V = QWEN3MOE_SHAPE.vocab;
    let agree = 0, klSum = 0;
    rows.forEach((c, t) => {
      const refRow = Float64Array.from({ length: V }, (_, id) => ref[t * V + oldIds[id]!]!);
      const p = logSoftmax(refRow), q = logSoftmax(c);
      let kl = 0, refBest = 0, cBest = 0;
      for (let i = 0; i < V; i++) {
        kl += Math.exp(p[i]!) * (p[i]! - q[i]!);
        if (refRow[i]! > refRow[refBest]!) refBest = i;
        if (c[i]! > c[cBest]!) cBest = i;
      }
      klSum += kl;
      if (refBest === cBest) agree++;
    });
    const meanKl = klSum / rows.length;
    if (process.env.SMOLMOE_SHOW) console.error(`qwen3moe: ${rows.length} positions, top-1 ${agree}/${rows.length}, mean KL ${meanKl}`);
    /* Measured: KL 5.0e-4 and 48/54. Mutants of the runtime fail both bounds: sigmoid routing (KL 0.056, 13/54),
     * no Q norm (0.0108, 38/54), top-k not renormalized (0.0093, 38/54). */
    expect(meanKl).toBeLessThan(0.002);
    expect(agree / rows.length).toBeGreaterThanOrEqual(0.8);
  }, 120_000);
});
