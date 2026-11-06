/**
 * Runtime contract on synthetic MoE checkpoints: determinism across threads,
 * batch sizes and kernels, routing accounting, sampling and stats.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { type Fixture, writeFixture } from "./support/fixture";
import { BINARY, SCALAR_BINARY, runOk, tempDir } from "./support/run";

const work = tempDir("runtime");
let small: Fixture;
let medium: Fixture;

beforeAll(() => {
  small = writeFixture(join(work.path, "small"));
  medium = writeFixture(join(work.path, "medium"), { config: { dim: 64, hidden_dim: 96, experts: 64 }, seed: 7 });
});
afterAll(() => work.dispose());

function args(f: Fixture, extra: readonly (string | number)[] = []): (string | number)[] {
  return ["--model", f.model, "--config", f.config, "--tokenizer", f.tokenizer,
    "--prompt", "Hi! routing test", "--tokens", 24, "--context", 128, "--memory-mib", 8, ...extra];
}

describe("determinism", () => {
  test("logits, tokens and text are bit-identical across threads, batch sizes and kernels", () => {
    const configs: [string, (string | number)[]][] = [
      [BINARY, ["--threads", 1, "--batch", 1]],
      [BINARY, ["--threads", 2, "--batch", 3]],
      [BINARY, ["--threads", 4, "--batch", 32]],
      [BINARY, ["--threads", 3, "--batch", 5, "--scalar"]],
      [SCALAR_BINARY, ["--threads", 2, "--batch", 7]],
    ];
    let expected: string | null = null;
    configs.forEach(([binary, extra], i) => {
      const trace = join(work.path, `trace-${i}`);
      const ids = join(work.path, `ids-${i}`);
      const r = runOk(args(medium, [...extra, "--trace", trace, "--token-ids", ids]), binary);
      const observed = [r.stdout.toString("hex"), readFileSync(trace, "utf8"), readFileSync(ids, "utf8")].join("|");
      expected ??= observed;
      expect(observed).toBe(expected);
    });
  });

  test("logit hash is reproducible and covers every processed token", () => {
    const a = runOk(args(small, ["--hash-logits"])).stats;
    const b = runOk(args(small, ["--hash-logits", "--threads", 1, "--batch", 2])).stats;
    expect(a.logits_hash).toMatch(/^[0-9a-f]{16}$/);
    expect(a.logits_hash).not.toBe("0000000000000000");
    expect(b.logits_hash).toBe(a.logits_hash);
  });
});

describe("routing", () => {
  test("expert calls equal processed tokens x (top_k + shared) x layers", () => {
    const { stats } = runOk(args(medium));
    const c = medium.settings;
    expect(stats.expert_calls).toBe(stats.processed_tokens * (c.top_k + c.shared_experts) * c.layers);
  });

  test("dense configuration (one expert, no shared) runs", () => {
    const dir = join(work.path, "dense");
    const dense = writeFixture(dir, { config: { experts: 1, top_k: 1, shared_experts: 0 } });
    const { stats } = runOk(args(dense));
    expect(stats.expert_calls).toBe(stats.processed_tokens * dense.settings.layers);
  });
});

describe("sampling", () => {
  test("a fixed seed reproduces the same sample; different seeds diverge", () => {
    const sample = (seed: number, threads: number) =>
      runOk(args(medium, ["--temperature", 1.5, "--top-k", 0, "--seed", seed, "--threads", threads, "--tokens", 40])).stdout.toString("hex");
    expect(sample(42, 1)).toBe(sample(42, 4));
    const outputs = new Set([42, 7, 1234, 99].map((seed) => sample(seed, 2)));
    expect(outputs.size).toBeGreaterThan(1);
  });

  test("top-k 1 sampling equals greedy decoding", () => {
    const greedy = runOk(args(medium)).stdout;
    const topk1 = runOk(args(medium, ["--temperature", 0.9, "--top-k", 1])).stdout;
    expect(topk1.equals(greedy)).toBe(true);
  });

  test("top-p and top-k combinations run deterministically", () => {
    const run = () => runOk(args(medium, ["--temperature", 0.8, "--top-k", 10, "--top-p", 0.5, "--seed", 3])).stdout;
    expect(run().equals(run())).toBe(true);
  });
});

describe("statistics", () => {
  test("stats report heap within budget and mapped weights separately", () => {
    const statsFile = join(work.path, "stats.json");
    const r = runOk(args(small, ["--stats", statsFile]));
    const fromFile = JSON.parse(readFileSync(statsFile, "utf8"));
    expect(fromFile.logits_hash).toBe(r.stats.logits_hash);
    expect(r.stats.tracked_peak_bytes).toBeLessThanOrEqual(r.stats.budget_bytes);
    expect(r.stats.mapped_bytes).toBeGreaterThan(0);
    expect(r.stats.prompt_tokens).toBeGreaterThan(1);
    /* The last emitted token is fed back only if generation stopped at EOS. */
    const fed = r.stats.processed_tokens - r.stats.prompt_tokens;
    expect([r.stats.generated_tokens - 1, r.stats.generated_tokens]).toContain(fed);
    expect(r.stats.version).toMatch(/^\d+\.\d+\.\d+$/);
  });

  test("token-id file holds BOS, prompt and generated IDs", () => {
    const ids = join(work.path, "ids.txt");
    runOk(["--model", small.model, "--config", small.config, "--tokenizer", small.tokenizer,
      "--prompt", "Hi!", "--tokens", 3, "--token-ids", ids]);
    const list = readFileSync(ids, "utf8").trim().split("\n").map(Number);
    expect(list.slice(0, 2)).toEqual([256, 259]);
    expect(list.length).toBeGreaterThanOrEqual(3);
  });
});
