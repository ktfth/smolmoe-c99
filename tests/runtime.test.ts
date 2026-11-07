/**
 * Runtime contract on synthetic MoE checkpoints: determinism across threads,
 * batch sizes and kernels, routing accounting, sampling and stats.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { type Fixture, type ModelConfig, configText, writeFixture } from "./support/fixture";
import { BINARY, SCALAR_BINARY, runFails, runOk, tempDir } from "./support/run";

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

describe("partial RoPE (rotary_dim, rope_attention_factor)", () => {
  /* medium: dim 64, 4 heads -> head_dim 16 */
  const withRope = (name: string, extra: Partial<ModelConfig>, run: readonly (string | number)[] = []) => {
    const config = join(work.path, `${name}.ini`);
    writeFileSync(config, configText({ ...medium.settings, ...extra }));
    return runOk([...args({ ...medium, config }, run), "--hash-logits"]).stats.logits_hash;
  };

  test("explicit defaults (all dims, factor 1) are bit-identical to omitting the keys", () => {
    expect(withRope("rope-default", { rotary_dim: 16, rope_attention_factor: 1 })).toBe(withRope("rope-none", {}));
  });

  test("rotating part of each head, or scaling cos/sin, changes the logits", () => {
    const base = withRope("rope-base", {});
    expect(withRope("rope-half", { rotary_dim: 8 })).not.toBe(base);
    expect(withRope("rope-factor", { rope_attention_factor: 1.1902380714238083 })).not.toBe(base);
  });

  test("partial RoPE stays bit-identical across threads, batch sizes and kernels", () => {
    const partial = { rotary_dim: 12, rope_attention_factor: 1.19 };
    const a = withRope("rope-a", partial, ["--threads", 1, "--batch", 1]);
    expect(withRope("rope-b", partial, ["--threads", 3, "--batch", 5])).toBe(a);
    expect(withRope("rope-c", partial, ["--threads", 2, "--batch", 32, "--scalar"])).toBe(a);
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

  test("--profile splits forward time into phases without changing the logits", () => {
    const plain = runOk(args(medium, ["--hash-logits"])).stats;
    const { stats } = runOk(args(medium, ["--hash-logits", "--profile"]));
    expect(stats.logits_hash).toBe(plain.logits_hash);
    expect(plain.profile).toBeUndefined();
    expect(Object.keys(stats.profile)).toEqual(["embed", "norm", "qkv", "rope", "attention", "attn_out", "ffn_up",
      "activation", "attn_quant", "ffn_down", "routing", "residual", "logits"]);
    const seconds = Object.values(stats.profile) as number[];
    seconds.forEach((s) => expect(s).toBeGreaterThanOrEqual(0));
    const total = seconds.reduce((a, b) => a + b, 0);
    expect(total).toBeGreaterThan(0);
    expect(total).toBeLessThanOrEqual(stats.prefill_seconds + stats.decode_seconds + 0.05);
  });

  test("without --memory-mib the heap budget is 512 MiB plus the KV cache; with it, exactly what was asked", () => {
    const base = ["--model", small.model, "--config", small.config, "--tokenizer", small.tokenizer, "--prompt", "Hi!",
      "--tokens", 2, "--context", 256];
    const auto = runOk(base).stats;
    expect(auto.budget_bytes).toBe(512 * 1048576 + auto.kv_bytes);
    expect(runOk([...base, "--memory-mib", 8]).stats.budget_bytes).toBe(8 * 1048576);
  });

  test("the default budget covers the KV cache only up to 2048 positions; a huge --context still fails cleanly", () => {
    /* medium: 2 layers x kv_dim 32 x 8 bytes = 512 B per position, so 2^20 positions need 512 MiB of KV alone */
    const config = join(work.path, "huge-context.ini");
    writeFileSync(config, configText({ ...medium.settings, max_context: 1048576 }));
    const stderr = runFails(["--model", medium.model, "--config", config, "--tokenizer", medium.tokenizer,
      "--prompt", "Hi!", "--tokens", 2, "--context", 1048576]);
    expect(stderr).toMatch(/memory budget too small/);
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

describe("perplexity", () => {
  test("scores every next token, generates nothing and is batch-invariant", () => {
    const score = (batch: number) => {
      const r = runOk(["--model", medium.model, "--config", medium.config, "--tokenizer", medium.tokenizer,
        "--prompt", "Hi! the quick brown fox jumps over the lazy dog", "--context", 128, "--perplexity",
        "--batch", batch]);
      return { stdout: r.stdout, stats: JSON.parse(r.stderr.trim().split("\n").at(-1)!) };
    };
    const a = score(32), b = score(3);
    expect(a.stdout.length).toBe(0);
    expect(a.stats.generated_tokens).toBe(0);
    expect(a.stats.perplexity_tokens).toBe(a.stats.prompt_tokens - 1);
    expect(a.stats.perplexity).toBeGreaterThan(1);
    expect(Math.abs(Math.exp(a.stats.nll) / a.stats.perplexity - 1)).toBeLessThan(1e-5);
    expect(b.stats.nll).toBe(a.stats.nll);
  });
});
