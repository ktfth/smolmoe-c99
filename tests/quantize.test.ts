/**
 * smolmoe-quantize: Q8 -> Q5 rewrite. The Q5 checkpoint must keep the
 * determinism contract, shrink the payload and stay close to the Q8 model.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { parseHexFloat } from "../scripts/compare-traces";
import { type Fixture, writeFixture } from "./support/fixture";
import { BINARY, QUANTIZE_BINARY, SCALAR_BINARY, run, runFails, runOk, tempDir } from "./support/run";

const work = tempDir("quantize");
let q8: Fixture;
let q5Model: string;

beforeAll(() => {
  q8 = writeFixture(join(work.path, "q8"), { config: { dim: 64, hidden_dim: 96, experts: 16 }, seed: 11 });
  q5Model = join(work.path, "q5.smoe");
  const r = run(["--in", q8.model, "--out", q5Model], QUANTIZE_BINARY);
  if (r.code !== 0) throw new Error(r.stderr);
});
afterAll(() => work.dispose());

const args = (model: string, extra: readonly (string | number)[] = []) =>
  ["--model", model, "--config", q8.config, "--tokenizer", q8.tokenizer, "--prompt", "Hi! quantized",
    "--tokens", 16, "--context", 128, "--memory-mib", 8, ...extra];

describe("Q5 checkpoints", () => {
  test("payload shrinks to 22/36 of Q8 for converted matrices", () => {
    expect(statSync(q5Model).size).toBeLessThan(statSync(q8.model).size * 0.7);
  });

  test("logits are bit-identical across threads, batch sizes and kernels", () => {
    const traces = [
      [BINARY, ["--threads", 1, "--batch", 1]],
      [BINARY, ["--threads", 4, "--batch", 9]],
      [BINARY, ["--threads", 2, "--batch", 32, "--scalar"]],
      [SCALAR_BINARY, ["--threads", 3, "--batch", 4]],
    ].map(([binary, extra], i) => {
      const trace = join(work.path, `q5-trace-${i}`);
      runOk(args(q5Model, [...(extra as (string | number)[]), "--trace", trace]), binary as string);
      return readFileSync(trace, "utf8");
    });
    expect(new Set(traces).size).toBe(1);
  });

  test("Q5 logits stay strongly correlated with the Q8 model", () => {
    const parse = (path: string) => readFileSync(path, "utf8").trim().split("\n")
      .flatMap((line) => line.split(" ").slice(1).map(parseHexFloat));
    const t8 = join(work.path, "t8"), t5 = join(work.path, "t5");
    runOk(args(q8.model, ["--trace", t8, "--tokens", 1]));
    runOk(args(q5Model, ["--trace", t5, "--tokens", 1]));
    const a = parse(t8), b = parse(t5);
    const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
    const ma = mean(a), mb = mean(b);
    let cov = 0, va = 0, vb = 0;
    a.forEach((x, i) => { cov += (x - ma) * (b[i] - mb); va += (x - ma) ** 2; vb += (b[i] - mb) ** 2; });
    const correlation = cov / Math.sqrt(va * vb);
    expect(correlation).toBeGreaterThan(0.9);
    expect(correlation).toBeLessThan(1); /* the formats really differ */
  });

  test("corrupt Q5 scale is rejected at load", () => {
    const bytes = Buffer.from(readFileSync(q5Model));
    const firstQ5 = bytes.readUInt32LE(16 + 48) === 2 ? Number(bytes.readBigUInt64LE(16 + 56)) : -1;
    expect(firstQ5).toBeGreaterThan(0);
    bytes.writeUInt16LE(0x7e00, firstQ5); /* f16 NaN */
    const bad = join(work.path, "bad-q5.smoe");
    writeFileSync(bad, bytes);
    expect(runFails(args(bad))).toMatch(/invalid scale in tensor embed/);
  });

  test("the converter refuses to overwrite its input and validates options", () => {
    expect(runFails(["--in", q8.model, "--out", q8.model], QUANTIZE_BINARY)).toMatch(/must differ/);
    expect(run(["--in", q8.model], QUANTIZE_BINARY).code).not.toBe(0);
    expect(runFails(["--in", q8.model, "--out", join(work.path, "x.smoe"), "--simulate-bits", 2], QUANTIZE_BINARY))
      .toMatch(/simulate-bits/);
  });
});
