/**
 * Single-file models (SMOEQ8 v3 with embedded config/tokenizer) and
 * self-contained executables (runtime + appended container).
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { copyFileSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { configText, type Fixture, writeFixture } from "./support/fixture";
import { BINARY, BUNDLE_BINARY, EXE_SUFFIX, run, runFails, runOk, tempDir } from "./support/run";

const work = tempDir("bundle");
let fx: Fixture;
let single: string;
let app: string;

const PROMPT = ["--prompt", "Hi! bundled", "--tokens", 12, "--context", 128, "--memory-mib", 8, "--hash-logits"];

beforeAll(() => {
  fx = writeFixture(join(work.path, "fx"), { config: { experts: 4 }, seed: 5 });
  single = join(work.path, "single.smoe");
  app = join(work.path, `app${EXE_SUFFIX}`);
  for (const args of [
    ["--model", fx.model, "--config", fx.config, "--tokenizer", fx.tokenizer, "--out", single],
    ["--model", single, "--exe", BINARY, "--out", app],
  ]) {
    const r = run(args, BUNDLE_BINARY);
    if (r.code !== 0) throw new Error(r.stderr);
  }
});
afterAll(() => work.dispose());

const threeFiles = () => runOk(["--model", fx.model, "--config", fx.config, "--tokenizer", fx.tokenizer, ...PROMPT]);

describe("single-file model", () => {
  test("runs with --model only and matches the three-file run bit for bit", () => {
    const reference = threeFiles();
    const bundled = runOk(["--model", single, ...PROMPT]);
    expect(bundled.stats.logits_hash).toBe(reference.stats.logits_hash);
    expect(bundled.stdout.equals(reference.stdout)).toBe(true);
  });

  test("explicit --config and --tokenizer override the embedded copies", () => {
    const other = join(work.path, "short.ini");
    writeFileSync(other, configText({ ...fx.settings, max_context: 64 }));
    expect(runFails(["--model", single, "--config", other, ...PROMPT])).toMatch(/exceeds the model's max_context/);
  });

  test("a model without embedded config asks for --config", () => {
    expect(runFails(["--model", fx.model, ...PROMPT])).toMatch(/no config: pass --config/);
  });

  test("blob records are rejected in containers older than v3", () => {
    const bytes = Buffer.from(readFileSync(single));
    bytes[7] = 2;
    const bad = join(work.path, "v2-with-blobs.smoe");
    writeFileSync(bad, bytes);
    expect(runFails(["--model", bad, ...PROMPT])).toMatch(/invalid blob record/);
  });

  test("bundling refuses a config that does not match the weights", () => {
    const wrong = join(work.path, "wrong.ini");
    writeFileSync(wrong, configText({ ...fx.settings, layers: 3 }));
    const stderr = runFails(["--model", fx.model, "--config", wrong, "--tokenizer", fx.tokenizer,
      "--out", join(work.path, "never.smoe")], BUNDLE_BINARY);
    expect(stderr).toMatch(/tensor count does not match config/);
  });
});

describe("self-contained executable", () => {
  test("runs with no model files and matches the three-file run", () => {
    const reference = threeFiles();
    const bundled = runOk([...PROMPT], app);
    expect(bundled.stats.logits_hash).toBe(reference.stats.logits_hash);
    expect(bundled.stdout.equals(reference.stdout)).toBe(true);
  });

  test("tokenize-only works from the embedded tokenizer", () => {
    const r = run(["--prompt", "Hi! café", "--tokenize-only"], app);
    expect(r.code).toBe(0);
    expect(r.stdout.toString()).toBe("Hi! café");
  });

  test("re-bundling an executable keeps only its runtime part", () => {
    const again = join(work.path, `again${EXE_SUFFIX}`);
    expect(run(["--model", single, "--exe", app, "--out", again], BUNDLE_BINARY).code).toBe(0);
    expect(readFileSync(again).length).toBe(readFileSync(app).length);
    expect(runOk([...PROMPT], again).stats.logits_hash).toBe(threeFiles().stats.logits_hash);
  });

  test("a corrupt trailer is reported, not ignored", () => {
    const bytes = Buffer.from(readFileSync(app));
    bytes.writeBigUInt64LE(BigInt(12345), bytes.length - 24);
    const broken = join(work.path, `broken${EXE_SUFFIX}`);
    writeFileSync(broken, bytes);
    expect(runFails([...PROMPT], broken)).toMatch(/corrupt bundle trailer/);
  });

  test("an executable without a bundle still requires --model", () => {
    const plain = join(work.path, `plain${EXE_SUFFIX}`);
    copyFileSync(BINARY, plain);
    const r = run(["--prompt", "x"], plain);
    expect(r.code).toBe(1);
    expect(r.stderr).toContain("usage:");
  });

  test("chat mode explains that a plain-text model cannot chat", () => {
    expect(runFails([], app)).toMatch(/chat needs a chat model/);
  });
});
