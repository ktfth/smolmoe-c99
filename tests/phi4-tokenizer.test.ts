/**
 * Phi-4-mini-instruct's o200k tokenizer and chat template against reference IDs
 * (tests/goldens/phi4-tokenizer.json). The SMQBP1 v2 file is built from the
 * Hugging Face tokenizer.json in SMOLMOE_PHI4_SRC (default downloads/phi-4-mini-instruct);
 * skipped when it is absent, as it is distributed separately from the source.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import golden from "./goldens/phi4-tokenizer.json";
import { BASE_CONFIG, configText } from "./support/fixture";
import { buildPhi4Qbp, hfToRuntime, PHI4_TOKENIZER_SHA256, PHI4_VOCAB, phi4Source } from "./support/phi4-qbp";
import { run, tempDir } from "./support/run";

interface GoldenCase {
  readonly text?: string;
  readonly system?: string;
  readonly user?: string;
  readonly reference_ids: readonly number[];
  readonly source: string;
}

const source = phi4Source();
const work = tempDir("phi4");
afterAll(() => work.dispose());

describe.skipIf(!source)("Phi-4-mini tokenizer (o200k, Phi chat template)", () => {
  let tokenizer = "", config = "", count = 0;

  beforeAll(() => {
    tokenizer = join(work.path, "tokenizer.qbp");
    config = join(work.path, "config.ini");
    writeFileSync(tokenizer, buildPhi4Qbp(source!));
    writeFileSync(config, configText({ ...BASE_CONFIG, vocab_size: PHI4_VOCAB, max_context: 8192 }));
  });

  const tokenize = (prompt: string, extra: readonly string[] = []) => {
    const input = join(work.path, `p${count}.txt`), ids = join(work.path, `ids${count++}.txt`);
    writeFileSync(input, prompt);
    const r = run(["--config", config, "--tokenizer", tokenizer, "--prompt-file", input, "--context", 4096,
      "--tokenize-only", "--token-ids", ids, ...extra]);
    if (r.code !== 0) throw new Error(`smolmoe failed: ${r.stderr}`);
    return { stdout: r.stdout.toString(), ids: readFileSync(ids, "utf8").trim().split("\n").map(Number) };
  };

  const cases = (golden as { cases: GoldenCase[] }).cases;

  test("the goldens are for the pinned tokenizer.json, and the v2 file has the expected layout", () => {
    expect(golden.tokenizer_sha256).toBe(PHI4_TOKENIZER_SHA256);
    expect(cases.filter((c) => c.text !== undefined).length).toBeGreaterThan(30);
    /* header + 200,064 records + 1,398,614 piece bytes + 199,742 merges */
    expect(readFileSync(tokenizer).length).toBe(32 + 8 * PHI4_VOCAB + 1_398_614 + 12 * 199_742);
  });

  test("text matches the reference IDs (remapped) and round-trips without normalization", () => {
    for (const c of cases.filter((x) => x.text !== undefined)) {
      const r = tokenize(c.text!);
      expect({ text: c.text, ids: r.ids }).toEqual({ text: c.text, ids: c.reference_ids.map(hfToRuntime) });
      expect(r.stdout).toBe(c.text!);
    }
  });

  test("--chat matches the hand-rendered Phi template, rstrip included", () => {
    for (const c of cases.filter((x) => x.user !== undefined)) {
      const r = tokenize(c.user!, ["--chat", "--system", c.system!]);
      expect({ chat: c, ids: r.ids }).toEqual({ chat: c, ids: c.reference_ids.map(hfToRuntime) });
    }
  });

  test("runtime IDs of the template: <|system|>S<|end|><|user|>U<|end|><|assistant|>", () => {
    expect(tokenize("U", ["--chat", "--system", "S"]).ids).toEqual([200022, 83, 257, 200021, 85, 257, 200020]);
  });
});
