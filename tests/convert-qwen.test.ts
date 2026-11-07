/**
 * Golden: converting the pinned Qwen2.5-0.5B-Instruct source with the TS
 * converter reproduces the shipped files byte for byte (SHA-256 of
 * models/manifest.json, versioned config.ini / provenance.json / licence).
 * Runs only when the source is present: `bun scripts/convert.ts fetch-qwen2
 * downloads/qwen2.5-0.5b-instruct-src` (or SMOLMOE_QWEN_SOURCE=DIR).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";

import { sha256File } from "../scripts/hf/hash";
import { BINARY, BUNDLE_BINARY, ROOT, run, runOk, tempDir } from "./support/run";

const SOURCE = resolve(ROOT, process.env.SMOLMOE_QWEN_SOURCE ?? join("downloads", "qwen2.5-0.5b-instruct-src"));
const SHIPPED = join(ROOT, "models", "qwen2.5-0.5b-instruct");
const present = existsSync(join(SOURCE, "model.safetensors")) && existsSync(join(SOURCE, "tokenizer.json"));

interface ManifestFile {
  readonly path: string;
  readonly bytes: number;
  readonly sha256: string;
}
const manifest = JSON.parse(readFileSync(join(ROOT, "models", "manifest.json"), "utf8")) as { files: ManifestFile[] };
const golden = (name: string) => manifest.files.find((f) => f.path === `qwen2.5-0.5b-instruct/${name}`)!;

const convert = (...args: string[]) =>
  Bun.spawnSync(["bun", join(ROOT, "scripts", "convert.ts"), ...args], { stdout: "pipe", stderr: "pipe" });

describe.skipIf(!present)("Qwen2.5-0.5B-Instruct golden conversion", () => {
  const work = tempDir("convert-qwen");
  const out = join(work.path, "out");
  afterAll(() => work.dispose());

  /* Runs first (tests run in order); the rest compare its output. */
  test("bun scripts/convert.ts qwen2 converts the pinned source", () => {
    const r = convert("qwen2", SOURCE, out);
    expect(r.stderr.toString()).toContain("done in");
    expect(r.exitCode).toBe(0);
  }, 600_000);

  test.each(["model.smoe", "tokenizer.qbp"])("%s matches the manifest SHA-256", (name) => {
    const g = golden(name);
    expect(readFileSync(join(out, name)).length).toBe(g.bytes);
    expect(sha256File(join(out, name))).toBe(g.sha256);
  });

  test.each(["config.ini", "provenance.json", "LICENSE-Qwen.txt"])("%s equals the versioned file", (name) => {
    expect(readFileSync(join(out, name)).equals(readFileSync(join(SHIPPED, name)))).toBe(true);
  });

  test("the TS checker accepts the output", () => {
    const r = convert("check", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
      join(out, "tokenizer.qbp"));
    expect(r.exitCode).toBe(0);
    expect(JSON.parse(r.stdout.toString())).toMatchObject({ version: 2, records: 315, blobs: 0,
      tokenizer: { kind: "SMQBP1", version: 1, merges: 151387 } });
  });

  test.skipIf(!existsSync(BINARY))("the runtime loads the converted model", () => {
    const r = runOk(["--model", join(out, "model.smoe"), "--config", join(out, "config.ini"), "--tokenizer",
      join(out, "tokenizer.qbp"), "--prompt", "Olá", "--tokens", 2, "--context", 64, "--temperature", 0]);
    expect(r.stats.generated_tokens).toBeGreaterThan(0);
  }, 120_000);

  test.skipIf(!existsSync(BUNDLE_BINARY))("the TS single-file bundle equals smolmoe-bundle's", () => {
    const mine = join(work.path, "mine.smoe"), theirs = join(work.path, "theirs.smoe");
    const files = [join(out, "model.smoe"), join(out, "config.ini"), join(out, "tokenizer.qbp")];
    expect(convert("bundle", ...files, mine).exitCode).toBe(0);
    expect(run(["--model", files[0]!, "--config", files[1]!, "--tokenizer", files[2]!, "--out", theirs], BUNDLE_BINARY).code).toBe(0);
    expect(sha256File(mine)).toBe(sha256File(theirs));
  }, 120_000);
});
