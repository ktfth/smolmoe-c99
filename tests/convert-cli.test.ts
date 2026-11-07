/**
 * scripts/convert.ts command line (exit codes and messages) and the pinned
 * downloader (verified rename, skip when present, refusal on a bad hash).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { fetchPinned } from "../scripts/hf/fetch";
import { ROOT, tempDir } from "./support/run";

const work = tempDir("convert-cli");
afterAll(() => work.dispose());

const cli = (...args: string[]) => {
  const r = Bun.spawnSync(["bun", join(ROOT, "scripts", "convert.ts"), ...args], { stdout: "pipe", stderr: "pipe" });
  return { code: r.exitCode, stdout: r.stdout.toString(), stderr: r.stderr.toString() };
};

describe("command line", () => {
  test.each([
    [[], /usage: bun scripts\/convert\.ts/],
    [["qwen2", "only-one-arg"], /usage/],
    [["qwen2", "a", "b", "--threads", "0"], /--threads must be 1\.\.64/],
    [["phi3", "a", "b", "--container", "v4"], /--container must be v2 or v3/],
    [["check", "m", "--bogus"], /bogus/],
    [["frobnicate"], /usage/],
  ])("%p is a usage error (exit 2)", (args, error) => {
    const r = cli(...args);
    expect(r.code).toBe(2);
    expect(r.stderr).toMatch(error);
  });

  test("runtime failures exit 1 with a convert: prefix", () => {
    const junk = join(work.path, "junk.smoe");
    writeFileSync(junk, "this is not a model file");
    const r = cli("check", junk, "--config", junk);
    expect(r.code).toBe(1);
    expect(r.stderr).toMatch(/^convert: unsupported checkpoint format/);
    const missing = cli("qwen2", join(work.path, "nowhere"), join(work.path, "out"));
    expect(missing.code).toBe(1);
    expect(missing.stderr).toMatch(/^convert: .*config\.json/);
  });
});

describe("pinned downloads", () => {
  const body = new TextEncoder().encode("weights");
  const digest = new Bun.CryptoHasher("sha256").update(body).digest("hex");
  const served: string[] = [];
  const fake = async (url: string) => {
    served.push(url);
    return new Response(url.endsWith("bad.bin") ? "tampered" : body);
  };

  test("downloads, verifies and renames; skips files already present and verified", async () => {
    const dir = join(work.path, "dl");
    const files = [{ name: "w.bin", revision: "abc123", sha256: digest }, { name: "LICENSE", revision: "main" }];
    const first = await fetchPinned("org/model", files, dir, () => {}, fake);
    expect(first.fetched).toEqual(["w.bin", "LICENSE"]);
    expect(served).toEqual(["https://huggingface.co/org/model/resolve/abc123/w.bin",
      "https://huggingface.co/org/model/resolve/main/LICENSE"]);
    expect(readFileSync(join(dir, "w.bin")).toString()).toBe("weights");
    expect((await fetchPinned("org/model", files, dir, () => {}, fake)).fetched).toEqual([]);
  });

  test("a hash mismatch leaves nothing behind", async () => {
    const dir = join(work.path, "dl-bad");
    await expect(fetchPinned("org/model", [{ name: "bad.bin", revision: "abc123", sha256: digest }], dir, () => {}, fake))
      .rejects.toThrow(/SHA-256/);
    expect(existsSync(join(dir, "bad.bin"))).toBe(false);
    expect(readdirSync(dir)).toEqual([]);
  });

  test.each([["../evil", "main"], ["ok.bin", "../main"]])("rejects unsafe names %p@%p", async (name, revision) => {
    await expect(fetchPinned("org/model", [{ name, revision }], join(work.path, "dl-x"), () => {}, fake)).rejects.toThrow(/invalid/);
  });

  test("rejects a malformed repo id", async () => {
    await expect(fetchPinned("not a repo", [], join(work.path, "dl-y"), () => {}, fake)).rejects.toThrow(/invalid repo/);
  });
});
