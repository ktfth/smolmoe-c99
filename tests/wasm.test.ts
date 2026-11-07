/**
 * The WebAssembly build (make wasi) against the native binary: the SIMD128
 * kernels are bit-identical to the scalar ones, and the runtime gives the
 * same logits and the same chat as the native build. The browser build runs
 * the same C code, so these are the browser's numbers too. Skipped when
 * build/wasi has not been built, unless SM_EXPECT_WASM=1 (CI).
 */
import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { BINARY, bundledModel, modelArgs, ROOT, runWithInput, type RunResult, type Stats } from "./support/run";
import { guestArgs } from "./support/wasi-run";

const WASI_DIR = join(ROOT, "build", "wasi");
const RUNTIME = join(WASI_DIR, "smolmoe.wasm");
const KERNEL_TESTS = join(WASI_DIR, "test-kernels.wasm");
const THREADED = join(WASI_DIR, "smolmoe-mt.wasm"); /* shared memory, worker instances (the browser build) */
const RUNNER = join(ROOT, "tests", "support", "wasi-run.ts");
const built = existsSync(RUNTIME) && existsSync(KERNEL_TESTS) && existsSync(THREADED);
const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
const stories = bundledModel("tinystories-15m", "tokenizer.spb");
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);

/* Runs a module through the WASI host, like a native binary (4 worker instances for a threaded one). */
function wasm(module: string, args: readonly (string | number)[], input = ""): RunResult {
  const proc = Bun.spawnSync([process.execPath, RUNNER, module, ...args.map(String)], {
    stdin: new TextEncoder().encode(input),
    stdout: "pipe",
    stderr: "pipe",
    env: { ...process.env, SM_EXPECT_SIMD: "1", SMOLMOE_WASM_THREADS: "4" },
  });
  return { code: proc.exitCode ?? -1, stdout: Buffer.from(proc.stdout), stderr: proc.stderr.toString() };
}

function stats(r: RunResult): Stats {
  if (r.code !== 0) throw new Error(`failed (${r.code}): ${r.stderr}`);
  return JSON.parse(r.stderr.trim().split("\n").at(-1) ?? "") as Stats;
}

/* stderr without the stats line, whose timings differ between runs. Native
 * Windows writes stderr in text mode (CRLF). */
function notices(r: RunResult): string {
  return r.stderr.replaceAll("\r\n", "\n").split("\n").filter((line) => !line.includes('{"version"')).join("\n");
}

test("the WebAssembly build is present where CI expects it", () => {
  if (process.env.SM_EXPECT_WASM === "1") expect(built).toBe(true);
});

test("the WASI runner maps path arguments only, never a filesystem root", () => {
  const model = join(ROOT, "models", "x", "model.smoe");
  const mapped = guestArgs(["--prompt", "/sair", "--model", join(ROOT, "package.json"), "--out", model, "--dir", ROOT]);
  expect(mapped.args).toEqual(["--prompt", "/sair", "--model", "/host0/package.json", "--out", model, "--dir", "/host0"]);
  expect(mapped.preopens).toEqual({ "/host0": ROOT });
});

describe.skipIf(!built)("WebAssembly build", () => {
  test("SIMD128 kernels are selected and bit-identical to the scalar ones", () => {
    const r = wasm(KERNEL_TESTS, []);
    expect(r.stderr).toBe("");
    expect(r.code).toBe(0);
    expect(r.stdout.toString()).toContain("kernel tests passed");
  }, SLOW);

  test.skipIf(!stories)("TinyStories: the same logits as the native build, with SIMD128 and scalar", () => {
    const args = [...modelArgs(stories!), "--prompt", "Once upon a time", "--tokens", 32, "--hash-logits"];
    const native = stats(runWithInput(args, "", BINARY));
    const simd = stats(wasm(RUNTIME, args));
    const scalar = stats(wasm(RUNTIME, [...args, "--scalar"]));
    expect(simd.backend).toBe("simd128");
    expect(scalar.backend).toBe("scalar");
    expect(simd.logits_hash).toBe(native.logits_hash);
    expect(scalar.logits_hash).toBe(native.logits_hash);
  }, SLOW);

  test.skipIf(!stories)("TinyStories: the threaded module gives the same logits with 1, 2 and 4 threads", () => {
    const args = [...modelArgs(stories!), "--prompt", "Once upon a time", "--tokens", 32, "--hash-logits"];
    const native = stats(runWithInput(args, "", BINARY)).logits_hash;
    for (const threads of [1, 2, 4]) {
      const s = stats(wasm(THREADED, [...args, "--threads", threads]));
      expect(s.threads).toBe(threads);
      expect(s.logits_hash).toBe(native);
    }
  }, SLOW);

  test.skipIf(!qwen)("Qwen: the same logits as the native build", () => {
    const args = [...modelArgs(qwen!), "--chat", "--prompt", "Qual é a capital do Brasil?", "--tokens", 8,
      "--hash-logits"];
    const native = stats(runWithInput(args, "", BINARY)).logits_hash;
    expect(stats(wasm(RUNTIME, args)).logits_hash).toBe(native);
    expect(stats(wasm(THREADED, [...args, "--threads", 4])).logits_hash).toBe(native);
  }, SLOW);

  test.skipIf(!qwen)("Qwen chat with a tool call, threaded module: the same reply and notices as the native build", () => {
    const args = [...modelArgs(qwen!), "-i", "--now", "2026-11-06T10:30", "--tokens", 48];
    const input = "Quanto é 15% de 230?\n/sair\n";
    const native = runWithInput(args, input, BINARY);
    const web = wasm(THREADED, [...args, "--threads", 4], input);
    expect(web.code).toBe(0);
    expect(native.stderr).toContain("[ferramenta] calcular");
    expect(web.stdout.toString()).toBe(native.stdout.toString());
    expect(notices(web)).toBe(notices(native));
  }, SLOW);
});
