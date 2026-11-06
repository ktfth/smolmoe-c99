/**
 * Untrusted input handling: malformed checkpoints, configs, tokenizers and
 * command lines must fail with a clear "smolmoe:" error, never crash.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { configText, type Fixture, writeFixture } from "./support/fixture";
import { run, runFails, runOk, tempDir } from "./support/run";

const work = tempDir("validation");
let base: Fixture;

beforeAll(() => {
  base = writeFixture(join(work.path, "base"));
});
afterAll(() => work.dispose());

function cli(f: { model: string; config: string; tokenizer: string }, extra: readonly (string | number)[] = []) {
  return ["--model", f.model, "--config", f.config, "--tokenizer", f.tokenizer, "--prompt", "Hi!", "--tokens", 4,
    "--context", 64, "--memory-mib", 4, ...extra];
}

function variant(name: string, mutate: (bytes: Buffer) => Buffer): { model: string; config: string; tokenizer: string } {
  const model = join(work.path, `${name}.smoe`);
  writeFileSync(model, mutate(Buffer.from(readFileSync(base.model))));
  return { model, config: base.config, tokenizer: base.tokenizer };
}

function expectError(args: readonly (string | number)[], pattern: RegExp): void {
  const stderr = runFails(args);
  expect(stderr).toStartWith("smolmoe: ");
  expect(stderr).toMatch(pattern);
}

describe("checkpoint", () => {
  test("truncated file is rejected", () => {
    const bad = variant("truncated", (b) => b.subarray(0, b.length - 100));
    expectError(cli(bad), /extent/);
  });

  test("bad magic is rejected", () => {
    const bad = variant("magic", (b) => { b[0] = 0x58; return b; });
    expectError(cli(bad), /unsupported checkpoint format/);
  });

  test("wrong tensor shape is rejected", () => {
    const bad = variant("shape", (b) => { b.writeUInt32LE(b.readUInt32LE(16 + 40) + 1, 16 + 40); return b; });
    expectError(cli(bad), /invalid tensor extent|wrong tensor shape/);
  });

  test("non-finite and negative Q8 scales are rejected at load", () => {
    const embed = base.tensors.find((t) => t.name === "embed")!;
    const nan = variant("nan", (b) => { b.writeFloatLE(Number.NaN, embed.offset + 36 * 3); return b; });
    expectError(cli(nan), /non-finite or negative scale in tensor embed/);
    const negative = variant("neg", (b) => { b.writeFloatLE(-1, embed.offset); return b; });
    expectError(cli(negative), /tensor embed/);
  });

  test("non-finite F32 norm weight is rejected", () => {
    const norm = base.tensors.find((t) => t.name === "final_norm")!;
    const bad = variant("norm", (b) => { b.writeFloatLE(Number.POSITIVE_INFINITY, norm.offset + 8); return b; });
    expectError(cli(bad), /final_norm/);
  });

  test("unsorted directory is rejected", () => {
    const bad = variant("unsorted", (b) => {
      const first = Buffer.from(b.subarray(16, 16 + 72));
      b.copy(b, 16, 16 + 72, 16 + 144);
      first.copy(b, 16 + 72);
      return b;
    });
    expectError(cli(bad), /sorted|extent/);
  });

  test("tensor count mismatch with config is rejected", () => {
    const config = join(work.path, "layers.ini");
    writeFileSync(config, configText({ ...base.settings, layers: 3 }));
    expectError(cli({ ...base, config }), /tensor count does not match config/);
  });
});

describe("config and tokenizer", () => {
  test("unknown keys, missing keys and unsupported architectures are rejected", () => {
    const write = (name: string, text: string) => {
      const p = join(work.path, name);
      writeFileSync(p, text);
      return p;
    };
    const text = configText(base.settings);
    expectError(cli({ ...base, config: write("unknown.ini", text + "extra=1\n") }), /unknown config key/);
    expectError(cli({ ...base, config: write("missing.ini", text.replace(/^dim=.*\n/m, "")) }), /missing required/);
    expectError(cli({ ...base, config: write("kimi.ini", text.replace("smolmoe_v1", "kimi_k3")) }), /unsupported architecture/);
    expectError(cli({ ...base, config: write("dup.ini", text + "dim=32\n") }), /duplicate config key/);
  });

  test("malformed BPE merges are rejected", () => {
    const tok = join(work.path, "bad.bpe");
    writeFileSync(tok, "SMBPE1\n72 105\n72 105\n");
    expectError(cli({ ...base, tokenizer: tok }), /duplicate BPE pair/);
    writeFileSync(tok, "SMBPE1\n256 105\n258 33\n");
    expectError(cli({ ...base, tokenizer: tok }), /special token/);
    writeFileSync(tok, "NOTBPE\n");
    expectError(cli({ ...base, tokenizer: tok }), /invalid tokenizer magic/);
  });
});

describe("command line", () => {
  test("invalid values are rejected", () => {
    for (const extra of [["--memory-mib", 0], ["--context", 513], ["--tokens", 80], ["--memory-mib", "-1"],
      ["--unknown", "x"], ["--tokens", "3junk"], ["--top-p", 0], ["--top-p", 1.5], ["--temperature", -1],
      ["--batch", 0], ["--threads", 9999], ["--seed", 0]] as const) {
      expect(run(cli(base, extra)).code).not.toBe(0);
    }
  });

  test("missing required arguments print usage", () => {
    const r = run(["--prompt", "x"]);
    expect(r.code).toBe(1);
    expect(r.stderr).toContain("usage:");
  });

  test("--version and --help succeed", () => {
    expect(run(["--version"]).stdout.toString()).toMatch(/^smolmoe \d+\.\d+\.\d+/);
    expect(run(["--help"]).code).toBe(0);
  });

  test("heap budget too small is rejected before the KV cache is allocated", () => {
    const config = join(work.path, "longctx.ini");
    writeFileSync(config, configText({ ...base.settings, max_context: 65536 }));
    expectError(cli({ ...base, config }, ["--context", 65536, "--memory-mib", 1]), /memory budget too small/);
  });

  test("prompt plus requested tokens must fit the context", () => {
    expectError(cli(base, ["--context", 16, "--tokens", 15]), /exceeds --context/);
    expectError(cli(base, ["--context", 8, "--prompt", "x".repeat(40)]), /prompt too long|exceeds --context/);
  });

  test("outputs may not overwrite inputs or each other", () => {
    const before = readFileSync(base.model);
    expectError(cli(base, ["--stats", base.model]), /aliases an input/);
    expectError(cli(base, ["--trace", base.config]), /aliases an input/);
    const out = join(work.path, "same.txt");
    expectError(cli(base, ["--stats", out, "--trace", out]), /alias each other/);
    expect(readFileSync(base.model).equals(before)).toBe(true);
  });

  test("missing files fail cleanly", () => {
    expectError(cli({ ...base, model: join(work.path, "nope.smoe") }), /cannot open/);
    expectError(cli({ ...base, tokenizer: join(work.path, "nope.bpe") }), /cannot open tokenizer/);
  });
});

describe("tokenizer round trip", () => {
  test("UTF-8 and arbitrary bytes, including NUL, survive encode/decode", () => {
    const input = join(work.path, "prompt.bin");
    const payload = Buffer.concat([Buffer.from("Hi! café 🌳 "), Buffer.from([0, 1, 2, 255, 128, 10])]);
    writeFileSync(input, payload);
    const r = run(["--config", base.config, "--tokenizer", base.tokenizer, "--prompt-file", input,
      "--context", 128, "--tokenize-only"]);
    expect(r.code).toBe(0);
    expect(r.stdout.equals(payload)).toBe(true);
  });

  test("empty prompt encodes to BOS only", () => {
    const ids = join(work.path, "empty.ids");
    const r = run(["--config", base.config, "--tokenizer", base.tokenizer, "--prompt", "", "--tokenize-only",
      "--token-ids", ids]);
    expect(r.code).toBe(0);
    expect(readFileSync(ids, "utf8")).toBe("256\n");
  });

  test("generation works with an empty prompt", () => {
    expect(runOk(cli(base, ["--prompt", ""])).stats.prompt_tokens).toBe(1);
  });
});
