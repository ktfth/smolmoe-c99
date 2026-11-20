/**
 * --prefix-cache: the processed system prompt kept in a file. A later chat
 * reads it instead of running the prefill, and the replies are the same with
 * and without it. A cache is used only after the last prefix position,
 * computed again over it, gives the logits stored with it: a damaged file,
 * another system prompt or another model of the same shape gets it rebuilt.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { BINARY, bundledModel, modelArgs, run, runWithInput, tempDir } from "./support/run";

const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);
const NOW = ["--now", "2026-11-06T10:30"];
const QUESTION = "Qual é a capital da França?\n/sair\n";

interface ChatStats {
  readonly prefill_seconds: number;
  readonly prefix_cache?: string;
}

describe.skipIf(!qwen)("--prefix-cache", () => {
  const work = tempDir("prefix-cache");
  afterAll(() => work.dispose());

  const chat = (extra: readonly (string | number)[], model = qwen!) => {
    const r = runWithInput([...modelArgs(model), ...NOW, "--tokens", 8, ...extra], QUESTION);
    expect(r.code).toBe(0);
    /* The stats line follows the last "> " prompt on the same line. */
    const at = r.stderr.lastIndexOf('{"version"');
    expect(at).toBeGreaterThanOrEqual(0);
    const stats = JSON.parse(r.stderr.slice(at).split(/\r?\n/)[0]!) as ChatStats;
    return { reply: r.stdout.toString(), stderr: r.stderr, stats };
  };

  test("the second chat reads the cache, skips the prefill and answers exactly as without a cache", () => {
    const file = join(work.path, "qwen.kv");
    const plain = chat([]);
    expect(plain.stats.prefix_cache).toBeUndefined();
    const first = chat(["--prefix-cache", file]);
    expect(first.stats.prefix_cache).toBe("gravado");
    expect(existsSync(file)).toBe(true);
    const second = chat(["--prefix-cache", file]);
    expect(second.stats.prefix_cache).toBe("carregado");
    expect(first.reply).toBe(plain.reply);
    expect(second.reply).toBe(plain.reply);
    /* The system prompt with the tools is hundreds of tokens; loading it costs one token's forward. */
    expect(second.stats.prefill_seconds).toBeLessThan(first.stats.prefill_seconds / 2);
  }, SLOW);

  /* Without the tools the system prompt is short, so these chats take seconds; what is checked is the same. */
  test("a damaged cache, another system prompt or another model of the same shape is rebuilt, never used", () => {
    const file = join(work.path, "check.kv");
    const plain = chat(["--no-tools"]);
    expect(chat(["--no-tools", "--prefix-cache", file]).stats.prefix_cache).toBe("gravado");

    /* One flipped byte in the middle of the keys and values. */
    const bytes = readFileSync(file);
    bytes[Math.floor(bytes.length / 2)]! ^= 0x40;
    writeFileSync(file, bytes);
    const damaged = chat(["--no-tools", "--prefix-cache", file]);
    expect(damaged.stats.prefix_cache).toBe("refeito");
    expect(damaged.stderr).toContain("não conferiu");
    expect(damaged.reply).toBe(plain.reply);
    expect(chat(["--no-tools", "--prefix-cache", file]).stats.prefix_cache).toBe("carregado");

    expect(chat(["--no-tools", "--prefix-cache", file, "--system", "Responda em uma frase."]).stats.prefix_cache).toBe("refeito");

    const q5 = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
    const q5Model = q5 && join(q5.model, "..", "model-q5.smoe");
    if (q5 && q5Model && existsSync(q5Model)) {
      expect(chat(["--no-tools", "--prefix-cache", file]).stats.prefix_cache).toBe("refeito"); /* back to the Q8 prompt */
      /* Same layers, kv_dim, vocabulary and prompt tokens; other weights. */
      expect(chat(["--no-tools", "--prefix-cache", file], { ...q5, model: q5Model }).stats.prefix_cache).toBe("refeito");
    }
  }, SLOW);

  /* --kv-f16 halves the cache's elements: a cache of one width is never read as the other. */
  test("a float16 KV cache is kept and read in float16, and a float32 chat rebuilds it", () => {
    const file = join(work.path, "f16.kv");
    const plain = chat(["--no-tools", "--kv-f16"]);
    expect(chat(["--no-tools", "--kv-f16", "--prefix-cache", file]).stats.prefix_cache).toBe("gravado");
    const loaded = chat(["--no-tools", "--kv-f16", "--prefix-cache", file]);
    expect(loaded.stats.prefix_cache).toBe("carregado");
    expect(loaded.reply).toBe(plain.reply);
    const f32 = chat(["--no-tools", "--prefix-cache", file]);
    expect(f32.stats.prefix_cache).toBe("refeito");
    expect(f32.reply).toBe(chat(["--no-tools"]).reply);
  }, SLOW);

  test.skipIf(process.platform === "win32" || !Bun.which("mkfifo"))("a FIFO as the cache never hangs the chat", () => {
    /* Opening a FIFO with no writer blocks forever unless the open is non-blocking. */
    const fifo = join(work.path, "cache.fifo");
    expect(Bun.spawnSync(["mkfifo", fifo]).exitCode).toBe(0);
    const r = Bun.spawnSync([BINARY, ...modelArgs(qwen!), "--no-tools", "--tokens", 4, "--prefix-cache", fifo].map(String),
      { stdin: new TextEncoder().encode(QUESTION), stdout: "pipe", stderr: "pipe", timeout: 60_000 });
    expect(r.exitCode).toBe(0);
  }, SLOW);

  test("--prefix-cache is for the chat, and may not overwrite an input", () => {
    const one = run([...modelArgs(qwen!), "--prompt", "Olá", "--prefix-cache", join(work.path, "x.kv")]);
    expect(one.code).not.toBe(0);
    expect(one.stderr).toMatch(/--prefix-cache is for the interactive chat/);
    const alias = runWithInput([...modelArgs(qwen!), "--prefix-cache", qwen!.config], "/sair\n");
    expect(alias.code).not.toBe(0);
    expect(alias.stderr).toMatch(/aliases an input/);
  });
});
