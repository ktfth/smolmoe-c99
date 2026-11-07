/**
 * The browser build (make web) driven by the page's own host code
 * (web/wasm-host.ts) under bun: memory limits, worker threads, loading a
 * self-contained .smoe, and a tool chat that must match the native chat.
 * The browser-only parts (download, cache, DOM) are checked by hand; see
 * docs/WEB.md.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { readableNotice } from "../web/format";
import { INITIAL_PAGES, MAXIMUM_PAGES, type PoolWorker, utf8Prefix, WasmEngine } from "../web/wasm-host";
import { BINARY, BUNDLE_BINARY, bundledModel, modelArgs, ROOT, run, runWithInput, tempDir } from "./support/run";

const WEB_WASM = join(ROOT, "build", "web", "smolmoe.wasm");
const built = existsSync(WEB_WASM);
const qwen = bundledModel("qwen2.5-0.5b-instruct", "tokenizer.qbp");
const SLOW = Number(process.env.SMOLMOE_SLOW_MS ?? 180_000);
const work = tempDir("web");
afterAll(() => work.dispose());

test("the browser build is present where CI expects it", () => {
  if (process.env.SM_EXPECT_WASM === "1") expect(built).toBe(true);
});

describe("page text", () => {
  test("a tool notice reads as the tool and its summary", () => {
    const line = '[ferramenta] calcular: {"name": "calcular", "arguments": {"expressao": "15% de 230"}} -> '
      + '{"resumo": "O resultado é 34,5.", "expressao": "15% de 230", "resultado": "34,5"}';
    expect(readableNotice(line)).toBe("ferramenta calcular: O resultado é 34,5.");
    expect(readableNotice('[ferramenta] dias_ate: {...} -> {"erro": "data inválida"}')).toBe("ferramenta dias_ate: erro: data inválida");
  });

  test("other notices, and results without a summary, show as they came", () => {
    expect(readableNotice("(contexto cheio: conversa reiniciada)")).toBe("(contexto cheio: conversa reiniciada)");
    const odd = '[ferramenta] data_atual: {} -> {"data": "6 de novembro"}';
    expect(readableNotice(odd)).toBe(odd);
    expect(readableNotice("[ferramenta] x: y -> {not json}")).toBe("[ferramenta] x: y -> {not json}");
  });

  test("a message cut to fit the engine never splits a character", () => {
    const bytes = new TextEncoder().encode("ação");
    expect(utf8Prefix(bytes, 100)).toBe(bytes.length);
    expect(utf8Prefix(bytes, 2)).toBe(1); /* "a" + half of "ç" -> "a" */
    expect(utf8Prefix(bytes, 3)).toBe(3);
  });
});

describe.skipIf(!built)("browser build", () => {
  test("the module's memory import accepts the host's shared memory", async () => {
    const module = new WebAssembly.Module(await Bun.file(WEB_WASM).arrayBuffer());
    const memory = WebAssembly.Module.imports(module).find((i) => i.kind === "memory") as unknown as
      { type: { minimum: number; maximum: number; shared: boolean } };
    expect(memory.type.shared).toBe(true);
    expect(memory.type.minimum).toBeLessThanOrEqual(INITIAL_PAGES);
    expect(memory.type.maximum).toBe(MAXIMUM_PAGES);
  });

  test.skipIf(!qwen)("Qwen tool chat through the page's host: the same reply as the native chat", async () => {
    const single = join(work.path, "qwen.smoe");
    expect(run([...modelArgs(qwen!), "--out", single], BUNDLE_BINARY).code).toBe(0);
    const module = new WebAssembly.Module(await Bun.file(WEB_WASM).arrayBuffer());
    const pool = new URL("../web/pool-worker.ts", import.meta.url);
    const engine = await WasmEngine.create(module, 4, () => new Worker(pool) as unknown as PoolWorker);
    try {
      const file = Bun.file(single);
      const ptr = engine.allocate(file.size);
      let offset = 0;
      for await (const chunk of file.stream()) {
        engine.fill(ptr, offset, chunk);
        offset += chunk.length;
      }
      engine.open(ptr, file.size, 2048);
      expect(engine.backend()).toBe("simd128");
      expect(engine.stats().threads).toBe(4);

      let reply = "";
      const notices: string[] = [];
      const ok = engine.send("Quanto é 15% de 230?", { text: (s) => { reply += s; }, notice: (l) => notices.push(l) },
        new Date(2026, 10, 6, 10, 30));
      expect(ok).toBe(true);
      expect(notices.some((l) => l.startsWith("[ferramenta] calcular"))).toBe(true);

      const native = runWithInput([...modelArgs(qwen!), "-i", "--now", "2026-11-06T10:30", "--tokens", 512, "--context", 2048],
        "Quanto é 15% de 230?\n/sair\n", BINARY);
      expect(reply).toBe(native.stdout.toString());
    } finally {
      engine.dispose();
    }
  }, SLOW);
});
