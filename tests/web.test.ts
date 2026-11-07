/**
 * The browser build (make web) driven by the page's own host code
 * (web/wasm-host.ts) under bun: memory limits, worker threads, loading a
 * self-contained .smoe, and a tool chat that must match the native chat.
 * The cache logic runs against a fake Cache API; the DOM and a real
 * download were checked by hand (docs/WEB.md).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { readableNotice } from "../web/format";
import { cacheKey, type Fetch, forget, type ModelCache, openSource, remember, SourceError } from "../web/model-source";
import { EngineError, INITIAL_PAGES, MAX_MODEL_BYTES, MAXIMUM_PAGES, type PoolWorker, utf8Prefix, WasmEngine } from "../web/wasm-host";
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

describe("model source and cache", () => {
  const model = { name: "teste", file: "m.smoe", bytes: 20, sha256: "abc", url: "https://exemplo.test/m.smoe" };
  const bytes = Uint8Array.from({ length: 20 }, (_, i) => i);

  /* An in-memory Cache API. */
  function fakeCache() {
    const store = new Map<string, { body: Uint8Array; headers: Headers }>();
    const cache: ModelCache = {
      match: async (key) => {
        const hit = store.get(key);
        return hit ? new Response(hit.body, { headers: hit.headers }) : undefined;
      },
      put: async (key, response) => {
        store.set(key, { body: new Uint8Array(await response.arrayBuffer()), headers: response.headers });
      },
      delete: async (key) => store.delete(key),
    };
    return { cache, store };
  }

  const serve = (body: Uint8Array, length?: number, status = 200): Fetch => async () => {
    const headers = new Headers();
    if (length !== undefined) headers.set("content-length", String(length));
    return new Response(status === 200 ? body : null, { status, headers });
  };
  const failFetch: Fetch = async () => {
    throw new Error("fetch must not run");
  };
  const readAll = async (s: ReadableStream<Uint8Array>) => new Uint8Array(await new Response(s).arrayBuffer());

  test("a download is not cached by openSource; remember() keeps it and the next open reads the cache", async () => {
    const { cache, store } = fakeCache();
    const first = await openSource(model, cache, serve(bytes, 20));
    expect(first.cached).toBe(false);
    expect(await readAll(first.stream)).toEqual(bytes);
    expect(store.size).toBe(0);
    let reads = 0;
    await remember(model, cache, (offset, length) => {
      ++reads;
      return bytes.slice(offset, offset + length);
    });
    expect(reads).toBeGreaterThan(0);
    expect(store.has(cacheKey(model))).toBe(true);
    const second = await openSource(model, cache, failFetch);
    expect(second).toMatchObject({ cached: true, size: 20 });
    expect(await readAll(second.stream)).toEqual(bytes);
  });

  test("a cached copy of the wrong size is dropped and downloaded again", async () => {
    const { cache, store } = fakeCache();
    store.set(cacheKey(model), { body: bytes.slice(0, 5), headers: new Headers({ "content-length": "5" }) });
    const source = await openSource(model, cache, serve(bytes, 20));
    expect(source.cached).toBe(false);
    expect(store.size).toBe(0);
  });

  test("a server announcing another size is refused; no Content-Length falls back to the expected size", async () => {
    const { cache } = fakeCache();
    await expect(openSource(model, cache, serve(bytes, 19))).rejects.toThrow(SourceError);
    await expect(openSource(model, cache, serve(bytes, 20, 404))).rejects.toThrow("HTTP 404");
    expect((await openSource(model, cache, serve(bytes))).size).toBe(20);
  });

  test("forget() drops a cached copy, and a new SHA-256 is a new cache entry", async () => {
    const { cache, store } = fakeCache();
    await remember(model, cache, (offset, length) => bytes.slice(offset, offset + length));
    expect(cacheKey({ ...model, sha256: "def" })).not.toBe(cacheKey(model));
    await forget(model, cache);
    expect(store.size).toBe(0);
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
      expect(() => engine.allocate(MAX_MODEL_BYTES + 1)).toThrow(EngineError);
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
