/* The page's engine worker: owns the WebAssembly engine, so the page never
 * blocks. Streams a model straight into the module's memory (from the
 * browser's cache, a download, or a file the user picked), opens it, keeps a
 * downloaded model in the cache once it opened, then answers messages.
 * Protocol: web/protocol.ts; model sources: web/model-source.ts. */
import { CACHE_NAME, forget, type ModelCache, openSource, remember } from "./model-source";
import type { FromWorker, LoadPhase, ToWorker } from "./protocol";
import { EngineError, type PoolWorker, WasmEngine } from "./wasm-host";

let engine: WasmEngine | undefined;

function post(message: FromWorker): void {
  postMessage(message);
}

/* Copies a stream of `size` bytes into the module's memory, reporting progress. */
async function load(stream: ReadableStream<Uint8Array>, size: number, phase: LoadPhase): Promise<number> {
  const ptr = engine!.allocate(size);
  const reader = stream.getReader();
  let offset = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.length > size) {
      await reader.cancel();
      throw new EngineError("o arquivo é maior que o esperado");
    }
    engine!.fill(ptr, offset, value);
    offset += value.length;
    if (offset - last >= 8 << 20 || offset === size) {
      last = offset;
      post({ type: "progress", phase, loaded: offset, total: size });
    }
  }
  if (offset !== size) throw new EngineError("o arquivo veio incompleto");
  return ptr;
}

async function open(message: Extract<ToWorker, { type: "open" }>): Promise<void> {
  const module = await WebAssembly.compileStreaming(fetch(new URL("./smolmoe.wasm", import.meta.url)));
  engine = await WasmEngine.create(module, message.threads,
    () => new Worker(new URL("./pool-worker.js", import.meta.url), { type: "module" }) as unknown as PoolWorker);
  const cache: ModelCache | undefined = message.model && "caches" in self ? await caches.open(CACHE_NAME) : undefined;
  let ptr: number;
  let size: number;
  let cached = false;
  if (message.file) {
    size = message.file.size;
    ptr = await load(message.file.stream(), size, "file");
  } else {
    const source = await openSource(message.model!, cache, (url) => fetch(url));
    size = source.size;
    cached = source.cached;
    ptr = await load(source.stream, size, cached ? "cache" : "download");
  }
  post({ type: "progress", phase: "prepare", loaded: size, total: size });
  try {
    engine.open(ptr, size, message.context);
  } catch (error) {
    if (cached) await forget(message.model!, cache); /* a cached copy the engine refuses is not kept */
    throw error;
  }
  const stats = engine.stats();
  post({ type: "ready", backend: engine.backend(), threads: stats.threads, context: stats.context, stats });
  if (cache && !cached) {
    /* Out of quota or storage off: the model just downloads again next time. */
    await remember(message.model!, cache, (offset, length) => engine!.read(ptr + offset, length)).catch(() => undefined);
  }
}

function send(text: string): void {
  const ok = engine!.send(text, {
    text: (chunk) => post({ type: "text", chunk }),
    notice: (line) => post({ type: "notice", line }),
  });
  post({ type: "done", ok, stats: engine!.stats() });
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  try {
    if (message.type === "open") await open(message);
    else if (message.type === "send") send(message.text);
    else if (message.type === "reset") engine?.reset();
  } catch (error) {
    engine?.dispose();
    post({ type: "error", message: error instanceof Error ? error.message : String(error), fatal: true });
  }
};
