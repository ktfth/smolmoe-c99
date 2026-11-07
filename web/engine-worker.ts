/* The page's engine worker: owns the WebAssembly engine, so the page never
 * blocks. Loads a model (a download, kept in the browser's cache, or a file
 * the user picked) straight into the module's memory, then answers messages.
 * Protocol: web/protocol.ts. */
import type { FromWorker, ToWorker } from "./protocol";
import { EngineError, type PoolWorker, WasmEngine } from "./wasm-host";

const CACHE = "smolmoe-modelos-v1";

let engine: WasmEngine | undefined;

function post(message: FromWorker): void {
  postMessage(message);
}

/* A model's bytes as a stream and their count, from the cache when the URL
 * was downloaded before; a download is cached while it streams. */
async function source(url: string): Promise<{ stream: ReadableStream<Uint8Array>; size: number; cached: boolean }> {
  const cache = "caches" in self ? await caches.open(CACHE) : undefined;
  const hit = await cache?.match(url);
  if (hit?.body) return { stream: hit.body, size: Number(hit.headers.get("content-length") ?? 0), cached: true };
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new EngineError(`download falhou: HTTP ${response.status}`);
  const size = Number(response.headers.get("content-length") ?? 0);
  if (!cache) return { stream: response.body, size, cached: false };
  const [mine, kept] = response.body.tee();
  const headers = new Headers({ "content-length": String(size), "content-type": "application/octet-stream" });
  cache.put(url, new Response(kept, { headers })).catch(() => {
    /* Out of quota: the model just downloads again next time. */
  });
  return { stream: mine, size, cached: false };
}

/* Copies a stream of known size into the module's memory, reporting progress. */
async function load(stream: ReadableStream<Uint8Array>, size: number, phase: "download" | "cache" | "file"): Promise<number> {
  if (!size) throw new EngineError("tamanho do modelo desconhecido (o servidor não enviou Content-Length)");
  const ptr = engine!.allocate(size);
  const reader = stream.getReader();
  let offset = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (offset + value.length > size) throw new EngineError("o arquivo é maior que o anunciado");
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
  let ptr: number;
  let size: number;
  if (message.file) {
    size = message.file.size;
    ptr = await load(message.file.stream(), size, "file");
  } else {
    const s = await source(message.url!);
    size = s.size;
    ptr = await load(s.stream, size, s.cached ? "cache" : "download");
  }
  post({ type: "progress", phase: "prepare", loaded: size, total: size });
  engine.open(ptr, size, message.context);
  const stats = engine.stats();
  post({ type: "ready", backend: engine.backend(), threads: stats.threads, context: stats.context, stats });
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
    post({ type: "error", message: error instanceof Error ? error.message : String(error), fatal: true });
  }
};
