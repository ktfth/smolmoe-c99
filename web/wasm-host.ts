/* Host of the browser build (build/web/smolmoe.wasm, from src/web.c): shared
 * memory, the pool's worker instances, the few WASI calls the module makes,
 * and the engine's text and notices. Runs in the page's engine worker, and
 * under bun in the tests, unchanged. */

/* Memory pages: at least the module's declared minimum, and exactly its
 * declared maximum (Makefile: --max-memory=7FFF0000, 2 GiB - 64 KiB). */
export const INITIAL_PAGES = 1024;
export const MAXIMUM_PAGES = 32767;
/* src/threadpool.c: WASM_WORKERS. */
export const MAX_WORKERS = 15;
/* The largest model the 2 GiB memory holds with room for the KV cache and
 * the activations: Qwen 0.5B (560 MB) and TinyStories fit, Phi-4-mini not. */
export const MAX_MODEL_BYTES = 1536 * 1048576;

const ENOSYS = 52;
const EBADF = 8;
const FILETYPE_CHARACTER_DEVICE = 2;

export interface EngineEvents {
  text(chunk: string): void;
  notice(line: string): void;
}

export interface EngineStats {
  readonly generated: number;
  readonly decodeSeconds: number;
  readonly promptTokens: number;
  readonly prefillSeconds: number;
  readonly threads: number;
  readonly context: number;
}

/* A worker able to run one pool slot (web/pool-worker.ts). */
export interface PoolWorker {
  postMessage(message: unknown): void;
  terminate(): void;
  addEventListener(type: "message" | "error", listener: (event: { data?: unknown; message?: string }) => void): void;
}

interface Exports {
  readonly _initialize: () => void;
  readonly sm_alloc: (size: number) => number;
  readonly sm_open: (data: number, size: number, context: number, threads: number) => number;
  readonly sm_message: () => number;
  readonly sm_message_cap: () => number;
  readonly sm_send: (len: number, year: number, month: number, day: number, hour: number, minute: number) => number;
  readonly sm_reset: () => void;
  readonly sm_stats: () => number;
  readonly sm_backend: () => number;
}

/* The module stopped (die() in C): its message, from stderr. */
export class EngineError extends Error {}

class Exit {
  constructor(readonly code: number) {}
}

/* Starts one pool worker on the shared memory and waits until it is in. */
function startWorker(spawn: () => PoolWorker, module: WebAssembly.Module, memory: WebAssembly.Memory,
  slot: number): Promise<PoolWorker> {
  return new Promise((ready, fail) => {
    const worker = spawn();
    worker.addEventListener("message", () => ready(worker));
    worker.addEventListener("error", (event) => fail(new EngineError(`worker ${slot}: ${event.message ?? "erro"}`)));
    worker.postMessage({ module, memory, slot });
  });
}

export class WasmEngine {
  private readonly decoder = new TextDecoder();
  private readonly streamDecoder = new TextDecoder(); /* reply bytes can split a character */
  private stderr = "";
  private events: EngineEvents | undefined;

  private constructor(
    private readonly memory: WebAssembly.Memory,
    private readonly exports: Exports,
    private readonly workers: readonly PoolWorker[],
  ) {}

  /* Instantiates the module with `threads - 1` pool workers (capped). */
  static async create(module: WebAssembly.Module, threads: number, spawn: () => PoolWorker): Promise<WasmEngine> {
    const memory = new WebAssembly.Memory({ initial: INITIAL_PAGES, maximum: MAXIMUM_PAGES, shared: true });
    const count = Math.max(0, Math.min(threads - 1, MAX_WORKERS));
    const workers = await Promise.all(Array.from({ length: count }, (_, slot) => startWorker(spawn, module, memory, slot)));
    let engine: WasmEngine | undefined;
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
      wasi_snapshot_preview1: new Proxy(
        {
          fd_write: (fd: number, iovs: number, count: number, written: number) => engine!.fdWrite(fd, iovs, count, written),
          fd_fdstat_get: (fd: number, out: number) => engine!.fdstat(fd, out),
          clock_time_get: (id: number, _precision: bigint, out: number) => engine!.clock(id, out),
          proc_exit: (code: number) => {
            throw new Exit(code);
          },
          fd_read: () => EBADF,
          fd_seek: () => EBADF,
          fd_close: () => EBADF,
        } as Record<string, (...args: never[]) => number>,
        { get: (target, name: string) => target[name] ?? (() => ENOSYS) },
      ),
      smolmoe: {
        workers: () => workers.length,
        text: (ptr: number, len: number) => engine!.text(ptr, len),
        reply_end: () => engine!.replyEnd(),
        notice: (ptr: number) => engine!.notice(ptr),
      },
    });
    engine = new WasmEngine(memory, instance.exports as unknown as Exports, workers);
    engine.call(() => engine!.exports._initialize());
    return engine;
  }

  /* Room for the model; fill it before open(). */
  allocate(size: number): number {
    if (!Number.isInteger(size) || size <= 0 || size > MAX_MODEL_BYTES)
      throw new EngineError(`um modelo de ${Math.round(size / 1e6)} MB não cabe no navegador (máximo: ${Math.round(MAX_MODEL_BYTES / 1e6)} MB)`);
    const ptr = this.call(() => this.exports.sm_alloc(size));
    if (!ptr) throw new EngineError(`sem memória para ${Math.round(size / 1048576)} MB`);
    return ptr;
  }

  /* Copies a piece of the model to ptr + offset (memory.buffer changes as memory grows). */
  fill(ptr: number, offset: number, chunk: Uint8Array): void {
    new Uint8Array(this.memory.buffer).set(chunk, ptr + offset);
  }

  /* Opens the model written at data[0..size). */
  open(data: number, size: number, context: number): void {
    if (!this.call(() => this.exports.sm_open(data, size, context, this.workers.length + 1)))
      throw new EngineError("um modelo já está aberto");
  }

  /* A copy of bytes the module holds (the model, to keep it in the cache). */
  read(ptr: number, len: number): Uint8Array {
    return this.copy(ptr, len);
  }

  /* Answers one message; returns false when it did not fit in the context. */
  send(text: string, events: EngineEvents, now: Date = new Date()): boolean {
    const bytes = new TextEncoder().encode(text);
    const cap = this.exports.sm_message_cap();
    const len = utf8Prefix(bytes, cap);
    new Uint8Array(this.memory.buffer).set(bytes.subarray(0, len), this.exports.sm_message());
    this.events = events;
    try {
      return this.call(() => this.exports.sm_send(len, now.getFullYear(), now.getMonth() + 1, now.getDate(),
        now.getHours(), now.getMinutes())) !== 0;
    } finally {
      this.events = undefined;
    }
  }

  reset(): void {
    this.call(() => this.exports.sm_reset());
  }

  stats(): EngineStats {
    const at = this.exports.sm_stats();
    const v = new Float64Array(this.memory.buffer.slice(at, at + 48));
    return { generated: v[0]!, decodeSeconds: v[1]!, promptTokens: v[2]!, prefillSeconds: v[3]!, threads: v[4]!, context: v[5]! };
  }

  backend(): string {
    return this.cString(this.exports.sm_backend());
  }

  /* Ends the pool workers (they wait inside the module). A pool worker that
   * traps after it started leaves the engine waiting for it: reload. */
  dispose(): void {
    for (const worker of this.workers) worker.terminate();
  }

  /* Runs a call into the module; die() there becomes an EngineError with
   * what it wrote to stderr during this call. */
  private call<T>(fn: () => T): T {
    this.stderr = "";
    try {
      return fn();
    } catch (error) {
      if (error instanceof Exit) throw new EngineError(this.stderr.trim().replace(/^smolmoe: /, "") || `saída ${error.code}`);
      throw error;
    }
  }

  private view(): DataView {
    return new DataView(this.memory.buffer);
  }

  private copy(ptr: number, len: number): Uint8Array {
    return new Uint8Array(this.memory.buffer).slice(ptr, ptr + len);
  }

  private cString(ptr: number): string {
    const bytes = new Uint8Array(this.memory.buffer);
    let end = ptr;
    while (end < bytes.length && bytes[end] !== 0) ++end;
    return this.decoder.decode(bytes.slice(ptr, end));
  }

  fdWrite(fd: number, iovs: number, count: number, written: number): number {
    const v = this.view();
    let total = 0;
    for (let i = 0; i < count; ++i) {
      const ptr = v.getUint32(iovs + 8 * i, true);
      const len = v.getUint32(iovs + 8 * i + 4, true);
      if (fd === 2) this.stderr += this.decoder.decode(this.copy(ptr, len));
      total += len;
    }
    v.setUint32(written, total, true);
    return 0;
  }

  fdstat(fd: number, out: number): number {
    if (fd > 2) return EBADF;
    const v = this.view();
    for (let i = 0; i < 24; i += 8) v.setBigUint64(out + i, 0n, true);
    v.setUint8(out, FILETYPE_CHARACTER_DEVICE);
    return 0;
  }

  clock(id: number, out: number): number {
    const ns = id === 0 ? BigInt(Date.now()) * 1_000_000n : BigInt(Math.round(performance.now() * 1e6));
    this.view().setBigUint64(out, ns, true);
    return 0;
  }

  text(ptr: number, len: number): void {
    const chunk = this.streamDecoder.decode(this.copy(ptr, len), { stream: true });
    if (chunk) this.events?.text(chunk);
  }

  replyEnd(): void {
    const rest = this.streamDecoder.decode();
    if (rest) this.events?.text(rest);
  }

  notice(ptr: number): void {
    this.events?.notice(this.cString(ptr));
  }
}

/* The longest prefix of bytes, at most cap long, that ends on a character. */
export function utf8Prefix(bytes: Uint8Array, cap: number): number {
  if (bytes.length <= cap) return bytes.length;
  let n = cap;
  while (n > 0 && (bytes[n]! & 0xc0) === 0x80) --n;
  return n;
}
