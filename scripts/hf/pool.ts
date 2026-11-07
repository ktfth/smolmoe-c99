/**
 * A fixed pool of worker threads running RowJobs. With threads <= 1 jobs run
 * inline on the main thread (same code, no workers). Results are returned per
 * job; ordering is the caller's business (the container writer keeps order).
 */
import { type RowJob, RowConverter } from "./rows";
import type { WorkerReply, WorkerRequest } from "./worker";

export interface RowPool {
  readonly threads: number;
  run(job: RowJob): Promise<Uint8Array>;
  close(): Promise<void>;
}

const CLOSE_GRACE_MS = 2000;

interface Waiting {
  readonly id: number;
  readonly job: RowJob;
  readonly resolve: (bytes: Uint8Array) => void;
  readonly reject: (error: Error) => void;
}

function inlinePool(): RowPool {
  const converter = new RowConverter();
  return {
    threads: 1,
    run: async (job) => converter.run(job),
    close: async () => converter.close(),
  };
}

export function createPool(threads: number): RowPool {
  if (!Number.isInteger(threads) || threads < 1) throw new Error(`invalid thread count: ${threads}`);
  if (threads === 1) return inlinePool();
  const workers: Worker[] = [];
  try {
    for (let i = 0; i < threads; i++) workers.push(new Worker(new URL("./worker.ts", import.meta.url).href));
  } catch (e) {
    for (const w of workers) w.terminate();
    throw e;
  }
  const busy = new Map<Worker, Waiting>();
  const queue: Waiting[] = [];
  let nextId = 0;
  let broken: Error | undefined;

  const dispatch = () => {
    for (const w of workers) {
      if (!queue.length) return;
      if (busy.has(w)) continue;
      const item = queue.shift()!;
      busy.set(w, item);
      const request: WorkerRequest = { id: item.id, job: item.job };
      w.postMessage(request);
    }
  };
  const failAll = (error: Error) => {
    broken = error;
    for (const item of [...busy.values(), ...queue.splice(0)]) item.reject(error);
    busy.clear();
  };
  for (const w of workers) {
    w.onmessage = (event: MessageEvent<WorkerReply>) => {
      const item = busy.get(w);
      busy.delete(w);
      if (!item || item.id !== event.data.id) return failAll(new Error("worker pool: reply for an unknown job"));
      if (event.data.ok) item.resolve(event.data.bytes);
      else item.reject(new Error(event.data.error));
      dispatch();
    };
    w.onerror = (event: ErrorEvent) => failAll(new Error(`worker crashed: ${event.message}`));
  }
  return {
    threads,
    run: (job) => new Promise<Uint8Array>((resolve, reject) => {
      if (broken) return reject(broken);
      queue.push({ id: nextId++, job: { ...job, rows: job.rows.slice() }, resolve, reject });
      dispatch();
    }),
    /* Ask each worker to close its source files (Windows keeps open files locked), then terminate. */
    close: async () => {
      failAll(new Error("worker pool closed"));
      await Promise.all(workers.map((w) => {
        const closed = new Promise<void>((resolve) => w.addEventListener("close", () => resolve(), { once: true }));
        w.postMessage("close");
        return Promise.race([closed, Bun.sleep(CLOSE_GRACE_MS)]);
      }));
      for (const w of workers) w.terminate();
    },
  };
}
