/**
 * Worker thread: runs RowJobs (read + decode + Q8) off the main thread and
 * transfers the result back. One RowConverter (open files, scratch buffers)
 * per worker.
 */
import { type RowJob, RowConverter } from "./rows";

export interface WorkerRequest {
  readonly id: number;
  readonly job: RowJob;
}

export type WorkerReply =
  | { readonly id: number; readonly ok: true; readonly bytes: Uint8Array }
  | { readonly id: number; readonly ok: false; readonly error: string };

interface WorkerScope {
  onmessage: ((event: MessageEvent<WorkerRequest | "close">) => void) | null;
  postMessage(message: WorkerReply, transfer?: ArrayBuffer[]): void;
  close(): void;
}

const scope = globalThis as unknown as WorkerScope;
const converter = new RowConverter();

scope.onmessage = (event) => {
  if (event.data === "close") {
    converter.close();
    scope.close();
    return;
  }
  const { id, job } = event.data;
  try {
    const bytes = converter.run(job);
    scope.postMessage({ id, ok: true, bytes }, [bytes.buffer as ArrayBuffer]);
  } catch (e) {
    scope.postMessage({ id, ok: false, error: e instanceof Error ? e.message : String(e) });
  }
};
