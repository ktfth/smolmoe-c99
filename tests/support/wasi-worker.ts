/* One worker instance of a threaded WASI module (src/threadpool.c): the same
 * shared memory as the main instance, its own stack. It reports ready, then
 * waits inside the module until a thread pool hands it work. Workers do no
 * I/O, so every WASI call answers ENOSYS. */
import { parentPort, workerData } from "node:worker_threads";

interface WorkerExports {
  readonly __stack_pointer: WebAssembly.Global;
  sm_worker_stack(slot: number): number;
  sm_worker_main(slot: number): void;
}

const { module, memory, slot } = workerData as { module: WebAssembly.Module; memory: WebAssembly.Memory; slot: number };
const ENOSYS = 52;
const instance = new WebAssembly.Instance(module, {
  env: { memory },
  wasi_snapshot_preview1: new Proxy({}, { get: () => () => ENOSYS }),
  smolmoe: { workers: () => 0 },
});
const exports = instance.exports as unknown as WorkerExports;
exports.__stack_pointer.value = exports.sm_worker_stack(slot);
parentPort!.postMessage("ready");
exports.sm_worker_main(slot);
