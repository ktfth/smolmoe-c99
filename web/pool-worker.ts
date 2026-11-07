/* One thread of the browser build's pool (src/threadpool.c): an instance of
 * the module on the engine's shared memory, with its own stack. It answers
 * the first message with "ready", then waits inside the module for work and
 * never returns. Pool workers make no system calls. */
interface WorkerExports {
  readonly __stack_pointer: WebAssembly.Global;
  sm_worker_stack(slot: number): number;
  sm_worker_main(slot: number): void;
}

const ENOSYS = 52;

self.onmessage = (event: MessageEvent<{ module: WebAssembly.Module; memory: WebAssembly.Memory; slot: number }>) => {
  const { module, memory, slot } = event.data;
  const instance = new WebAssembly.Instance(module, {
    env: { memory },
    wasi_snapshot_preview1: new Proxy({}, { get: () => () => ENOSYS }),
    smolmoe: new Proxy({}, { get: () => () => 0 }),
  });
  const exports = instance.exports as unknown as WorkerExports;
  exports.__stack_pointer.value = exports.sm_worker_stack(slot);
  postMessage("ready");
  exports.sm_worker_main(slot);
};
