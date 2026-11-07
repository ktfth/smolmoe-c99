/** Worker: hashes the file whose path it receives, replies once. */
import { sha256File } from "./hash";

interface Scope {
  onmessage: ((event: MessageEvent<string>) => void) | null;
  postMessage(message: { readonly ok: boolean; readonly value: string }): void;
}

const scope = globalThis as unknown as Scope;
scope.onmessage = (event) => {
  try {
    scope.postMessage({ ok: true, value: sha256File(event.data) });
  } catch (e) {
    scope.postMessage({ ok: false, value: e instanceof Error ? e.message : String(e) });
  }
};
