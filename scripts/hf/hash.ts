/**
 * SHA-256 of source files, in parallel worker threads so hashing multi-GB
 * shards overlaps the conversion. Fixed 4 MiB reads keep memory bounded.
 */
import { closeSync, openSync, readSync } from "node:fs";

const BLOCK = 4 << 20;

export function sha256File(path: string): string {
  const hasher = new Bun.CryptoHasher("sha256");
  const buf = new Uint8Array(BLOCK);
  const fd = openSync(path, "r");
  try {
    for (let n; (n = readSync(fd, buf, 0, buf.length, null)) > 0;) hasher.update(buf.subarray(0, n));
  } finally {
    closeSync(fd);
  }
  return hasher.digest("hex");
}

interface HashReply {
  readonly ok: boolean;
  readonly value: string;
}

/** Hashes each file in its own worker; resolves to path -> hex digest. */
export async function sha256Files(paths: readonly string[]): Promise<ReadonlyMap<string, string>> {
  const workers: Worker[] = [];
  try {
    const digests = await Promise.all(paths.map((path) => new Promise<string>((resolve, reject) => {
      const worker = new Worker(new URL("./hash-worker.ts", import.meta.url).href);
      workers.push(worker);
      worker.onmessage = (event: MessageEvent<HashReply>) => {
        if (event.data.ok) resolve(event.data.value);
        else reject(new Error(`${path}: ${event.data.value}`));
      };
      worker.onerror = (event: ErrorEvent) => reject(new Error(`${path}: hashing failed: ${event.message}`));
      worker.onmessageerror = () => reject(new Error(`${path}: hashing failed: unreadable reply`));
      worker.postMessage(path);
    })));
    return new Map(paths.map((p, i) => [p, digests[i]!]));
  } finally {
    for (const w of workers) w.terminate(); /* also stops the other hashes when one fails */
  }
}
