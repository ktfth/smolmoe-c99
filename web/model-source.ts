/* Where a model's bytes come from: the browser's cache, or a download. A model
 * enters the cache only after it opened, copied back out of the module's
 * memory, so a bad download or a file the engine refuses is never kept. The
 * cache key carries the model's SHA-256, so a new version is a new entry.
 * Cache and fetch are passed in, so the tests can fake them. */
import type { ModelEntry } from "./protocol";

export const CACHE_NAME = "smolmoe-modelos-v1";
const CHUNK = 8 << 20;

/* The part of the Cache API used here. */
export interface ModelCache {
  match(key: string): Promise<Response | undefined>;
  put(key: string, response: Response): Promise<void>;
  delete(key: string): Promise<boolean>;
}

export type Fetch = (url: string) => Promise<Response>;

export interface Source {
  readonly stream: ReadableStream<Uint8Array>;
  readonly size: number;
  readonly cached: boolean;
}

export class SourceError extends Error {}

export function cacheKey(model: ModelEntry): string {
  return `${model.url}${model.url.includes("?") ? "&" : "?"}sha256=${model.sha256}`;
}

/* The model's bytes and their count, from the cache when a complete copy is
 * there; a cached copy of the wrong size is dropped. */
export async function openSource(model: ModelEntry, cache: ModelCache | undefined, fetchUrl: Fetch): Promise<Source> {
  const key = cacheKey(model);
  const hit = await cache?.match(key);
  if (hit?.body) {
    const size = Number(hit.headers.get("content-length") ?? 0);
    if (size === model.bytes) return { stream: hit.body, size, cached: true };
    await hit.body.cancel();
    await cache!.delete(key);
  }
  const response = await fetchUrl(model.url);
  if (!response.ok || !response.body) throw new SourceError(`download falhou: HTTP ${response.status}`);
  const announced = Number(response.headers.get("content-length") ?? 0);
  if (announced && announced !== model.bytes) {
    await response.body.cancel();
    throw new SourceError(`o servidor anunciou ${announced} bytes, e o modelo tem ${model.bytes}`);
  }
  return { stream: response.body, size: model.bytes, cached: false };
}

/* Keeps an opened model in the cache, read back in pieces from the module's
 * memory (read(offset, length)), so no second full copy sits in JS memory. */
export async function remember(model: ModelEntry, cache: ModelCache, read: (offset: number, length: number) => Uint8Array): Promise<void> {
  let offset = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (offset >= model.bytes) {
        controller.close();
        return;
      }
      const length = Math.min(CHUNK, model.bytes - offset);
      controller.enqueue(read(offset, length));
      offset += length;
    },
  });
  const headers = new Headers({ "content-length": String(model.bytes), "content-type": "application/octet-stream" });
  await cache.put(cacheKey(model), new Response(stream, { headers }));
}

/* Drops a cached copy the engine refused. */
export async function forget(model: ModelEntry, cache: ModelCache | undefined): Promise<void> {
  await cache?.delete(cacheKey(model));
}
