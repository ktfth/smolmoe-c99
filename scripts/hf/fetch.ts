/**
 * Downloads pinned Hugging Face files (repo + revision, optional SHA-256) into
 * a directory. Each file goes to `NAME.download`, is verified, then renamed;
 * files already present and verified are skipped.
 */
import { closeSync, existsSync, mkdirSync, openSync, renameSync, rmSync, writeSync } from "node:fs";
import { join } from "node:path";

import { sha256File } from "./hash";

export interface PinnedFile {
  readonly name: string;
  readonly revision: string;
  readonly sha256?: string;
}

const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const REPO = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;
const IDLE_MS = 60_000;

export type Fetcher = (url: string) => Promise<Response>;

/** Rejects when `step` does not settle within IDLE_MS (a stalled connection). */
async function idle<T>(step: Promise<T>, url: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const stalled = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${url}: no data for ${IDLE_MS / 1000} s`)), IDLE_MS);
  });
  try {
    return await Promise.race([step, stalled]);
  } finally {
    clearTimeout(timer);
  }
}

/* Synchronous writes: the stream is pulled no faster than the disk accepts it. */
async function download(url: string, dest: string, fetcher: Fetcher): Promise<void> {
  const res = await idle(fetcher(url), url);
  if (!res.ok || !res.body) {
    await res.body?.cancel().catch(() => undefined);
    throw new Error(`${url}: HTTP ${res.status}`);
  }
  const fd = openSync(dest, "w");
  const reader = res.body.getReader();
  try {
    for (let r = await idle(reader.read(), url); !r.done; r = await idle(reader.read(), url))
      for (let off = 0; off < r.value.length;) off += writeSync(fd, r.value, off);
  } catch (e) {
    await reader.cancel().catch(() => undefined);
    throw e;
  } finally {
    closeSync(fd);
  }
}

const defaultFetcher: Fetcher = (url) => {
  const token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN;
  return token ? fetch(url, { headers: { Authorization: `Bearer ${token}` } }) : fetch(url);
};

export async function fetchPinned(repo: string, files: readonly PinnedFile[], dir: string,
  log: (line: string) => void, fetcher: Fetcher = defaultFetcher): Promise<{ readonly fetched: readonly string[] }> {
  if (!REPO.test(repo)) throw new Error(`invalid repo: ${repo}`);
  mkdirSync(dir, { recursive: true });
  const fetched: string[] = [];
  for (const f of files) {
    if (!SAFE_NAME.test(f.name) || !SAFE_NAME.test(f.revision)) throw new Error(`invalid pinned file: ${f.name}@${f.revision}`);
    const dest = join(dir, f.name);
    if (existsSync(dest) && (!f.sha256 || sha256File(dest) === f.sha256)) continue;
    const partial = `${dest}.download`;
    const url = `https://huggingface.co/${repo}/resolve/${f.revision}/${f.name}`;
    log(`downloading ${url}`);
    try {
      await download(url, partial, fetcher);
      const digest = f.sha256 ? sha256File(partial) : undefined;
      if (f.sha256 && digest !== f.sha256) throw new Error(`${f.name}: SHA-256 ${digest}, expected ${f.sha256}`);
      renameSync(partial, dest);
      fetched.push(f.name);
    } finally {
      rmSync(partial, { force: true });
    }
  }
  return { fetched };
}
