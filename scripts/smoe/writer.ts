/**
 * Sequential SMOEQ8 writer. Payload chunks are produced per record, possibly
 * asynchronously (worker pool); up to `lookahead` chunks are started ahead
 * and written strictly in plan order, so the output is deterministic and the
 * file grows sequentially (no seeks, no NTFS zero-fill of gaps).
 */
import { closeSync, openSync, writeSync } from "node:fs";

import { type PlannedRecord, encodeDirectory } from "./layout";
import type { Version } from "./format";

/** Bytes, or a thunk started ahead of time that resolves to bytes. */
export type Chunk = Uint8Array | (() => Promise<Uint8Array>);

type Pending = Uint8Array | Promise<Uint8Array>;

interface Item {
  readonly record: PlannedRecord;
  readonly chunk: Pending;
}

function writeAll(fd: number, bytes: Uint8Array): void {
  for (let off = 0; off < bytes.length;) off += writeSync(fd, bytes, off, bytes.length - off);
}

/** Every non-alias record in plan order: zero padding to its offset, then its chunks. */
function* items(plan: readonly PlannedRecord[], start: number,
  chunksOf: (r: PlannedRecord) => Iterable<Chunk>): Generator<Item | { readonly end: PlannedRecord }> {
  let position = start;
  for (const record of plan) {
    if (record.alias) continue;
    if (record.offset < position) throw new Error(`${record.name}: payload offset precedes the write position`);
    if (record.offset > position) yield { record, chunk: new Uint8Array(record.offset - position) };
    for (const c of chunksOf(record)) {
      const chunk = typeof c === "function" ? c() : c;
      /* A job may fail while an earlier one is awaited: mark it handled now, it is still awaited in order. */
      if (chunk instanceof Promise) chunk.catch(() => undefined);
      yield { record, chunk };
    }
    yield { end: record };
    position = record.offset + record.bytes;
  }
}

/** Writes header, directory and payloads to `path`; returns the file size. */
export async function writeContainer(path: string, version: Version, plan: readonly PlannedRecord[],
  chunksOf: (r: PlannedRecord) => Iterable<Chunk>, lookahead = 16): Promise<number> {
  const directory = encodeDirectory(version, plan);
  const fd = openSync(path, "w");
  const window: (Item | { readonly end: PlannedRecord })[] = [];
  try {
    writeAll(fd, directory);
    let position = directory.length;
    const source = items(plan, position, chunksOf);
    let started = 0;
    const fill = () => {
      while (started < lookahead) {
        const next = source.next();
        if (next.done) return;
        window.push(next.value);
        if ("chunk" in next.value) ++started;
      }
    };
    fill();
    while (window.length) {
      const head = window.shift()!;
      if ("end" in head) {
        const r = head.end;
        if (position !== r.offset + r.bytes)
          throw new Error(`${r.name}: wrote ${position - r.offset} bytes, expected ${r.bytes}`);
        continue;
      }
      const bytes = await head.chunk;
      --started;
      writeAll(fd, bytes);
      position += bytes.length;
      fill();
    }
    return position;
  } catch (e) {
    /* Let started jobs settle so none rejects unobserved. */
    await Promise.allSettled(window.flatMap((w) => ("chunk" in w ? [w.chunk] : [])));
    throw e;
  } finally {
    closeSync(fd);
  }
}
