/**
 * Single-file packaging, the TS twin of `smolmoe-bundle --out` (without --exe):
 * the model's tensors plus __config__ and __tokenizer__ blobs in a v3,
 * 64-byte-aligned container. Payloads are copied in bounded chunks.
 */
import { closeSync, openSync, readSync, renameSync, rmSync } from "node:fs";
import { resolve } from "node:path";

import { checkFiles } from "./check";
import { BLOB_CONFIG, BLOB_TOKENIZER, DTYPE, type Dtype, compareNames } from "./format";
import { type RecordSpec, blobRecord, planContainer } from "./layout";
import { readContainer } from "./reader";
import { type Chunk, writeContainer } from "./writer";

const COPY_CHUNK = 16 << 20;

function readRange(fd: number, position: number, length: number): Uint8Array {
  const buf = new Uint8Array(length);
  for (let done = 0; done < length;) {
    const n = readSync(fd, buf, done, length - done, position + done);
    if (n <= 0) throw new Error("unexpected end of model file");
    done += n;
  }
  return buf;
}

/** Writes `out` (via a temp file); returns its size. The inputs are validated first, like the C tool. */
export async function bundleSingleFile(model: string, config: Uint8Array, tokenizer: Uint8Array, out: string): Promise<number> {
  if (resolve(out) === resolve(model)) throw new Error("the output must differ from the input model");
  checkFiles({ model, config, tokenizer });
  const parsed = readContainer(model);
  const blobs = new Map([[BLOB_CONFIG, config], [BLOB_TOKENIZER, tokenizer]]);
  const records: RecordSpec[] = parsed.records.filter((r) => r.dtype !== DTYPE.BLOB).map((r) => ({
    name: r.name, rows: r.rows, cols: r.cols, dtype: r.dtype as Dtype, alias: r.alias ? 1 : 0, bytes: r.bytes,
  }));
  for (const [name, bytes] of blobs) records.push(blobRecord(name, bytes.length));
  records.sort((a, b) => compareNames(a.name, b.name));
  const plan = planContainer(records, "aligned64");
  const source = new Map(parsed.records.map((r) => [r.name, r]));
  const partial = `${out}.partial`;
  const fd = openSync(model, "r");
  try {
    const size = await writeContainer(partial, 3, plan, function* (r): Generator<Chunk, void> {
      const blob = blobs.get(r.name);
      if (blob) {
        yield blob;
        return;
      }
      const src = source.get(r.name)!;
      for (let done = 0; done < src.bytes; done += COPY_CHUNK) {
        const length = Math.min(COPY_CHUNK, src.bytes - done);
        yield () => Promise.resolve(readRange(fd, src.offset + done, length));
      }
    }, 4);
    checkFiles({ model: partial });
    renameSync(partial, out);
    return size;
  } catch (e) {
    rmSync(partial, { force: true });
    throw e;
  } finally {
    closeSync(fd);
  }
}
