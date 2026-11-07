/**
 * The unit of conversion work: gather source rows (any order, possibly
 * repeated) of one tensor and quantize them to Q8 in that order. Reads are
 * coalesced: rows are visited in source order and nearby rows share one read.
 * Used inline and inside worker threads (worker.ts).
 */
import { closeSync, openSync, readSync } from "node:fs";

import { decodeRow } from "./decode";
import { DTYPE_BYTES, type SourceDtype } from "./safetensors";
import { q8RowBytes, quantizeQ8Row } from "../smoe/quant";

export interface RowJob {
  readonly file: string;
  /** absolute byte offset of the tensor's first element */
  readonly offset: number;
  readonly dtype: SourceDtype;
  readonly cols: number;
  /** number of rows in the source tensor (bounds check) */
  readonly sourceRows: number;
  /** source row of each output row */
  readonly rows: Uint32Array;
}

const GAP_ROWS = 16; /* read through gaps up to this many rows instead of seeking */
const MAX_SEGMENT_BYTES = 32 << 20;

export class RowConverter {
  private readonly fds = new Map<string, number>();
  private buffer = new Uint8Array(0);
  private row = new Float32Array(0);

  private fd(file: string): number {
    let fd = this.fds.get(file);
    if (fd === undefined) {
      fd = openSync(file, "r");
      this.fds.set(file, fd);
    }
    return fd;
  }

  private read(file: string, position: number, length: number): Uint8Array {
    if (this.buffer.length < length) this.buffer = new Uint8Array(Math.max(length, this.buffer.length * 2));
    const fd = this.fd(file);
    for (let done = 0; done < length;) {
      const n = readSync(fd, this.buffer, done, length - done, position + done);
      if (n <= 0) throw new Error(`${file}: unexpected end of file`);
      done += n;
    }
    return this.buffer;
  }

  /** Q8 bytes of the job's rows, in job order. */
  run(job: RowJob): Uint8Array {
    const { cols, rows } = job;
    const rowBytes = cols * DTYPE_BYTES[job.dtype], outRowBytes = q8RowBytes(cols);
    for (const r of rows) if (r >= job.sourceRows) throw new Error(`${job.file}: row ${r} is outside the tensor`);
    if (this.row.length < cols) this.row = new Float32Array(cols);
    const out = new Uint8Array(rows.length * outRowBytes);
    const view = new DataView(out.buffer);
    const order = Uint32Array.from(rows.keys()).sort((a, b) => rows[a]! - rows[b]! || a - b);
    const maxRows = Math.max(1, Math.floor(MAX_SEGMENT_BYTES / rowBytes));
    for (let s = 0; s < order.length;) {
      const first = rows[order[s]!]!;
      let e = s + 1;
      while (e < order.length && rows[order[e]!]! - rows[order[e - 1]!]! <= GAP_ROWS && rows[order[e]!]! - first < maxRows) ++e;
      const last = rows[order[e - 1]!]!;
      const buf = this.read(job.file, job.offset + first * rowBytes, (last - first + 1) * rowBytes);
      for (let i = s; i < e; i++) {
        const pos = order[i]!;
        decodeRow(job.dtype, buf, (rows[pos]! - first) * rowBytes, cols, this.row);
        quantizeQ8Row(this.row, 0, cols, out, view, pos * outRowBytes);
      }
      s = e;
    }
    return out;
  }

  close(): void {
    for (const fd of this.fds.values()) closeSync(fd);
    this.fds.clear();
  }
}
