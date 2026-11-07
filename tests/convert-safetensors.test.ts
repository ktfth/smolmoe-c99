/**
 * Safetensors reading for the converter: header validation (single and
 * sharded), BF16/F16/F32 row decoding with non-finite rejection, and the
 * row converter used by the worker pool (gathered rows -> Q8).
 */
import { afterAll, describe, expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { decodeRow } from "../scripts/hf/decode";
import { createPool } from "../scripts/hf/pool";
import { RowConverter } from "../scripts/hf/rows";
import { type SourceDtype, openCheckpoint, readSafetensorsHeader } from "../scripts/hf/safetensors";
import { quantizeQ8 } from "../scripts/smoe/quant";
import { tempDir } from "./support/run";

const work = tempDir("convert-st");
afterAll(() => work.dispose());

interface Spec {
  readonly name: string;
  readonly dtype: SourceDtype;
  readonly shape: readonly number[];
  readonly data: Uint8Array;
}

const bf16 = (values: readonly number[]) => {
  const f = new Float32Array(values);
  const u = new Uint32Array(f.buffer);
  return new Uint8Array(Uint16Array.from(u, (x) => x >>> 16).buffer);
};
const f32 = (values: readonly number[]) => new Uint8Array(new Float32Array(values).buffer);

/** Writes a safetensors file; `patch` may edit the header object before it is encoded. */
function writeSt(path: string, specs: readonly Spec[], patch?: (h: Record<string, unknown>) => void, pad = 0): void {
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  let offset = 0;
  for (const s of specs) {
    header[s.name] = { dtype: s.dtype, shape: s.shape, data_offsets: [offset, offset + s.data.length] };
    offset += s.data.length;
  }
  patch?.(header);
  const json = new TextEncoder().encode(JSON.stringify(header) + " ".repeat(pad));
  const head = new Uint8Array(8);
  new DataView(head.buffer).setBigUint64(0, BigInt(json.length), true);
  writeFileSync(path, Buffer.concat([head, json, ...specs.map((s) => s.data)]));
}

const values = Array.from({ length: 6 * 40 }, (_, i) => Math.sin(i) * (i % 7));
const SPECS: readonly Spec[] = [
  { name: "w.bf16", dtype: "BF16", shape: [6, 40], data: bf16(values) },
  { name: "w.f32", dtype: "F32", shape: [6, 40], data: f32(values) },
  { name: "v.f16", dtype: "F16", shape: [3], data: new Uint8Array(new Uint16Array([0x3c00, 0xc000, 0x3800]).buffer) },
];

describe("safetensors header", () => {
  test("parses names, dtypes, shapes and absolute offsets", () => {
    const p = join(work.path, "ok.safetensors");
    writeSt(p, SPECS, undefined, 3);
    const h = readSafetensorsHeader(p);
    expect(h.metadata).toEqual({ format: "pt" });
    const t = h.tensors.find((x) => x.name === "w.f32")!;
    expect(t).toMatchObject({ dtype: "F32", shape: [6, 40], bytes: 960 });
    expect(t.offset).toBe(h.dataStart + 480);
  });

  const cases: [string, (h: Record<string, unknown>) => void, RegExp][] = [
    ["an unsupported dtype", (h) => { (h["w.bf16"] as { dtype: string }).dtype = "I8"; }, /unsupported dtype I8/],
    ["a shape/size mismatch", (h) => { (h["w.bf16"] as { shape: number[] }).shape = [6, 41]; }, /size/],
    ["overlapping data", (h) => { (h["w.f32"] as { data_offsets: number[] }).data_offsets = [0, 960]; }, /overlap/],
    ["data past the end", (h) => { (h["v.f16"] as { data_offsets: number[] }).data_offsets = [1440, 1446 + 2]; }, /size|range/],
    ["a negative offset", (h) => { (h["v.f16"] as { data_offsets: number[] }).data_offsets = [-6, 0]; }, /data_offsets/],
    ["a non-integer dimension", (h) => { (h["v.f16"] as { shape: number[] }).shape = [1.5]; }, /shape/],
    ["non-string metadata", (h) => { h.__metadata__ = { a: 1 }; }, /metadata/],
    ["metadata that is not an object", (h) => { h.__metadata__ = "abc"; }, /metadata/],
    ["an inherited property name as dtype", (h) => { (h["w.bf16"] as { dtype: string }).dtype = "constructor"; }, /unsupported dtype/],
  ];
  test.each(cases)("rejects %s", (_what, patch, error) => {
    const p = join(work.path, "bad.safetensors");
    writeSt(p, SPECS, patch);
    expect(() => readSafetensorsHeader(p)).toThrow(error);
  });

  test("accepts a zero-size tensor that shares an offset", () => {
    const p = join(work.path, "empty-tensor.safetensors");
    writeSt(p, SPECS, (h) => { h["z.empty"] = { dtype: "F32", shape: [0, 4], data_offsets: [480, 480] }; });
    expect(readSafetensorsHeader(p).tensors.find((t) => t.name === "z.empty")!.bytes).toBe(0);
  });

  test("rejects an oversized or truncated header and a non-object header", () => {
    const p = join(work.path, "huge.safetensors");
    const head = new Uint8Array(16);
    new DataView(head.buffer).setBigUint64(0, BigInt(17 << 20), true);
    writeFileSync(p, head);
    expect(() => readSafetensorsHeader(p)).toThrow(/header/);
    new DataView(head.buffer).setBigUint64(0, 100n, true);
    writeFileSync(p, head);
    expect(() => readSafetensorsHeader(p)).toThrow(/header/);
    writeFileSync(p, Buffer.concat([Buffer.from([2, 0, 0, 0, 0, 0, 0, 0]), Buffer.from("[]")]));
    expect(() => readSafetensorsHeader(p)).toThrow(/object/);
  });
});

describe("checkpoint directories", () => {
  test("single file and sharded index give the same tensor map", () => {
    const single = join(work.path, "single");
    mkdirSync(single);
    writeSt(join(single, "model.safetensors"), SPECS);
    const sharded = join(work.path, "sharded");
    mkdirSync(sharded);
    writeSt(join(sharded, "a-00001.safetensors"), SPECS.slice(0, 2));
    writeSt(join(sharded, "a-00002.safetensors"), SPECS.slice(2));
    const index = { metadata: { total_size: 1446 }, weight_map: { "w.bf16": "a-00001.safetensors",
      "w.f32": "a-00001.safetensors", "v.f16": "a-00002.safetensors" } };
    writeFileSync(join(sharded, "model.safetensors.index.json"), JSON.stringify(index));
    const a = openCheckpoint(single), b = openCheckpoint(sharded);
    expect([...a.tensors.keys()].sort()).toEqual([...b.tensors.keys()].sort());
    expect(b.files.length).toBe(2);
    expect(b.tensors.get("v.f16")!.file.endsWith("a-00002.safetensors")).toBe(true);
  });

  const shardCase = (name: string, weightMap: Record<string, string>, totalSize: number, error: RegExp) =>
    test(`rejects ${name}`, () => {
      const dir = join(work.path, `shard-${name.replace(/\W+/g, "-")}`);
      mkdirSync(dir, { recursive: true });
      writeSt(join(dir, "a-00001.safetensors"), SPECS.slice(0, 2));
      writeSt(join(dir, "a-00002.safetensors"), SPECS.slice(2));
      writeFileSync(join(dir, "model.safetensors.index.json"),
        JSON.stringify({ metadata: { total_size: totalSize }, weight_map: weightMap }));
      expect(() => openCheckpoint(dir)).toThrow(error);
    });
  shardCase("a path traversal", { "w.bf16": "../a-00001.safetensors", "w.f32": "a-00001.safetensors",
    "v.f16": "a-00002.safetensors" }, 1446, /shard file name/);
  shardCase("a tensor missing from its shard", { "w.bf16": "a-00002.safetensors", "w.f32": "a-00001.safetensors",
    "v.f16": "a-00002.safetensors" }, 1446, /do not match/);
  shardCase("an unlisted tensor", { "w.f32": "a-00001.safetensors", "v.f16": "a-00002.safetensors" }, 1446, /do not match/);
  shardCase("a wrong total_size", { "w.bf16": "a-00001.safetensors", "w.f32": "a-00001.safetensors",
    "v.f16": "a-00002.safetensors" }, 1447, /total_size/);

  test("rejects a directory without weights", () => {
    const dir = join(work.path, "empty");
    mkdirSync(dir);
    expect(() => openCheckpoint(dir)).toThrow(/model\.safetensors/);
  });
});

describe("row decoding and conversion", () => {
  test("decodeRow widens BF16, F16 and F32 exactly", () => {
    const out = new Float32Array(3);
    decodeRow("BF16", bf16([1, -2.5, 0.15625]), 0, 3, out);
    expect([...out]).toEqual([1, -2.5, 0.15625]);
    decodeRow("F16", new Uint8Array(new Uint16Array([0x3c00, 0xc000, 0x0001]).buffer), 0, 3, out);
    expect([...out]).toEqual([1, -2, 2 ** -24]);
    decodeRow("F32", f32([0.1, 3, -4]), 0, 3, out);
    expect([...out]).toEqual([Math.fround(0.1), 3, -4]);
  });

  test.each([
    ["BF16", new Uint8Array(new Uint16Array([0x3f80, 0x7fc0]).buffer)],
    ["BF16", new Uint8Array(new Uint16Array([0xff80, 0]).buffer)],
    ["F16", new Uint8Array(new Uint16Array([0x7c00, 0]).buffer)],
    ["F32", f32([1, Number.NaN])],
  ] as const)("decodeRow rejects non-finite %s", (dtype, bytes) => {
    expect(() => decodeRow(dtype, bytes, 0, 2, new Float32Array(2))).toThrow(/non-finite/);
  });

  test("gathered rows (any order, duplicates) quantize like the plain matrix rows", async () => {
    const p = join(work.path, "rows.safetensors");
    writeSt(p, SPECS);
    const t = readSafetensorsHeader(p).tensors.find((x) => x.name === "w.bf16")!;
    const order = Uint32Array.from([5, 0, 3, 3, 1]);
    const decoded = new Float32Array(6 * 40);
    const raw = bf16(values);
    for (let r = 0; r < 6; r++) decodeRow("BF16", raw, r * 80, 40, decoded.subarray(r * 40));
    const expected = quantizeQ8(Float32Array.from([...order].flatMap((r) => [...decoded.subarray(r * 40, r * 40 + 40)])), 5, 40);
    const job = { file: t.file, offset: t.offset, dtype: t.dtype, cols: 40, sourceRows: 6, rows: order };
    const conv = new RowConverter();
    try {
      expect(Buffer.from(conv.run(job)).equals(Buffer.from(expected))).toBe(true);
      expect(() => conv.run({ ...job, rows: Uint32Array.from([6]) })).toThrow(/row 6/);
    } finally {
      conv.close();
    }
    const pool = createPool(2);
    try {
      const results = await Promise.all([pool.run(job), pool.run({ ...job, rows: order.slice() }), pool.run(job)]);
      for (const r of results) expect(Buffer.from(r).equals(Buffer.from(expected))).toBe(true);
      await expect(pool.run({ ...job, rows: Uint32Array.from([9]) })).rejects.toThrow(/row 9/);
    } finally {
      await pool.close();
    }
  });
});
