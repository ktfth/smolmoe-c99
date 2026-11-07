/**
 * Tests for the reference's own safetensors reader: header parsing, the shard
 * index, row reads, and validation. Uses a synthetic two-shard checkpoint in a
 * temporary directory.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { parseSafetensorsHeader, WeightStore } from "../scripts/ref/safetensors";

interface FakeTensor {
  name: string;
  dtype: string;
  shape: number[];
  bits: number[];
}

function safetensorsBytes(tensors: readonly FakeTensor[], pad = 0): Uint8Array {
  const header: Record<string, unknown> = { __metadata__: { format: "pt" } };
  let offset = 0;
  for (const t of tensors) {
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + t.bits.length * 2] };
    offset += t.bits.length * 2;
  }
  const json = new TextEncoder().encode(JSON.stringify(header) + " ".repeat(pad));
  const out = new Uint8Array(8 + json.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  const data = new DataView(out.buffer, 8 + json.length);
  let p = 0;
  for (const t of tensors) for (const b of t.bits) (data.setUint16(p, b, true), (p += 2));
  return out;
}

let dir = "";

beforeAll(() => {
  dir = mkdtempSync(join(tmpdir(), "ref-st-"));
  const a = { name: "a", dtype: "BF16", shape: [2, 3], bits: [0x3f80, 0x4000, 0x4040, 0xbf80, 0xc000, 0xc040] };
  const b = { name: "b", dtype: "BF16", shape: [3], bits: [0x3f00, 0x3e80, 0x0000] };
  const c = { name: "c", dtype: "F32", shape: [1], bits: [0, 0x3f80] };
  writeFileSync(join(dir, "s1.safetensors"), safetensorsBytes([a], 3)); // odd padding: data starts unaligned
  writeFileSync(join(dir, "s2.safetensors"), safetensorsBytes([b, c]));
  const index = { metadata: {}, weight_map: { a: "s1.safetensors", b: "s2.safetensors", c: "s2.safetensors" } };
  writeFileSync(join(dir, "model.safetensors.index.json"), JSON.stringify(index));
});

afterAll(() => rmSync(dir, { recursive: true, force: true }));

describe("header parsing", () => {
  test("reads length, dtype, shape, and absolute data offsets", () => {
    const bytes = safetensorsBytes([{ name: "x", dtype: "BF16", shape: [2], bits: [1, 2] }]);
    const headerLength = Number(new DataView(bytes.buffer).getBigUint64(0, true));
    const parsed = parseSafetensorsHeader(bytes.subarray(0, 8 + headerLength), "mem");
    const x = parsed.get("x")!;
    expect(x.shape).toEqual([2]);
    expect(x.dtype).toBe("BF16");
    expect(x.start).toBe(8 + headerLength);
    expect(x.byteLength).toBe(4);
    expect(parsed.has("__metadata__")).toBe(false);
  });

  test("rejects offsets that disagree with the shape", () => {
    const bytes = safetensorsBytes([{ name: "x", dtype: "BF16", shape: [3], bits: [1, 2] }]);
    const headerLength = Number(new DataView(bytes.buffer).getBigUint64(0, true));
    expect(() => parseSafetensorsHeader(bytes.subarray(0, 8 + headerLength), "mem")).toThrow();
  });
});

describe("WeightStore over a sharded checkpoint", () => {
  test("reads whole tensors and row ranges as raw bf16 bits", () => {
    const store = WeightStore.open(dir);
    try {
      expect(store.shape("a")).toEqual([2, 3]);
      expect(Array.from(store.readRowsBf16("a", 1, 1))).toEqual([0xbf80, 0xc000, 0xc040]);
      expect(Array.from(store.readVector("b"))).toEqual([0.5, 0.25, 0]);
      expect(Array.from(store.readRows("a", 0, 2))).toEqual([1, 2, 3, -1, -2, -3]);
    } finally {
      store.close();
    }
  });

  test("rejects unknown tensors, wrong dtypes, and out-of-range rows", () => {
    const store = WeightStore.open(dir);
    try {
      expect(() => store.shape("missing")).toThrow();
      expect(() => store.readVector("c")).toThrow();
      expect(() => store.readRowsBf16("a", 1, 2)).toThrow();
      expect(() => store.readVector("a")).toThrow(); // 2-D tensor is not a vector
    } finally {
      store.close();
    }
  });
});
