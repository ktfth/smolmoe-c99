/**
 * SMOEQ8 layout, writer and TS re-check (a port of src/modelfile.c and the
 * model_init lookups). Cross-checked against the fixture writer and, when
 * built, against smolmoe-bundle's v3 output byte for byte.
 */
import { afterAll, describe, expect, test } from "bun:test";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { type ModelConfig, formatConfig, tensorSpecs } from "../scripts/smoe/config";
import { DTYPE, type Dtype, payloadBytes } from "../scripts/smoe/format";
import {
  type RecordSpec, blobRecord, chooseVersion, encodeDirectory, planContainer, tensorRecord,
} from "../scripts/smoe/layout";
import { quantizeQ8 } from "../scripts/smoe/quant";
import { checkModel, parseContainer, readBlob, readContainer, verifyPayload } from "../scripts/smoe/reader";
import { type Chunk, writeContainer } from "../scripts/smoe/writer";
import { writeFixture } from "./support/fixture";
import { BUNDLE_BINARY, run, tempDir } from "./support/run";

const work = tempDir("convert-container");
afterAll(() => work.dispose());

const SMALL: ModelConfig = {
  architecture: "smolmoe_qwen2", dim: 32, hidden_dim: 40, layers: 2, heads: 4, kv_heads: 2, experts: 2, top_k: 1,
  vocab_size: 260, max_context: 64, shared_experts: 1, bos_id: 256, eos_id: 257, rms_eps: 1e-6, rope_theta: 10000,
};

let seed = 1;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32 - 0.5; };

/** Records and payloads of a valid tied model for SMALL. */
function smallModel(): { records: RecordSpec[]; payload: Map<string, Uint8Array> } {
  const records: RecordSpec[] = [];
  const payload = new Map<string, Uint8Array>();
  for (const t of tensorSpecs(SMALL)) {
    if (t.name === "lm_head") { records.push(tensorRecord("lm_head", t.rows, t.cols, DTYPE.Q8, 1)); continue; }
    const x = Float32Array.from({ length: t.rows * t.cols }, random);
    const bytes = t.kind === "vector" ? new Uint8Array(x.buffer) : quantizeQ8(x, t.rows, t.cols);
    records.push(tensorRecord(t.name, t.rows, t.cols, t.kind === "vector" ? DTYPE.F32 : DTYPE.Q8));
    payload.set(t.name, bytes);
  }
  return { records, payload };
}

async function writeModel(path: string, records: readonly RecordSpec[], payload: ReadonlyMap<string, Uint8Array>,
  layout: "compact" | "aligned64", version = chooseVersion(records)): Promise<number> {
  const plan = planContainer(records, layout);
  return writeContainer(path, version, plan, (r): Chunk[] => [payload.get(r.name)!]);
}

describe("layout", () => {
  const { records } = smallModel();

  test("compact: contiguous payloads right after the directory; the alias shares embed's extent", () => {
    const plan = planContainer(records, "compact");
    let next = 16 + 72 * plan.length;
    for (const r of plan) {
      if (r.alias) continue;
      expect(r.offset).toBe(next);
      next = r.offset + r.bytes;
    }
    const embed = plan.find((r) => r.name === "embed")!, head = plan.find((r) => r.name === "lm_head")!;
    expect([head.offset, head.bytes, head.alias]).toEqual([embed.offset, embed.bytes, 1]);
  });

  test("aligned64: every payload starts on a 64-byte boundary", () => {
    const plan = planContainer([...records, blobRecord("__config__", 13)].sort((a, b) => (a.name < b.name ? -1 : 1)),
      "aligned64");
    expect(plan.every((r) => r.offset % 64 === 0)).toBe(true);
    expect(plan[0]!.name).toBe("__config__");
  });

  test("version: v3 with blobs, v2 with an alias, v1 otherwise", () => {
    expect(chooseVersion(records)).toBe(2);
    expect(chooseVersion(records.filter((r) => !r.alias))).toBe(1);
    expect(chooseVersion([blobRecord("__tokenizer__", 4), ...records])).toBe(3);
  });

  test.each([
    ["unsorted names", (r: RecordSpec[]) => [r[1]!, r[0]!, ...r.slice(2)], /sorted/],
    ["duplicate names", (r: RecordSpec[]) => [r[0]!, ...r], /sorted/],
    ["an alias other than lm_head", (r: RecordSpec[]) => r.map((x) => (x.name === "final_norm" ? { ...x, alias: 1 as const } : x)), /alias/],
    ["an alias with a different shape", (r: RecordSpec[]) => r.map((x) => (x.name === "lm_head" ? { ...x, rows: 259, bytes: payloadBytes(DTYPE.Q8, 259, 32) } : x)), /alias/],
    ["a tensor named like a blob", (r: RecordSpec[]) => [tensorRecord("__x", 1, 4, DTYPE.F32), ...r], /__/],
    ["a blob without the __ prefix", (r: RecordSpec[]) => [blobRecord("abc", 1), ...r], /blob/],
    ["a 40-byte name", (r: RecordSpec[]) => [tensorRecord("a".repeat(40), 1, 4, DTYPE.F32), ...r], /name/],
  ])("rejects %s", (_what, mutate, error) => {
    expect(() => planContainer(mutate([...records]), "compact")).toThrow(error);
  });
});

describe("writer", () => {
  test("v1 compact reproduces the fixture writer byte for byte", async () => {
    const fx = writeFixture(join(work.path, "fx"), { config: { experts: 3 }, seed: 9 });
    const original = readFileSync(fx.model);
    const parsed = parseContainer(original, original.length);
    const records = parsed.records.map((r) => tensorRecord(r.name, r.rows, r.cols, r.dtype as Dtype));
    const payload = new Map(parsed.records.map((r) => [r.name, original.subarray(r.offset, r.offset + r.bytes)]));
    const out = join(work.path, "fx-copy.smoe");
    expect(await writeModel(out, records, payload, "compact", 1)).toBe(original.length);
    expect(readFileSync(out).equals(original)).toBe(true);
  });

  test("a producer that emits the wrong byte count is refused", async () => {
    const plan = planContainer([tensorRecord("embed", 2, 32, DTYPE.Q8)], "compact");
    await expect(writeContainer(join(work.path, "short.smoe"), 1, plan, () => [new Uint8Array(10)]))
      .rejects.toThrow(/embed: wrote 10 bytes, expected 72/);
  });

  test("asynchronous chunks are written in plan order", async () => {
    const plan = planContainer([tensorRecord("a", 1, 2, DTYPE.F32), tensorRecord("b", 1, 2, DTYPE.F32)], "compact");
    const delayed = (v: number, ms: number) => () => Bun.sleep(ms).then(() => new Uint8Array(8).fill(v));
    const path = join(work.path, "order.smoe");
    await writeContainer(path, 1, plan, (r) => [r.name === "a" ? delayed(1, 30) : delayed(2, 1)], 4);
    expect([...readFileSync(path).subarray(16 + 144)]).toEqual([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]);
  });

  test.skipIf(!existsSync(BUNDLE_BINARY))("v3 aligned64 with blobs equals smolmoe-bundle's output", async () => {
    const fx = writeFixture(join(work.path, "fxb"), { config: { experts: 2 }, seed: 4 });
    const bundled = join(work.path, "bundled.smoe");
    const r = run(["--model", fx.model, "--config", fx.config, "--tokenizer", fx.tokenizer, "--out", bundled], BUNDLE_BINARY);
    expect(r.stderr).toContain("container_bytes");
    const original = readFileSync(fx.model);
    const parsed = parseContainer(original, original.length);
    const blobs = new Map([["__config__", readFileSync(fx.config)], ["__tokenizer__", readFileSync(fx.tokenizer)]]);
    const records = [...[...blobs].map(([n, b]) => blobRecord(n, b.length)),
      ...parsed.records.map((x) => tensorRecord(x.name, x.rows, x.cols, x.dtype as Dtype))];
    const payload = new Map([...blobs, ...parsed.records.map((x) => [x.name, original.subarray(x.offset, x.offset + x.bytes)] as const)]);
    const out = join(work.path, "mine-v3.smoe");
    await writeModel(out, records, payload, "aligned64");
    expect(readFileSync(out).equals(readFileSync(bundled))).toBe(true);
  });
});

describe("reader (port of modelfile_parse / verify / model_init)", () => {
  const path = join(work.path, "small.smoe");
  let bytes: Buffer;

  test("accepts a valid tied model and finds its blobs", async () => {
    const { records, payload } = smallModel();
    const cfg = new TextEncoder().encode(formatConfig(SMALL));
    payload.set("__config__", cfg);
    await writeModel(path, [blobRecord("__config__", cfg.length), ...records], payload, "aligned64");
    bytes = readFileSync(path);
    const parsed = readContainer(path);
    expect([parsed.version, parsed.count, parsed.blobs]).toEqual([3, records.length + 1, 1]);
    verifyPayload(path, parsed);
    checkModel(SMALL, parsed);
    expect(new TextDecoder().decode(readBlob(path, parsed, "__config__")!)).toBe(formatConfig(SMALL));
    expect(readBlob(path, parsed, "__tokenizer__")).toBeNull();
  });

  const at = (i: number) => 16 + 72 * i;
  const mutations: [string, (b: Buffer) => void, RegExp][] = [
    ["bad magic", (b) => { b[0] = 0x58; }, /SMOEQ8/],
    ["version 4", (b) => { b[7] = 4; }, /SMOEQ8/],
    ["nonzero reserved field", (b) => { b.writeUInt32LE(1, 12); }, /SMOEQ8/],
    ["zero records", (b) => { b.writeUInt32LE(0, 8); }, /tensor count/],
    ["a directory past the end", (b) => { b.writeUInt32LE(1_000_000, 8); }, /truncated/],
    ["an unterminated name", (b) => { b.fill(0x61, at(1), at(1) + 40); }, /tensor name/],
    ["an empty name", (b) => { b[at(1)] = 0; }, /tensor name/],
    ["a blob in v2", (b) => { b[7] = 2; }, /invalid blob record/],
    ["an oversized column count", (b) => { b.writeUInt32LE(65537, at(1) + 44); }, /metadata|extent/],
    ["dtype 4", (b) => { b.writeUInt32LE(4, at(1) + 48); }, /metadata/],
    ["a wrong byte count", (b) => { b.writeBigUInt64LE(BigInt(b.readBigUInt64LE(at(1) + 64)) + 1n, at(1) + 64); }, /extent/],
    ["overlapping payloads", (b) => { b.writeBigUInt64LE(b.readBigUInt64LE(at(1) + 56), at(2) + 56); }, /extent/],
    ["a payload past the end", (b) => { b.writeBigUInt64LE(BigInt(b.length), at(2) + 56); }, /extent/],
    ["unsorted names", (b) => { b[at(2)] = 0x7a; }, /sorted/],
  ];
  test.each(mutations)("rejects %s", (_what, mutate, error) => {
    const b = Buffer.from(bytes);
    mutate(b);
    expect(() => parseContainer(b, b.length)).toThrow(error);
  });

  test("rejects an alias that does not match embed and an alias in v1", async () => {
    const parsed = readContainer(path);
    const head = parsed.records.findIndex((r) => r.name === "lm_head");
    const b = Buffer.from(bytes);
    b.writeBigUInt64LE(b.readBigUInt64LE(at(head) + 56) + 64n, at(head) + 56);
    expect(() => parseContainer(b, b.length)).toThrow(/alias/);
    const { records, payload } = smallModel();
    const v2 = join(work.path, "v2.smoe");
    await writeModel(v2, records, payload, "compact");
    const v1 = Buffer.from(readFileSync(v2));
    expect(v1[7]).toBe(2);
    expect(() => parseContainer(v1, v1.length)).not.toThrow();
    v1[7] = 1;
    expect(() => parseContainer(v1, v1.length)).toThrow(/invalid tensor metadata: lm_head/);
  });

  test("payload verification: negative or NaN Q8 scales and NaN F32 values fail; -0.0 passes", () => {
    const parsed = readContainer(path);
    const q = parsed.records.find((r) => r.name === "l.0.q")!, norm = parsed.records.find((r) => r.name === "final_norm")!;
    const check = (patch: (b: Buffer) => void) => {
      const b = Buffer.from(bytes);
      patch(b);
      const p = join(work.path, "mutated.smoe");
      writeFileSync(p, b);
      return () => verifyPayload(p, readContainer(p));
    };
    expect(check((b) => b.writeFloatLE(-1, q.offset + 36))).toThrow(/l\.0\.q/);
    expect(check((b) => b.writeFloatLE(Number.NaN, q.offset))).toThrow(/l\.0\.q/);
    expect(check((b) => b.writeFloatLE(Number.POSITIVE_INFINITY, norm.offset + 4))).toThrow(/final_norm/);
    expect(check((b) => b.writeFloatLE(-0, q.offset))).not.toThrow();
  });

  test("model check: tensor count, shapes and dtype classes follow the config", () => {
    const parsed = readContainer(path);
    expect(() => checkModel({ ...SMALL, layers: 1 }, parsed)).toThrow(/tensor count/);
    expect(() => checkModel({ ...SMALL, dim: 64, heads: 8 }, parsed)).toThrow(/shape|type/);
    const swapped = { ...parsed, records: parsed.records.map((r) => (r.name === "final_norm" ? { ...r, dtype: DTYPE.Q8 as Dtype } : r)) };
    expect(() => checkModel(SMALL, swapped)).toThrow(/final_norm/);
    const missing = { ...parsed, records: parsed.records.map((r) => (r.name === "l.1.o" ? { ...r, name: "l.1.oo" } : r)) };
    expect(() => checkModel(SMALL, missing)).toThrow(/missing tensor: l\.1\.o/);
  });

  test("directory encoding is the documented 72-byte record", () => {
    const plan = planContainer([tensorRecord("embed", 3, 33, DTYPE.Q8)], "compact");
    const dir = Buffer.from(encodeDirectory(1, plan));
    expect(dir.subarray(0, 8).toString("latin1")).toBe("SMOEQ8\0\x01");
    expect([dir.readUInt32LE(8), dir.readUInt32LE(16 + 40), dir.readUInt32LE(16 + 44), dir.readUInt32LE(16 + 48)])
      .toEqual([1, 3, 33, 1]);
    expect([Number(dir.readBigUInt64LE(16 + 56)), Number(dir.readBigUInt64LE(16 + 64))]).toEqual([88, 3 * 2 * 36]);
  });
});
