/**
 * Synthetic SMOEQ8 checkpoints for tests: random but seeded weights in the
 * exact on-disk layout of docs/FORMAT.md, plus an SMBPE1 tokenizer and config.
 * These exercise routed/shared experts, which no bundled trained model has.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export interface ModelConfig {
  readonly architecture: "smolmoe_v1" | "smolmoe_qwen2";
  readonly dim: number;
  readonly hidden_dim: number;
  readonly layers: number;
  readonly heads: number;
  readonly kv_heads: number;
  readonly experts: number;
  readonly top_k: number;
  readonly vocab_size: number;
  readonly max_context: number;
  readonly shared_experts: number;
  readonly bos_id: number;
  readonly eos_id: number;
  readonly rms_eps: number;
  readonly rope_theta: number;
}

export const BASE_CONFIG: ModelConfig = {
  architecture: "smolmoe_v1",
  dim: 32,
  hidden_dim: 48,
  layers: 2,
  heads: 4,
  kv_heads: 2,
  experts: 8,
  top_k: 2,
  vocab_size: 260,
  max_context: 512,
  shared_experts: 1,
  bos_id: 256,
  eos_id: 257,
  rms_eps: 1e-5,
  rope_theta: 10000,
};

export const DTYPE_F32 = 0;
export const DTYPE_Q8 = 1;
const HEADER_BYTES = 16;
const RECORD_BYTES = 72;
const NAME_BYTES = 40;

export interface TensorShape {
  readonly name: string;
  readonly rows: number;
  readonly cols: number;
  readonly dtype: number;
}

export interface Fixture {
  readonly dir: string;
  readonly model: string;
  readonly config: string;
  readonly tokenizer: string;
  readonly settings: ModelConfig;
  readonly tensors: readonly PlacedTensor[];
}

export interface PlacedTensor extends TensorShape {
  readonly offset: number;
  readonly bytes: number;
}

export function tensorShapes(c: ModelConfig): TensorShape[] {
  const d = c.dim, h = c.hidden_dim, kd = c.kv_heads * (c.dim / c.heads);
  const list: TensorShape[] = [
    { name: "embed", rows: c.vocab_size, cols: d, dtype: DTYPE_Q8 },
    { name: "final_norm", rows: 1, cols: d, dtype: DTYPE_F32 },
    { name: "lm_head", rows: c.vocab_size, cols: d, dtype: DTYPE_Q8 },
  ];
  for (let l = 0; l < c.layers; l++) {
    const p = `l.${l}.`;
    list.push(
      { name: p + "attn_norm", rows: 1, cols: d, dtype: DTYPE_F32 },
      { name: p + "ffn_norm", rows: 1, cols: d, dtype: DTYPE_F32 },
      { name: p + "q", rows: d, cols: d, dtype: DTYPE_Q8 },
      { name: p + "k", rows: kd, cols: d, dtype: DTYPE_Q8 },
      { name: p + "v", rows: kd, cols: d, dtype: DTYPE_Q8 },
      { name: p + "o", rows: d, cols: d, dtype: DTYPE_Q8 },
      { name: p + "router", rows: c.experts, cols: d, dtype: DTYPE_Q8 },
    );
    if (c.architecture === "smolmoe_qwen2") {
      list.push(
        { name: p + "q_bias", rows: 1, cols: d, dtype: DTYPE_F32 },
        { name: p + "k_bias", rows: 1, cols: kd, dtype: DTYPE_F32 },
        { name: p + "v_bias", rows: 1, cols: kd, dtype: DTYPE_F32 },
      );
    }
    for (const [kind, count] of [["e", c.experts], ["s", c.shared_experts]] as const) {
      for (let e = 0; e < count; e++) {
        list.push(
          { name: `${p}${kind}.${e}.gate`, rows: h, cols: d, dtype: DTYPE_Q8 },
          { name: `${p}${kind}.${e}.up`, rows: h, cols: d, dtype: DTYPE_Q8 },
          { name: `${p}${kind}.${e}.down`, rows: d, cols: h, dtype: DTYPE_Q8 },
        );
      }
    }
  }
  return list.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

export function payloadBytes(t: TensorShape): number {
  return t.dtype === DTYPE_Q8 ? t.rows * Math.ceil(t.cols / 32) * 36 : t.rows * t.cols * 4;
}

/** mulberry32: small, seeded, reproducible. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function roundHalfEven(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Quantizes one row exactly as tools/checkpoint.py and q8pack.c do. */
export function q8Row(values: readonly number[]): Uint8Array {
  const blocks = Math.ceil(values.length / 32);
  const out = new Uint8Array(blocks * 36);
  const view = new DataView(out.buffer);
  for (let b = 0; b < blocks; b++) {
    const block = values.slice(b * 32, b * 32 + 32);
    const scale = Math.fround(Math.max(0, ...block.map(Math.abs)) / 127);
    view.setFloat32(b * 36, scale, true);
    block.forEach((v, j) => {
      const q = scale ? Math.max(-127, Math.min(127, roundHalfEven(v / scale))) : 0;
      view.setInt8(b * 36 + 4 + j, q);
    });
  }
  return out;
}

function tensorPayload(t: TensorShape, random: () => number): Uint8Array {
  const isNorm = t.name.endsWith("norm");
  const isBias = t.name.endsWith("bias");
  if (t.dtype === DTYPE_F32) {
    const out = new Uint8Array(payloadBytes(t));
    const view = new DataView(out.buffer);
    for (let i = 0; i < t.rows * t.cols; i++) {
      const v = isNorm ? 0.8 + 0.4 * random() : isBias ? (random() - 0.5) * 0.2 : random() - 0.5;
      view.setFloat32(i * 4, v, true);
    }
    return out;
  }
  const spread = t.name.endsWith("router") ? 4 : t.name === "embed" || t.name === "lm_head" ? 2 : 1;
  const rows: Uint8Array[] = [];
  for (let r = 0; r < t.rows; r++) {
    const values = Array.from({ length: t.cols }, () => (random() - 0.5) * spread / Math.sqrt(t.cols / 16));
    rows.push(q8Row(values));
  }
  return Buffer.concat(rows);
}

export function configText(c: ModelConfig): string {
  return Object.entries(c).map(([k, v]) => `${k}=${v}`).join("\n") + "\n";
}

/** SMBPE1: merge 258 = "H"+"i", 259 = 258+"!", further merges pair two bytes. */
export function tokenizerText(vocab: number): string {
  const merges = ["72 105", "258 33"];
  for (let k = 0; k < vocab - 260; k++) merges.push(`${97 + (k % 26)} ${32 + Math.floor(k / 26)}`);
  return ["SMBPE1", ...merges.slice(0, vocab - 258)].join("\n") + "\n";
}

export interface FixtureOptions {
  readonly config?: Partial<ModelConfig>;
  readonly seed?: number;
  /** Extra zero bytes before the first payload (tests 64-bit offsets). */
  readonly gap?: number;
}

export function writeFixture(dir: string, options: FixtureOptions = {}): Fixture {
  const settings: ModelConfig = { ...BASE_CONFIG, ...options.config };
  const shapes = tensorShapes(settings);
  const random = seededRandom(options.seed ?? 1);
  mkdirSync(dir, { recursive: true });
  let offset = HEADER_BYTES + shapes.length * RECORD_BYTES + (options.gap ?? 0);
  const placed: PlacedTensor[] = shapes.map((t) => {
    const p = { ...t, offset, bytes: payloadBytes(t) };
    offset += p.bytes;
    return p;
  });
  const directory = new Uint8Array(HEADER_BYTES + shapes.length * RECORD_BYTES);
  const view = new DataView(directory.buffer);
  directory.set(new TextEncoder().encode("SMOEQ8\0\x01"), 0);
  view.setUint32(8, shapes.length, true);
  placed.forEach((t, i) => {
    const base = HEADER_BYTES + i * RECORD_BYTES;
    const name = new TextEncoder().encode(t.name);
    if (name.length >= NAME_BYTES) throw new Error(`tensor name too long: ${t.name}`);
    directory.set(name, base);
    view.setUint32(base + 40, t.rows, true);
    view.setUint32(base + 44, t.cols, true);
    view.setUint32(base + 48, t.dtype, true);
    view.setBigUint64(base + 56, BigInt(t.offset), true);
    view.setBigUint64(base + 64, BigInt(t.bytes), true);
  });
  const model = join(dir, "model.smoe");
  const parts: Uint8Array[] = [directory, new Uint8Array(options.gap ?? 0), ...placed.map((t) => tensorPayload(t, random))];
  writeFileSync(model, Buffer.concat(parts));
  const config = join(dir, "config.ini");
  writeFileSync(config, configText(settings));
  const tokenizer = join(dir, "tokenizer.bpe");
  writeFileSync(tokenizer, tokenizerText(settings.vocab_size));
  return { dir, model, config, tokenizer, settings, tensors: placed };
}
