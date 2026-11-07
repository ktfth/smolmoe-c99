/**
 * config.ini: Python-identical writer, a port of src/config.c's parser and
 * validation, and the tensor set the runtime expects (src/model.c).
 *
 * Two optional keys extend the 15 required ones: rotary_dim (rotate only the
 * first N coordinates of each head) and rope_attention_factor (scale of RoPE
 * cos/sin). Writers append them, when present, after the required keys.
 */
import { compareNames } from "./format";

export type Architecture = "smolmoe_v1" | "smolmoe_qwen2";

export interface ModelConfig {
  readonly architecture: Architecture;
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
  readonly rotary_dim?: number;
  readonly rope_attention_factor?: number;
}

type IntKey = "dim" | "hidden_dim" | "layers" | "heads" | "kv_heads" | "experts" | "top_k" | "vocab_size"
  | "max_context" | "shared_experts" | "bos_id" | "eos_id" | "rotary_dim";
type RealKey = "rms_eps" | "rope_theta" | "rope_attention_factor";

/** Required keys, in the order Python's checkpoint.BASE_CONFIG writes them. */
export const REQUIRED_KEYS = ["architecture", "dim", "hidden_dim", "layers", "heads", "kv_heads", "experts", "top_k",
  "vocab_size", "max_context", "shared_experts", "bos_id", "eos_id", "rms_eps", "rope_theta"] as const;
export const OPTIONAL_KEYS = ["rotary_dim", "rope_attention_factor"] as const;
const REQUIRED_INT_KEYS: readonly Exclude<IntKey, "rotary_dim">[] = ["dim", "hidden_dim", "layers", "heads",
  "kv_heads", "experts", "top_k", "vocab_size", "max_context", "shared_experts", "bos_id", "eos_id"];
const REAL_KEYS: ReadonlySet<string> = new Set<RealKey>(["rms_eps", "rope_theta", "rope_attention_factor"]);

export const MAX_CONTEXT = 1_048_576;
const MAX_DIM = 65536;
const MAX_MERGES = 262144;
const MAX_ROPE_FACTOR = 1000;
export const BOS_ID = 256;
export const EOS_ID = 257;
const LINE_CAPACITY = 512;
const DECIMAL = /^[0-9]+$/;
const REAL = /^[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/;

/** Python repr() of a finite float: shortest round-trip digits, Python's exponent style. */
export function pyFloatRepr(x: number): string {
  if (!Number.isFinite(x)) throw new Error(`non-finite: ${x}`);
  if (x === 0) return Object.is(x, -0) ? "-0.0" : "0.0";
  const [mant, expText] = x.toExponential().split("e") as [string, string];
  const exp = Number(expText);
  const sign = x < 0 ? "-" : "";
  const digits = mant.replace("-", "").replace(".", "");
  if (exp >= -4 && exp < 16) {
    if (exp < 0) return `${sign}0.${"0".repeat(-exp - 1)}${digits}`;
    return `${sign}${digits.slice(0, exp + 1).padEnd(exp + 1, "0")}.${digits.slice(exp + 1) || "0"}`;
  }
  const m = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
  return `${sign}${m}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
}

const isU32 = (n: number, max: number) => Number.isSafeInteger(n) && n >= 0 && n <= max;

/** The checks of config_parse (src/config.c) plus the optional RoPE keys. Throws on failure. */
export function validateConfig(c: ModelConfig): ModelConfig {
  if (c.architecture !== "smolmoe_v1" && c.architecture !== "smolmoe_qwen2")
    throw new Error(`unsupported architecture: ${String(c.architecture)}`);
  for (const k of REQUIRED_INT_KEYS) if (!isU32(c[k], MAX_CONTEXT)) throw new Error(`invalid ${k}: ${c[k]}`);
  for (const k of ["rms_eps", "rope_theta"] as const) {
    if (typeof c[k] !== "number") throw new Error(`invalid ${k}: ${String(c[k])}`);
  }
  const eps = Math.fround(c.rms_eps), theta = Math.fround(c.rope_theta);
  if (!c.dim || c.dim > MAX_DIM || !c.hidden_dim || c.hidden_dim > MAX_DIM || !c.layers || c.layers > 1024 ||
    !c.heads || c.heads > c.dim || c.dim % c.heads || (c.dim / c.heads) % 2 ||
    !c.kv_heads || c.heads % c.kv_heads || !c.experts || c.experts > 4096 ||
    !c.top_k || c.top_k > c.experts || c.shared_experts > 16 || c.vocab_size < 258 || c.vocab_size > 258 + MAX_MERGES ||
    !c.max_context || c.max_context > MAX_CONTEXT || c.bos_id !== BOS_ID || c.eos_id !== EOS_ID ||
    !(eps > 0 && eps < 1) || !(theta > 1) || !Number.isFinite(theta))
    throw new Error("invalid config dimensions or special token IDs");
  if (c.rotary_dim !== undefined) {
    const hd = c.dim / c.heads;
    if (!isU32(c.rotary_dim, MAX_DIM) || c.rotary_dim < 2 || c.rotary_dim % 2 || c.rotary_dim > hd)
      throw new Error(`invalid rotary_dim: ${c.rotary_dim} (must be even, 2..${hd})`);
  }
  /* kept as a double by the runtime (src/config.c), so no float cast here */
  if (c.rope_attention_factor !== undefined && !(c.rope_attention_factor > 0 && c.rope_attention_factor <= MAX_ROPE_FACTOR))
    throw new Error(`invalid rope_attention_factor: ${c.rope_attention_factor} (must be in (0, ${MAX_ROPE_FACTOR}])`);
  return c;
}

/** `key=value\n` lines; floats in Python repr (byte-identical to tools/checkpoint.py write_config). */
export function formatConfig(c: ModelConfig): string {
  validateConfig(c);
  const keys = [...REQUIRED_KEYS, ...OPTIONAL_KEYS.filter((k) => c[k] !== undefined)];
  return keys.map((k) => {
    const v = c[k]!;
    return `${k}=${typeof v === "string" ? v : REAL_KEYS.has(k) ? pyFloatRepr(v) : String(v)}\n`;
  }).join("");
}

const trim = (s: string) => s.replace(/^[ \t\r\n]+|[ \t\r\n]+$/g, "");

function parseU32(value: string, key: string): number {
  const n = DECIMAL.test(value) ? Number(value) : Number.NaN;
  if (!(n <= MAX_CONTEXT)) throw new Error(`invalid ${key}: ${value}`);
  return n;
}

function parseReal(value: string, key: string): number {
  const n = REAL.test(value) ? Number(value) : Number.NaN;
  const underflow = n === 0 && /[1-9]/.test(value.split(/[eE]/)[0]!);
  if (!Number.isFinite(n) || underflow) throw new Error(`invalid ${key}: ${value}`);
  return n;
}

/** Splits like br_line: lines of at most 511 bytes including the newline, no NUL bytes. */
function configLines(bytes: Uint8Array): string[] {
  if (bytes.includes(0)) throw new Error("config contains a NUL byte");
  const lines: string[] = [];
  for (let start = 0; start < bytes.length;) {
    const nl = bytes.indexOf(10, start);
    const end = nl < 0 ? bytes.length : nl + 1;
    if (end - start > LINE_CAPACITY - 1) throw new Error("config line too long");
    lines.push(String.fromCharCode(...bytes.subarray(start, end))); /* byte-exact, like the C parser */
    start = end;
  }
  return lines;
}

/** Port of config_parse: strict key=value lines, '#' comments, every required key exactly once. */
export function parseConfig(text: string | Uint8Array): ModelConfig {
  const bytes = typeof text === "string" ? new TextEncoder().encode(text) : text;
  const known: ReadonlySet<string> = new Set<string>([...REQUIRED_KEYS, ...OPTIONAL_KEYS]);
  const values = new Map<string, string | number>();
  for (const raw of configLines(bytes)) {
    const line = trim(raw.split("#")[0]!);
    if (!line) continue;
    const eq = line.indexOf("=");
    if (eq < 0) throw new Error("config requires key=value lines");
    const key = trim(line.slice(0, eq)), value = trim(line.slice(eq + 1));
    if (!known.has(key)) throw new Error(`unknown config key: ${key}`);
    if (values.has(key)) throw new Error(`duplicate config key: ${key}`);
    if (key === "architecture") {
      if (value !== "smolmoe_v1" && value !== "smolmoe_qwen2") throw new Error(`unsupported architecture: ${value}`);
      values.set(key, value);
    } else values.set(key, REAL_KEYS.has(key) ? parseReal(value, key) : parseU32(value, key));
  }
  if (!REQUIRED_KEYS.every((k) => values.has(k))) throw new Error("config is missing required fields");
  const entries = [...REQUIRED_KEYS, ...OPTIONAL_KEYS].filter((k) => values.has(k)).map((k) => [k, values.get(k)]);
  return validateConfig(Object.fromEntries(entries) as unknown as ModelConfig);
}

export interface TensorSpec {
  readonly name: string;
  readonly rows: number;
  readonly cols: number;
  /** matrix: quantized (Q8/Q5), vector: F32 */
  readonly kind: "matrix" | "vector";
}

/** The exact tensor set of model_init, sorted by name (checkpoint.py shapes()). */
export function tensorSpecs(c: ModelConfig): readonly TensorSpec[] {
  const d = c.dim, h = c.hidden_dim, kd = c.kv_heads * (c.dim / c.heads);
  const m = (name: string, rows: number, cols: number): TensorSpec => ({ name, rows, cols, kind: "matrix" });
  const v = (name: string, cols: number): TensorSpec => ({ name, rows: 1, cols, kind: "vector" });
  const specs: TensorSpec[] = [m("embed", c.vocab_size, d), v("final_norm", d), m("lm_head", c.vocab_size, d)];
  for (let l = 0; l < c.layers; l++) {
    const p = `l.${l}.`;
    specs.push(v(`${p}attn_norm`, d), v(`${p}ffn_norm`, d), m(`${p}q`, d, d), m(`${p}k`, kd, d), m(`${p}v`, kd, d),
      m(`${p}o`, d, d), m(`${p}router`, c.experts, d));
    if (c.architecture === "smolmoe_qwen2") specs.push(v(`${p}q_bias`, d), v(`${p}k_bias`, kd), v(`${p}v_bias`, kd));
    for (const [kind, count] of [["e", c.experts], ["s", c.shared_experts]] as const) {
      for (let e = 0; e < count; e++)
        specs.push(m(`${p}${kind}.${e}.gate`, h, d), m(`${p}${kind}.${e}.up`, h, d), m(`${p}${kind}.${e}.down`, d, h));
    }
  }
  return specs.sort((a, b) => compareNames(a.name, b.name));
}

export function expectedTensorCount(c: ModelConfig): number {
  const qkvBias = c.architecture === "smolmoe_qwen2" ? 1 : 0;
  return 3 + c.layers * (7 + 3 * qkvBias + 3 * (c.experts + c.shared_experts));
}
