/**
 * Phi-3 config.json parsing for the independent reference. Only the subset this
 * reference implements is accepted; anything else is rejected rather than
 * silently approximated.
 *
 * Sources: transformers v4.49 configuration_phi3.py / modeling_phi3.py
 * (head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads))
 * and modeling_rope_utils.py (dim = int(head_dim * partial_rotary_factor);
 * len(short_factor) == len(long_factor) == dim // 2).
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

export interface Phi3Config {
  readonly hiddenSize: number;
  readonly intermediateSize: number;
  readonly numLayers: number;
  readonly numHeads: number;
  readonly numKvHeads: number;
  readonly headDim: number;
  readonly rotaryDim: number;
  readonly vocabSize: number;
  readonly rmsNormEps: number;
  readonly ropeTheta: number;
  readonly maxPositionEmbeddings: number;
  readonly originalMaxPositionEmbeddings: number;
  readonly shortFactor: readonly number[];
  readonly longFactor: readonly number[];
  /** null when the config has no sliding window. */
  readonly slidingWindow: number | null;
}

type Raw = Record<string, unknown>;

const isRecord = (v: unknown): v is Raw => typeof v === "object" && v !== null && !Array.isArray(v);

function positiveInt(raw: Raw, key: string): number {
  const v = raw[key];
  if (typeof v !== "number" || !Number.isSafeInteger(v) || v <= 0) throw new Error(`config: ${key} must be a positive integer`);
  return v;
}

function positiveNumber(raw: Raw, key: string): number {
  const v = raw[key];
  if (typeof v !== "number" || !Number.isFinite(v) || v <= 0) throw new Error(`config: ${key} must be a positive number`);
  return v;
}

function factorList(scaling: Raw, key: string, length: number): readonly number[] {
  const v = scaling[key];
  if (!Array.isArray(v) || v.length !== length || !v.every((f) => typeof f === "number" && Number.isFinite(f) && f > 0)) {
    throw new Error(`config: rope_scaling.${key} must hold ${length} positive numbers`);
  }
  return Object.freeze([...v] as number[]);
}

export function parsePhi3Config(raw: unknown): Phi3Config {
  if (!isRecord(raw)) throw new Error("config: not a JSON object");
  if (raw.tie_word_embeddings !== true) throw new Error("config: only tied embeddings (lm_head = embed_tokens) are supported");
  if (raw.hidden_act !== "silu") throw new Error(`config: hidden_act ${String(raw.hidden_act)} is not supported`);

  const hiddenSize = positiveInt(raw, "hidden_size");
  const numHeads = positiveInt(raw, "num_attention_heads");
  const numKvHeads = positiveInt(raw, "num_key_value_heads");
  if (numHeads % numKvHeads !== 0) throw new Error("config: num_attention_heads must be a multiple of num_key_value_heads");
  const headDim = raw.head_dim === undefined ? Math.floor(hiddenSize / numHeads) : positiveInt(raw, "head_dim");
  const partial = raw.partial_rotary_factor === undefined ? 1 : positiveNumber(raw, "partial_rotary_factor");
  const rotaryDim = Math.trunc(headDim * partial); // Python int() truncates
  if (rotaryDim % 2 !== 0 || rotaryDim > headDim) throw new Error(`config: rotary dim ${rotaryDim} is not usable`);

  const scaling = raw.rope_scaling;
  if (!isRecord(scaling)) throw new Error("config: rope_scaling missing");
  const ropeType = scaling.rope_type ?? scaling.type;
  if (ropeType !== "longrope") throw new Error(`config: rope type ${String(ropeType)} is not supported`);
  if (scaling.factor !== undefined || scaling.attention_factor !== undefined) {
    throw new Error("config: explicit rope_scaling.factor / attention_factor are not supported by this reference");
  }

  const sliding = raw.sliding_window;
  if (sliding !== undefined && sliding !== null && (typeof sliding !== "number" || !Number.isSafeInteger(sliding))) {
    throw new Error("config: sliding_window must be an integer or null");
  }

  return Object.freeze({
    hiddenSize,
    intermediateSize: positiveInt(raw, "intermediate_size"),
    numLayers: positiveInt(raw, "num_hidden_layers"),
    numHeads,
    numKvHeads,
    headDim,
    rotaryDim,
    vocabSize: positiveInt(raw, "vocab_size"),
    rmsNormEps: positiveNumber(raw, "rms_norm_eps"),
    ropeTheta: positiveNumber(raw, "rope_theta"),
    maxPositionEmbeddings: positiveInt(raw, "max_position_embeddings"),
    originalMaxPositionEmbeddings: positiveInt(raw, "original_max_position_embeddings"),
    shortFactor: factorList(scaling, "short_factor", rotaryDim / 2),
    longFactor: factorList(scaling, "long_factor", rotaryDim / 2),
    slidingWindow: typeof sliding === "number" ? sliding : null,
  });
}

export function loadPhi3Config(modelDir: string): Phi3Config {
  const raw: unknown = JSON.parse(readFileSync(join(modelDir, "config.json"), "utf8"));
  return parsePhi3Config(raw);
}
