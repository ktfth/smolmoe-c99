/**
 * Whole-model re-check, as the runtime would load it: container directory,
 * payload scan, config (explicit or the __config__ blob), tokenizer (explicit
 * or __tokenizer__; SMQBP1 files are fully validated) and the tensor set.
 */
import { BLOB_CONFIG, BLOB_TOKENIZER } from "./format";
import { type ModelConfig, parseConfig } from "./config";
import { parseQbp } from "./qbp";
import { type ParsedContainer, checkModel, readBlob, readContainer, verifyPayload } from "./reader";

export interface CheckInput {
  readonly model: string;
  readonly config?: Uint8Array;
  readonly tokenizer?: Uint8Array;
}

export interface CheckReport {
  readonly container: ParsedContainer;
  readonly config: ModelConfig;
  readonly tokenizer: { readonly kind: string; readonly version?: number; readonly merges?: number };
}

const QBP = "SMQBP1\0";

const ascii = (bytes: Uint8Array, n: number) => String.fromCharCode(...bytes.subarray(0, n));
const tokenizerKind = (bytes: Uint8Array): string => ascii(bytes, 6);

export function checkFiles(input: CheckInput): CheckReport {
  const container = readContainer(input.model);
  verifyPayload(input.model, container);
  const configBytes = input.config ?? readBlob(input.model, container, BLOB_CONFIG);
  if (!configBytes) throw new Error("no config: pass --config or embed __config__");
  const config = parseConfig(configBytes);
  const tok = input.tokenizer ?? readBlob(input.model, container, BLOB_TOKENIZER);
  if (!tok) throw new Error("no tokenizer: pass --tokenizer or embed __tokenizer__");
  let tokenizer: CheckReport["tokenizer"] = { kind: tokenizerKind(tok) };
  if (ascii(tok, 7) === QBP) {
    const parsed = parseQbp(tok, config.vocab_size);
    tokenizer = { kind: "SMQBP1", version: parsed.header.version, merges: parsed.merges.length / 3 };
  }
  checkModel(config, container);
  return { container, config, tokenizer };
}
