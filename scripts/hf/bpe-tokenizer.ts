/**
 * Hugging Face byte-level BPE (tokenizer.json) -> runtime vocabulary, as
 * tools/import_qwen2.py export_tokenizer did, generalized:
 *
 *   old = [id of the alphabet char of byte b, b = 0..255] + [BOS id, EOS id]
 *         + every other id in 0..vocabSize-1, ascending;   runtime id n = index in old
 *
 * Base pieces decode through the GPT-2 byte alphabet (flag 0); every added
 * token is its UTF-8 content (flag 1); ids without a token become
 * "<unused:OLD>" (flag 1). With flagUnreachable, base entries that are neither
 * a byte nor any merge's output (placeholders) are flagged too.
 */

export interface AddedToken {
  readonly id: number;
  readonly content: string;
}

export interface HfBpe {
  readonly vocab: ReadonlyMap<string, number>;
  readonly merges: readonly (readonly [string, string])[];
  readonly added: readonly AddedToken[];
}

export interface BuildOptions {
  readonly vocabSize: number;
  /** contents of the added tokens mapped to runtime ids 256 and 257 */
  readonly bos: string;
  readonly eos: string;
  readonly flagUnreachable: boolean;
}

export interface BuiltTokenizer {
  readonly pieces: readonly Uint8Array[];
  readonly special: Uint8Array;
  readonly merges: Uint32Array;
  /** runtime id -> source id (also the embedding row map) */
  readonly oldIds: Uint32Array;
  /** source id -> runtime id */
  readonly toNew: Uint32Array;
  readonly stats: {
    readonly base: number;
    readonly added: number;
    readonly unused: number;
    /** source ids of flagged-or-not base entries that no merge produces and that are not bytes */
    readonly unreachable: readonly number[];
  };
}

const fail = (text: string): never => { throw new Error(`tokenizer.json: ${text}`); };
const LONE_SURROGATE = /[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]/;

/** GPT-2 bytes_to_unicode: byte -> one-character string. */
export function byteAlphabet(): readonly string[] {
  const out: string[] = new Array<string>(256);
  let extra = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    out[b] = String.fromCodePoint(printable ? b : 256 + extra++);
  }
  return out;
}

const isId = (x: unknown): x is number => typeof x === "number" && Number.isSafeInteger(x) && x >= 0;

function readMerge(m: unknown, i: number): readonly [string, string] {
  if (typeof m === "string") {
    const parts = m.split(" ");
    if (parts.length !== 2 || !parts[0] || !parts[1]) fail(`malformed merge ${i}: ${JSON.stringify(m)}`);
    return [parts[0]!, parts[1]!];
  }
  if (Array.isArray(m) && m.length === 2 && typeof m[0] === "string" && typeof m[1] === "string" && m[0] && m[1])
    return [m[0], m[1]];
  return fail(`malformed merge ${i}`);
}

/** Narrows the parts of tokenizer.json this exporter reads. Model-specific gates live with each model. */
export function readHfBpe(json: unknown): HfBpe {
  const t = (json ?? {}) as Record<string, unknown>;
  const model = (t.model ?? {}) as Record<string, unknown>;
  if (model.type !== "BPE") fail("model.type must be BPE");
  if (typeof model.vocab !== "object" || model.vocab === null || Array.isArray(model.vocab)) fail("model.vocab must be an object");
  const vocab = new Map<string, number>();
  const ids = new Set<number>();
  for (const [piece, id] of Object.entries(model.vocab as Record<string, unknown>)) {
    if (!isId(id) || ids.has(id) || !piece) fail(`invalid or duplicate vocab id for ${JSON.stringify(piece)}`);
    vocab.set(piece, id as number);
    ids.add(id as number);
  }
  if (!Array.isArray(model.merges)) fail("model.merges must be an array");
  const merges = (model.merges as unknown[]).map(readMerge);
  if (!Array.isArray(t.added_tokens)) fail("added_tokens must be an array");
  const added = (t.added_tokens as unknown[]).map((x, i): AddedToken => {
    const a = (x ?? {}) as Record<string, unknown>;
    if (!isId(a.id) || typeof a.content !== "string" || !a.content || LONE_SURROGATE.test(a.content))
      fail(`invalid added_tokens[${i}]`);
    return { id: a.id as number, content: a.content as string };
  });
  if (new Set(added.map((a) => a.id)).size !== added.length) fail("added_tokens repeats an id");
  return { vocab, merges, added };
}

function specialId(added: readonly AddedToken[], content: string): number | undefined {
  const found = added.filter((a) => a.content === content);
  if (found.length > 1) fail(`added token ${content} appears more than once`);
  return found[0]?.id;
}

function permutation(hf: HfBpe, o: BuildOptions, alphabet: readonly string[]): { oldIds: Uint32Array; toNew: Uint32Array } {
  const bos = specialId(hf.added, o.bos), eos = specialId(hf.added, o.eos);
  const missing = [[o.bos, bos], [o.eos, eos]].filter(([, id]) => id === undefined).map(([c]) => c);
  if (missing.length) fail(`missing special tokens: ${missing.join(", ")}`);
  const head = alphabet.map((ch) => hf.vocab.get(ch) ?? fail(`byte character U+${ch.codePointAt(0)!.toString(16)} missing`));
  head.push(bos!, eos!);
  const oldIds = new Uint32Array(o.vocabSize);
  const toNew = new Uint32Array(o.vocabSize).fill(0xffffffff);
  head.forEach((id, n) => {
    if (id >= o.vocabSize || toNew[id] !== 0xffffffff) fail(`byte/BOS/EOS id ${id} is out of range or repeated`);
    oldIds[n] = id;
    toNew[id] = n;
  });
  let n = head.length;
  for (let id = 0; id < o.vocabSize; id++) if (toNew[id] === 0xffffffff) { oldIds[n] = id; toNew[id] = n++; }
  return { oldIds, toNew };
}

export function buildTokenizer(hf: HfBpe, o: BuildOptions): BuiltTokenizer {
  const alphabet = byteAlphabet();
  const inverse = new Map(alphabet.map((ch, b) => [ch, b]));
  const { oldIds, toNew } = permutation(hf, o, alphabet);
  const pieces = new Array<Uint8Array | undefined>(o.vocabSize).fill(undefined);
  const special = new Uint8Array(o.vocabSize).fill(1);
  for (const [piece, id] of hf.vocab) {
    if (id >= o.vocabSize) fail(`vocab id ${id} is not below vocab_size ${o.vocabSize}`);
    const bytes: number[] = [];
    for (const ch of piece) bytes.push(inverse.get(ch) ?? fail(`token ${id}: character outside the byte alphabet`));
    pieces[toNew[id]!] = Uint8Array.from(bytes);
    special[toNew[id]!] = 0;
  }
  const addedIds = new Set<number>();
  for (const a of hf.added) {
    if (a.id >= o.vocabSize) fail(`added token id ${a.id} is not below vocab_size`);
    pieces[toNew[a.id]!] = new TextEncoder().encode(a.content);
    special[toNew[a.id]!] = 1;
    addedIds.add(a.id);
  }
  const merges = new Uint32Array(hf.merges.length * 3);
  const outputs = new Set<number>();
  hf.merges.forEach(([a, b], i) => {
    const ia = hf.vocab.get(a), ib = hf.vocab.get(b), io = hf.vocab.get(a + b);
    if (ia === undefined || ib === undefined || io === undefined) fail(`merge ${i} (${a} ${b}) uses unknown pieces`);
    merges.set([toNew[ia!]!, toNew[ib!]!, toNew[io!]!], i * 3);
    outputs.add(io!);
  });
  const byteIds = new Set(oldIds.subarray(0, 256));
  const unreachable = [...hf.vocab.values()].filter((id) => !addedIds.has(id) && !byteIds.has(id) && !outputs.has(id))
    .sort((x, y) => x - y);
  if (o.flagUnreachable) for (const id of unreachable) special[toNew[id]!] = 1;
  let unused = 0;
  const filled = pieces.map((p, n) => p ?? (++unused, new TextEncoder().encode(`<unused:${oldIds[n]}>`)));
  return {
    pieces: filled, special, merges, oldIds, toNew,
    stats: { base: hf.vocab.size, added: hf.added.length, unused, unreachable },
  };
}
