# smolmoe architectures and binary contract

This specification describes the implemented model architectures, not Kimi K3.
`smolmoe_qwen2` extends the base architecture with F32 bias vectors
`l.N.q_bias` (1 × QD), `l.N.k_bias` and `l.N.v_bias` (1 × KD), added before RoPE
and KV storage. The importer permutes split-half Q/K rows and biases into this
runtime's adjacent-pair RoPE order. Llama 3.x uses the base `smolmoe_v1`
layout (no biases) plus the optional llama3 RoPE keys. `smolmoe_gemma3` adds the Gemma 3 norms,
head dimensions, attention windows and activation described below. All matrix
shapes are `(output rows, input columns)` and row-major. Let D = dim, H =
hidden_dim, V = vocab_size, E = experts, S = shared_experts, A = heads, K =
kv_heads, L = layers, HD = head_dim (default D/A), QD = A × HD, and KD = K × HD.

## Tensor names

| Name | Shape | Encoding |
|---|---|---|
| `embed` | V × D | Q8 |
| `final_norm` | 1 × D | F32 |
| `lm_head` | V × D | Q8, separate from or aliased to `embed` |
| `l.N.attn_norm`, `l.N.ffn_norm` | 1 × D | F32 |
| `l.N.post_attn_norm`, `l.N.post_ffn_norm` | 1 × D | F32, Gemma 3 only |
| `l.N.q_norm`, `l.N.k_norm` | 1 × HD | F32, Gemma 3 only |
| `l.N.q` | QD × D | Q8 |
| `l.N.o` | D × QD | Q8 |
| `l.N.k`, `l.N.v` | KD × D | Q8 |
| `l.N.q_bias` | 1 × QD | F32, Qwen2 only |
| `l.N.k_bias`, `l.N.v_bias` | 1 × KD | F32, Qwen2 only |
| `l.N.router` | E × D | Q8 |
| `l.N.e.I.gate`, `l.N.e.I.up` | H × D | Q8 |
| `l.N.e.I.down` | D × H | Q8 |
| `l.N.s.I.gate`, `l.N.s.I.up` | H × D | Q8 |
| `l.N.s.I.down` | D × H | Q8 |

N is the zero-based layer number, I the zero-based expert number. In the base
MoE layout every layer is routed; there are no special dense-first layers.
Qwen2 has the Q/K/V biases above and may alias its tied output head to `embed`.
Gemma 3 uses one dense expert with a zero router and aliases `lm_head` to
`embed`; it also uses the Gemma-only norm vectors above. The implemented layouts
do not have output gates, latent projections, router bias/correction/grouping,
or a shared-expert scalar gate. The Gemma converter stores each norm vector as
`1 + source_weight`.

## Forward pass

All runtime activations and weight scales are F32. For each token:

1. `x = embed[token]`. Gemma 3 scales this vector by `sqrt(D)`.
2. For each layer, `n = RMSNorm(x, attn_norm, rms_eps)`.
3. `q = Wq n`, `k = Wk n`, `v = Wv n`. Gemma 3 applies `q_norm` and `k_norm`
   to each head before adjacent-pair RoPE. Apply RoPE to q and k.
   Only the first R = `rotary_dim` coordinates of each head rotate (default R = head_dim);
   the rest pass through. For pair j = 0,2,…,R−2, `angle = position / rope_theta^(j/R)`,
   and the pair is rotated by `f·cos(angle)` and `f·sin(angle)` with f = `rope_attention_factor`
   (default 1; Phi-3 "longrope" uses it). The tables are computed in double and rounded to F32.
   This adjacent-pair convention matters when exporting trained checkpoints.
4. Append k and v to the layer's unquantized KV cache. Query head h maps to
   KV head `floor(h / (heads / kv_heads))`.
5. `att = softmax(q K^T / sqrt(head_dim)) V`, causal through current position.
   Gemma 3 applies `post_attn_norm` to `Wo att` before adding it to x.
6. `n = RMSNorm(x, ffn_norm, rms_eps)`; compute all router logits `r = Wr n`.
7. Select highest top_k logits, breaking exact ties by lower expert index.
   `p_i = sigmoid(r_i)`; `w_i = p_i / sum_selected(p)`.
8. For each selected expert in descending logit order,
   `x += w_i * Wdown_i (activation(Wgate_i n) * Wup_i n)`; `activation` is
   SiLU except for Gemma 3, which uses tanh-approximate GELU.
   Then add each shared expert's same MLP output in ascending expert index,
   with coefficient 1. Gemma 3 applies `post_ffn_norm` to its single dense
   expert's output before the residual addition. All experts use the same n from step 6.
9. After all layers, `logits = lm_head RMSNorm(x, final_norm, rms_eps)`.

RMSNorm is `x_i / sqrt(mean(x*x) + eps) * weight_i`.
SiLU is `z/(1+exp(-z))`. There are no residual scaling factors.
The generic router steps above are bypassed for a one-expert model. For Gemma 3,
every sixth layer (5, 11, 17, ...) uses global attention and `rope_theta`; the
other layers use the trailing `sliding_window` positions and `rope_local_theta`.
Model weights producing non-finite intermediates or a zero sum of selected
sigmoid probabilities are rejected; numerically extreme checkpoints are outside
the supported operating range.

Greedy selection uses the lowest token ID on ties, excluding BOS. Positive
temperature uses a full-vocabulary softmax and seeded xorshift32 sampling,
excluding BOS and permitting EOS. No top-p, top-k vocabulary sampling,
repetition penalty, or vocabulary masking is implemented.

## Checkpoint container

Every integer and IEEE F32 value on disk is **little-endian**. No packed C struct
is read directly from disk. Offsets are unsigned 64-bit bytes, limited by the
platform's signed 64-bit POSIX file-offset range. Sorted directory extents must
be non-overlapping and in the same order as directory names.

Header, 16 bytes:

| Offset | Type | Value |
|---|---|---|
| 0 | 8 bytes | `53 4d 4f 45 51 38 00 01` (`SMOEQ8`, NUL, version 1) |
| 8 | uint32 | tensor count |
| 12 | uint32 | zero, reserved |

Then tensor_count × 72-byte directory records:

| Offset | Type | Meaning |
|---|---|---|
| 0 | char[40] | ASCII name, NUL terminated; writer zero pads |
| 40 | uint32 | rows |
| 44 | uint32 | columns |
| 48 | uint32 | dtype 0=F32, 1=Q8, 2=Q5 (runtime >= 0.2) |
| 52 | uint32 | zero, reserved |
| 56 | uint64 | absolute file byte offset |
| 64 | uint64 | exact payload byte count |

Records are lexicographically sorted by name and unique. Payloads follow the
directory in the same order; gaps, including sparse gaps, are allowed. The tensor
directory must match config shapes and tensor count exactly. There is no aliasing,
compression, shard index, or stored executable code.

Version 2 uses header magic `SMOEQ8`, NUL, byte `02`. Its record field at offset
52 is a flag: 0 for ordinary extents, 1 only for `lm_head` aliasing `embed` with
identical offset, byte count, shape and dtype. This is the sole exception to the
version-1 no-alias/monotonically ordered payload rule. Other overlaps are rejected.
The Qwen importer uses this to avoid duplicating its large tied output head.

Q5 (dtype 2, written by `smolmoe-quantize`): blocks of 32 weights, 22 bytes each:

```text
float16_le scale          (signed; the block's extreme value maps toward -16)
uint32_le  high_bits      bit j = bit 4 of element j
uint8[16]  low_nibbles    byte j: element j in bits 0-3, element j+16 in bits 4-7
```

Weight = scale × (q − 16) with q in [0, 31]. Scales must be finite (checked at
load). The converter picks each block scale from 21 candidates around
`extreme / -16` to minimize squared error.

F32: rows × cols × 4 bytes.

Q8: independently quantized blocks of 32 weights within each row:

```text
float32_le scale
int8[32] quantized
```

Each block is 36 bytes; rows have `ceil(cols/32)` blocks, with zero padding in the
last block. Padding never participates in computation. Weight = scale × signed
int8. Scale is finite and nonnegative. The provided writer uses
`scale = max(abs(block))/127`, F32-rounded before quantizing; `round` is ties-to-even,
then clamps to [-127,127]. Zero blocks use zero scale. Finite input whose scale
underflows to zero is encoded as zero. This is a lossy conversion.

### Runtime arithmetic contract (v0.2)

Weights are never dequantized in bulk. Before each matrix product the runtime
quantizes the input activation vector into the same block-32 form:
`scale = max(abs(block)) / 127` (F32), `q = lrintf(x * (1/scale))` clamped to
[-127, 127], zero padding. Activation quantization costs +0.05% perplexity on
Qwen2.5-0.5B (measured against the v0.1 F32-activation engine, see
docs/DECISIONS.md) and enables exact integer dot products.

For each block, the integer products are summed exactly into eight lanes, where
lane `l` holds the products of elements `4l..4l+3`. Each lane accumulates
`(weight_scale * activation_scale) * (float)lane_sum` across blocks in increasing
block order, and the eight lanes are reduced as `((l0+l4)+(l2+l6)) +
((l1+l5)+(l3+l7))`. The scalar kernel and the AVX2 kernel (`maddubs`/`madd`,
separate multiply and add, never FMA) implement this order bit for bit, so
results do not depend on backend, thread count or prompt batch size.
`--fast` keeps the integer lanes and the final reduction, but each lane
update is one FMA, so those logits are not bit-identical to the default.
Reproducibility across operating systems is not promised yet: `expf`, `cos`
and `sin` come from the platform's libm.

F32 vectors (norms, biases) are copied to the heap at load time. Every Q8 scale
and F32 value is checked once at load (finite, scales non-negative), so the
inference loop performs no per-access validation.

## Config

See [models/qwen2.5-0.5b-instruct/config.ini](../models/qwen2.5-0.5b-instruct/config.ini). This is a strict, flat INI-like
`key=value` file with optional `#` comments; no sections. Unknown, duplicate,
missing, invalid, or out-of-range fields are errors. Model context is at most
1,048,576; D and H at most 65,536; layers at most 1024; E at most 4096; S at most
16. A divides D, K divides A, and D/A is positive and even. All tensors and
activations, the KV cache and tokenizer tables must fit `--memory-mib`; mapped
weights are not counted. Without the option the budget is 512 MiB plus the KV
cache of min(context, 2048) positions.

Optional keys are `rotary_dim` (even, 2 to `head_dim`; default `head_dim`),
`rope_attention_factor` (in (0, 1000]; default 1), `head_dim` (default D/A),
`sliding_window` (default 0, meaning full attention), and `rope_local_theta`
(default `rope_theta`). Gemma 3 sets `head_dim`, `sliding_window` and
`rope_local_theta` explicitly. Omitting an optional key gives the stated default.

Llama 3 RoPE scaling (Hugging Face `rope_type: llama3`) takes four optional keys
that must appear together: `rope_llama3_factor` (in [1, 1000]),
`rope_llama3_low_freq_factor` (> 0), `rope_llama3_high_freq_factor` (greater than
the low factor) and `rope_llama3_original_context` (positive). With them, each
global inverse frequency f = theta^(-2i/rotary_dim), of wavelength w = 2π/f, becomes
f when w < original/high, f/factor when w > original/low, and otherwise
(1 − s)·f/factor + s·f with s = (original/w − low)/(high − low). The local
frequencies (`rope_local_theta`) are not scaled. Without the keys nothing changes.

## Tokenizer

`SMBPE1` newline, then exactly `V - 258` merge lines `left_id right_id`.
Output of merge rank r is `258+r`. Inputs must be bytes or earlier merge tokens;
special IDs 256 and 257 and forward references are forbidden. Pairs must be
unique. Encode bytes, then repeatedly merge the available adjacent pair with
lowest rank, choosing the leftmost occurrence on ties. BOS is prepended and is
never merged. There is no text normalization or special token interpretation.
Tokens expanding beyond 65,536 decoded bytes are rejected on tokenizer load.

### Legacy llama2.c score-BPE import

The additional binary tokenizer format starts with an 8-byte magic
`53 4d 53 50 42 31 00 01` (`SMSPB1`, NUL, version 1). It has three little-endian
uint32 header fields: vocabulary size, maximum piece length, and total piece
bytes. Each vocabulary record is F32 score, uint32 byte length, and exactly that
many piece bytes. All records are in the runtime's remapped token-ID order.
Pieces are unique and nonempty; their total must match the header and no trailing
bytes are allowed. Max piece length is 65,536; total piece data at most 64 MiB.

The imported tokenizer stores original textual byte markers `<0x00>` through
`<0xFF>` as piece strings for lookup/merging, and emits the corresponding raw
byte for runtime IDs 0..255 when decoding. The ID permutation, applied to both
embedding/output-head rows and tokenizer records, maps source byte IDs 3..258
to 0..255, source BOS 1 to 256, source EOS 2 to 257, source UNK 0 to 258;
IDs 259 and above are unchanged. No trained token is discarded.

Nonempty prompts receive a dummy-space prefix after BOS. UTF-8 code points
are looked up in the piece table, falling back to byte IDs when absent. Adjacent
piece pairs are repeatedly concatenated and looked up; the available merge with
highest output-piece score wins, with leftmost tie handling. BOS is never merged.
`--tokenize-only` removes the leading dummy space when printing its round-trip.
Normal generation prints continuation bytes directly, without echoing the prompt.

Legacy dense FFNs are imported as `experts=1`, `top_k=1`, `shared_experts=0` with
a zero router matrix, so the sole selected expert's normalized weight is exactly
1. All other operations follow the main forward specification. The imported
model is dense, not a learned sparse MoE model. This llama2.c importer supports
the legacy 7-int header and F32 payload layout only; newer llama2.c formats and
MXFP4 are unsupported by it. Hugging Face conversion is handled separately by
`scripts/convert.ts` (see [MODELS.md](MODELS.md)); Kimi K3 is not supported.


### Byte-level BPE format (SMQBP1)

Used by Qwen2.5 (version 1) and Phi-4-mini, Gemma 3 and Llama 3 (version 2). All integers are
little-endian uint32.

| Offset | Version 1 (24-byte header) | Version 2 (32-byte header) |
|---|---|---|
| 0 | `SMQBP1`, NUL, byte `01` | `SMQBP1`, NUL, byte `02` |
| 8 | vocabulary size (must equal the config's) | same |
| 12 | merge count (at most 1,048,576) | same |
| 16 | total piece bytes (at most 64 MiB) | same |
| 20 | runtime `<\|im_start\|>` ID | normalizer: 0 none, 1 NFC |
| 24 | (records start) | pre-tokenizer: 1 Qwen2, 2 o200k, 3 Gemma, 4 Llama 3 |
| 28 | | chat template: 0 none, 1 ChatML, 2 Phi, 3 Gemma, 4 ChatML-BOS |

Version 1 means NFC, the Qwen2 pre-tokenizer and ChatML, so the published
Qwen2.5 `tokenizer.qbp` loads unchanged. Version 1 files are now also held to
the ChatML checks below (flagged 256 and 257, ID 257 = `<|im_end|>`, the
header's im_start = `<|im_start|>`), which the published file meets. Other
version bytes and enum values are rejected. The records follow the header:

- One piece record per ID: uint32 length (1..65,536), uint32 flag, then the raw
  bytes. Flag 1 marks special, added and unused entries (and untrained
  placeholders such as Phi-4's `<|endoftextNN|>`); they are never merged, print
  nothing and are never sampled, except EOS. IDs 0..255 are the single bytes
  0..255 with flag 0. ID 256 is BOS/end-of-text and ID 257 is EOS/end-of-turn,
  and both must have flag 1.
- One merge record per rank: uint32 left, right and output IDs. Inputs and
  output must be unflagged, the output's bytes must be the concatenation and no
  pair may repeat.

The importer permutes source IDs so that bytes come first and the two control
tokens land on 256 and 257; the same permutation reorders the embedding and the
tied head. Qwen2.5 maps `<|endoftext|>` 151643 to 256 and `<|im_end|>` 151645 to
257. Phi-4-mini maps the byte tokens (GPT-2 byte alphabet order) to their byte
values, `<|endoftext|>` 199999 to 256, `<|end|>` 200020 to 257, IDs 256..199,998
to +2, 200,000..200,019 to +1, and leaves 200,021 and above unchanged.
Gemma 3 uses normalizer 0, pre-tokenizer 3 and chat template 3; its converter
puts byte tokens at 0..255, `<bos>` at 256 and `<end_of_turn>` at 257. It requires
`<start_of_turn>` and reuses available `<unused:...>` special-token slots for
`<tool_call>` / `</tool_call>` when they are absent from the source tokenizer.
Llama 3 uses normalizer 0 and pre-tokenizer 4; its converter puts the byte tokens
at 0..255, `<|begin_of_text|>` at 256 and the end-of-turn token at 257, every other
ID following in ascending source order (Dolphin 3.0 Llama 3.2 1B: source IDs
256..127,999 move by +2 minus the bytes they displace). With `<|im_start|>` and
`<|im_end|>` (Dolphin 3.0) the template is 4 and ID 257 is `<|im_end|>`; without
them (Meta's base and Instruct releases) the template is 0 and ID 257 is
`<|eot_id|>`, or `<|end_of_text|>` when that is absent too. No base piece is
flagged: with `ignore_merges` a piece no merge produces is still reachable.

**Encoding.** Text is normalized (NFC, Unicode 15.0.0 tables) only when the
header says so; Phi-4-mini is not normalized. It is split into pre-tokens and each
pre-token's UTF-8 bytes are merged by increasing pair rank, leftmost first. The
Qwen2, Llama 3 and o200k pre-tokenizers are hand-written matchers equal to the published
regexes run by Oniguruma (the engine of Hugging Face tokenizers), with no runtime
regex or Unicode library. Gemma implements SentencePiece's
`Split(' ', MergedWithNext)` behavior in a separate boundary matcher.

- Qwen2: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
- Llama 3: the Qwen2 regex with `\p{N}{1,3}` in place of `\p{N}` (runs of up to three digits).
  Pre-tokenizer 4 also means `ignore_merges`: a pre-token whose bytes are exactly an
  unflagged piece is that token, and only other pre-tokens go through the merges, as
  in tiktoken and the Llama 3 `tokenizer.json`.
- o200k (GPT-4o, Phi-4-mini): `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`

`\s` is Oniguruma's: U+0009..000D, 0020, 0085, 00A0, 1680, 2000..200A, 2028,
2029, 202F, 205F and 3000 (JavaScript's differs: it has U+FEFF and lacks U+0085).
`(?i)` folds U+017F (ſ) to `s`. Character classes come from Unicode 16.0.0, the
version of Oniguruma 6.9.10 used by tokenizers 0.21 and 0.22. The tables are
generated by `bun scripts/gen-unicode.ts` from pinned, SHA-256 verified UCD
files: `src/unicode_class.h` (class bits, UnicodeData 16.0.0) and
`src/unicode_nfc.h` (NFC, UnicodeData and CompositionExclusions 15.0.0, unchanged
since the tables were first generated, matching the older NFC that Hugging Face
uses for Qwen).

**Chat templates.** `--chat` and the interactive chat insert role and end
markers as tokens and encode the user's text as plain text: marker strings typed
by the user are never conversation control (Hugging Face would match them).

- ChatML (Qwen2.5): `<|im_start|>{role}\n{text}<|im_end|>\n` per message, then
  `<|im_start|>assistant\n`. A reply is closed with `<|im_end|>\n`. In system
  text, `<tool_call>` and `</tool_call>` become their tokens and enable the
  chat's tools. Required: a flagged `<|im_start|>` (version 2) and ID 257 =
  `<|im_end|>`.
- Phi (Phi-4-mini): `<|system|>S<|end|><|user|>U<|end|><|assistant|>`, with no
  newlines. Every Phi marker has `rstrip`, so whitespace right after a role
  marker is dropped, as in the published template. A reply is closed with
  `<|end|>`. In system text, `<|tool|>`, `<|/tool|>`, `<|tool_call|>` and
  `<|/tool_call|>` become their tokens (with `rstrip`); the plain-text aliases
  `<tool_call>` and `</tool_call>` map to the call marker pair. The runtime
  offers its built-in tools for this template. Required: flagged `<|system|>`,
  `<|user|>` and `<|assistant|>`, ID 256 = `<|endoftext|>` and ID 257 = `<|end|>`.
- Gemma 3: `<start_of_turn>{role}\n{text}<end_of_turn>\n` per message, with
  `assistant` mapped to `model` and the initial system message mapped to `user`
  after BOS. The open reply is `<start_of_turn>model\n`; a reply ends with
  `<end_of_turn>`. In system text, `<tool_call>` and `</tool_call>` become their
  tokens and enable the chat's tools. Required: flagged `<start_of_turn>`, ID
  256 = `<bos>` and ID 257 = `<end_of_turn>`.
- ChatML-BOS (Dolphin 3.0 on Llama 3): ChatML as above, with BOS (256) before
  the system message that opens a conversation; plain prompts also start with
  BOS, as the Llama 3 tokenizer's post-processor adds it. The Llama 3 vocabulary
  has no `<tool_call>` token, so the chat runs without tools.
- None: plain text only; `--chat` is refused.

Generation stops at EOS (257). An empty plain prompt encodes as BOS.

## Version 3: single-file models

Version 3 adds dtype 3 (**blob**): opaque bytes with `rows = cols = 1`, `alias = 0` and a name starting with `__`.
`smolmoe-bundle` stores the config text as `__config__` and the tokenizer file as `__tokenizer__`; the runtime uses
them when `--config` / `--tokenizer` are omitted. Blobs are excluded from the tensor count check, and v1/v2 readers
reject v3 files at the header. The tied-embedding alias is located by name, so blobs may sort before `embed`.

## Self-contained executables

```text
[runtime executable][zero padding to a 64 KiB boundary][SMOEQ8 v3 container][trailer]
trailer, 32 bytes: "SMOEBDL1" | u64 container offset | u64 container length | "SMOEBDL1"
```

Without `--model`, the runtime maps its own executable (`/proc/self/exe`, `_NSGetExecutablePath`,
`GetModuleFileNameW`), validates the trailer (both magics, 64 KiB-aligned offset, offset + length + 32 = file size)
and parses the container in place. A trailer with both magics but inconsistent fields is a hard error, not a
fallback. Container offsets are relative to the container start, so the same container works as a file or appended.
Windows refuses to run executables above 4 GiB, so larger models must stay in a separate `.smoe` file. Code signing
must happen after bundling (Authenticode covers the whole file).
