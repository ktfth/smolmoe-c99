# smolmoe_v1 architecture and binary contract

This specification describes the implemented model, not Kimi K3.
`smolmoe_qwen2` extends the base architecture with F32 bias vectors
`l.N.q_bias` (1 × D), `l.N.k_bias` and `l.N.v_bias` (1 × KD), added before RoPE
and KV storage. The importer permutes split-half Q/K rows and biases into this
runtime's adjacent-pair RoPE order. Only the validated Qwen2 layout is supported.
All matrix shapes below are `(output rows, input columns)` and row-major.
Let D = dim, H = hidden_dim, V = vocab_size, E = experts, S = shared_experts,
A = heads, K = kv_heads, L = layers, and KD = K × D/A.

## Tensor names

| Name | Shape | Encoding |
|---|---|---|
| `embed` | V × D | Q8 |
| `final_norm` | 1 × D | F32 |
| `lm_head` | V × D | Q8, separate from embedding |
| `l.N.attn_norm`, `l.N.ffn_norm` | 1 × D | F32 |
| `l.N.q`, `l.N.o` | D × D | Q8 |
| `l.N.k`, `l.N.v` | KD × D | Q8 |
| `l.N.router` | E × D | Q8 |
| `l.N.e.I.gate`, `l.N.e.I.up` | H × D | Q8 |
| `l.N.e.I.down` | D × H | Q8 |
| `l.N.s.I.gate`, `l.N.s.I.up` | H × D | Q8 |
| `l.N.s.I.down` | D × H | Q8 |

N is the zero-based layer number, I the zero-based expert number.
Every layer is a routed MoE layer; there are no special dense-first layers.
There are no biases, tied embeddings, q/k norms, output gates, latent projections,
router bias/correction/grouping, or shared-expert scalar gate.

## Forward pass

All runtime activations and weight scales are F32. For each token:

1. `x = embed[token]`.
2. For each layer, `n = RMSNorm(x, attn_norm, rms_eps)`.
3. `q = Wq n`, `k = Wk n`, `v = Wv n`. Apply adjacent-pair RoPE to q and k.
   Only the first R = `rotary_dim` coordinates of each head rotate (default R = head_dim);
   the rest pass through. For pair j = 0,2,…,R−2, `angle = position / rope_theta^(j/R)`,
   and the pair is rotated by `f·cos(angle)` and `f·sin(angle)` with f = `rope_attention_factor`
   (default 1; Phi-3 "longrope" uses it). The tables are computed in double and rounded to F32.
   This adjacent-pair convention matters when exporting trained checkpoints.
4. Append k and v to the layer's unquantized KV cache. Query head h maps to
   KV head `floor(h / (heads / kv_heads))`.
5. `att = softmax(q K^T / sqrt(head_dim)) V`, causal through current position.
   `x += Wo att`.
6. `n = RMSNorm(x, ffn_norm, rms_eps)`; compute all router logits `r = Wr n`.
7. Select highest top_k logits, breaking exact ties by lower expert index.
   `p_i = sigmoid(r_i)`; `w_i = p_i / sum_selected(p)`.
8. For each selected expert in descending logit order,
   `x += w_i * Wdown_i (SiLU(Wgate_i n) * Wup_i n)`.
   Then add each shared expert's same MLP output in ascending expert index,
   with coefficient 1. All experts use the same n from step 6.
9. After all layers, `logits = lm_head RMSNorm(x, final_norm, rms_eps)`.

RMSNorm is `x_i / sqrt(mean(x*x) + eps) * weight_i`.
SiLU is `z/(1+exp(-z))`. There are no residual scaling factors.
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
weights are not counted.

Two keys are optional: `rotary_dim` (even, 2 to D/A; default D/A) and
`rope_attention_factor` (in (0, 1000]; default 1). Omitting them gives results
bit-identical to writing their defaults.

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
model is dense, not a learned sparse MoE model. The importer supports the legacy
7-int llama2.c header and F32 payload layout only; newer format versions,
MXFP4, Hugging Face checkpoints and Kimi K3 remain unsupported.


### Qwen byte-BPE format

The 24-byte header is `SMQBP1`, NUL, byte 1, then little-endian uint32 values:
vocabulary size, merge count, total piece bytes, and runtime im_start token ID.
Each piece record contains uint32 length, uint32 special/unused flag (0 or 1),
and exactly that many raw bytes. IDs 0..255 must decode to their corresponding
single byte. Runtime BOS/endoftext is 256 and EOS/im_end is 257; other source
IDs are permuted consistently in the tokenizer, embedding and tied head.
Each merge record contains uint32 left, right, output; record order is rank.
The loader validates concatenation, IDs, dimensions, duplicate pairs and lengths.

Qwen text is normalized to Unicode 15 NFC, split using the Qwen2 tokenizer's
pre-tokenization rules and merged by increasing pair rank. The built-in tables
and UTF-8 parser require no runtime Unicode/regex library. `--chat` inserts
role/end markers separately and encodes user content as plain text; it does not
interpret special marker strings inside the user's text as conversation control.
Special/unused outputs are masked except EOS. The supported chat template is
single-turn system/user/assistant with a generation prefix, not a tools protocol.

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
