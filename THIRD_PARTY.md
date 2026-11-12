# Bundled trained weights and tokenizer

The runtime and conversion tools are provided under the root MIT license.

`models/tinystories-15m/model.smoe` is a lossy Q8 conversion of
`karpathy/tinyllamas/stories15M.bin`, a model trained by Andrej Karpathy on
TinyStories. The source repository declares MIT:
https://huggingface.co/karpathy/tinyllamas

Source revision: `0bd21da7698eaf29a0d7de3992de8a46ef624add`.
Source F32 SHA-256:
`cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a`.

`models/tinystories-15m/tokenizer.spb` is a reordered-container conversion of
the tokenizer distributed by https://github.com/karpathy/llama2.c under that
repository's MIT license. Original tokenizer SHA-256:
`50a52ef822ee9e83de5ce9d0be0a025a773d019437f58b5ff9dcafb063ece361`.
The upstream copyright and permission notice is included in
`licenses/llama2.c-MIT.txt`. Tokenizer compatibility work used the upstream
implementation as a reference; it is not a full SentencePiece implementation.

No original upstream inference source is required or bundled. The independently
compiled upstream reference used for validation remains an optional development
tool. The model is trained for simple English story continuation, not general
chat, reliable factual answers, coding, Portuguese instruction following, or Kimi
capability equivalence. Its outputs can be incorrect or unsuitable.


## Qwen Portuguese model and Unicode data

`models/qwen2.5-0.5b-instruct/model.smoe` is a lossy Q8 conversion of
Qwen/Qwen2.5-0.5B-Instruct, revision eaa56b503cc0a8a4d15de1dd8bd2a7e95a716be2.
Original BF16 SHA-256:
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The Qwen model/tokenizer are Apache-2.0; the upstream license is retained in
`models/qwen2.5-0.5b-instruct/LICENSE-Qwen.txt`. Conversion changes the weight
representation, token ID ordering and rotary row order; the provenance JSON
records the transformation. The upstream inference code is not included.

## Phi-4-mini model

`models/phi-4-mini-instruct/` is produced locally by `scripts/convert.ts phi3`
(it is not distributed with this repository): a lossy Q8 conversion of
microsoft/Phi-4-mini-instruct, revision cfbefacb99257ffa30c83adab238a50856ac3083,
BF16 shard SHA-256 `bc703090b63eda16f639fa4de7ac54635c23105ab1da2f6ec4d3403151d38ee6`
and `7ff79b9d2d31076bac2663393451f6530f4fc8ca49b09002116c92c373dba983`. The
model and tokenizer are MIT; the upstream license is copied to
`models/phi-4-mini-instruct/LICENSE-Phi.txt`. Conversion changes the weight
representation, token ID ordering and rotary row order (provenance JSON). The
validation reference in `scripts/ref/` is an independent reimplementation; no
upstream inference code is included.

## Dolphin 3.0 Llama 3.2 1B model

`models/dolphin3.0-llama3.2-1b/` is produced locally by `scripts/convert.ts llama`
(it is not distributed with this repository): a lossy Q8 conversion of
dphn/Dolphin3.0-Llama3.2-1B, revision e753b6ebd7adf87036eb6a3e6de68acca5850e2f,
BF16 SHA-256 `fa82d5c8c1cf12762a318daa6a6fd90ba0322c989b66cd54e565b685ab3294b0`.
The model is a fine-tune of Meta Llama 3.2 1B under the Llama 3.2 Community
License (https://www.llama.com/llama3_2/license/); redistributing the weights or
a bundled executable requires following that license. Conversion changes the
weight representation, token ID ordering and rotary row order (provenance JSON).
No upstream inference code is included.

## Dear ImGui (desktop window)

`third_party/imgui/` holds Dear ImGui v1.91.9b by Omar Cornut and contributors,
under the MIT license kept in `third_party/imgui/LICENSE.txt`. Source:
https://github.com/ocornut/imgui/archive/refs/tags/v1.91.9b.tar.gz, SHA-256
`8e1bbc76c71d74fef2fb85db7e7ca8eba13d6a86623c54992b60162db554ffdb`. Only the
core, `misc/cpp/imgui_stdlib` and the Win32, Direct3D 11, GLFW and OpenGL 3
backends are copied, unmodified. It is compiled only into `smolmoe-gui`; the
runtime and the terminal chat do not use it. On Linux and macOS the window links
the system's GLFW (zlib/libpng license), which is not bundled.

## Unicode data

`src/unicode_class.h` (UnicodeData.txt 16.0.0) and `src/unicode_nfc.h`
(UnicodeData.txt and CompositionExclusions.txt 15.0.0) are generated from the
Unicode Character Database by `scripts/gen-unicode.ts`, which pins each file's
SHA-256. See `licenses/Unicode-LICENSE.txt`. Tokenizer validation used Hugging
Face tokenizers 0.21.4 (Qwen) and js-tiktoken 1.0.21 `o200k_base` (Phi-4-mini)
as development-only references; neither is bundled or required at runtime.
