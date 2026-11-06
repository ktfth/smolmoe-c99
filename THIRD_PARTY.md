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

`src/unicode_data.h` is generated from Python's Unicode 15.0.0 database using
`tools/generate_unicode.py`. See `licenses/Unicode-LICENSE.txt`. Tokenizer
validation used Hugging Face tokenizers 0.21.4 as a development-only reference;
its code/wheel is not bundled or required at runtime.
