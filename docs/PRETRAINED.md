# Real trained model: TinyStories 15M

The package includes Karpathy's trained `stories15M.bin` converted to Q8, plus
its compatible score-BPE tokenizer. No GPU, external inference service, BLAS,
PyTorch, NumPy or other ML framework is used by the executable. Inference is C99
and single-threaded, with an AVX2 dispatch path on the tested host.

## Execute offline

```sh
make
make pretrained-demo
```

Equivalent command:

```sh
./smolmoe --model examples/tinystories15m/model.smoe \
  --config examples/tinystories15m/config.ini \
  --tokenizer examples/tinystories15m/tokenizer.spb \
  --prompt 'Once upon a time, there was a little robot' \
  --tokens 80 --context 256 --memory-mib 16
```

Prompt:

> Once upon a time, there was a little robot

Actual generated continuation (cut off at the requested 80 tokens):

```text
. The robot was very sad because it was all alone. It wanted to find a friend to play with.
One day, the robot saw a big ball. The robot thought, "Maybe I can find a friend to play with." So, the robot went to the ball and started to play. The robot was very happy.
But then, the robot saw a little girl. The girl
```

A second prompt, `The little bird found a`, generated:

```text
 nest in the tree. It was a nice and cozy nest. The bird was so happy to have a home.
The bird flew around the tree and sang a happy song. Suddenly, a big wind came and ble
```

The text is produced by the trained weights at inference, not selected from
stored answer strings or echoed from a fixed fixture. The runtime outputs only
the continuation; it does not echo the prompt. Token limits may end mid-sentence.

## Measured execution

Linux x86-64, AMD EPYC 9V45, GCC 13.3.0. Statistics come from an untraced,
80-new-token greedy run; prompt length was 11 tokens including BOS.
A warm filesystem cache is possible. These results describe this host and this
small English story model, not a Kimi model or a cold-disk performance guarantee.

| Measurement | Observed |
|---|---:|
| Backend | avx2 |
| Application allocation budget | 16 MiB |
| Tracked peak allocation | 15.55 MiB |
| Process RSS high-water mark | 13.75 MiB |
| Model file | 27,475,936 bytes |
| First generated token, including init/prefill | 58.38 ms |
| Init + prefill + decode | 450.49 ms |
| Decode forward passes / decode time | 201.45 per second |

The model file is larger than the 8 MiB budget also exercised below. Budget
accounting excludes allocator implementation overhead, libc, stack and OS page
cache; RSS is reported separately. `read_bytes` counts bytes returned by `pread`,
not physical storage traffic. The tested host has more than 8 GB of physical RAM;
this is evidence of small process allocations, not a test on a physical 8 GB PC.

## Exact memory-budget regression

`make pretrained-test` compares full hexadecimal logits, token IDs and generated
bytes for the first 24 generated tokens across 8, 16, 32 and 64 MiB budgets,
with automatic AVX2 dispatch and forced scalar execution: **8 runs matched exactly**.
Each mode's complete logits SHA-256 was:

`00b83cf374d18848c6593541aad272cb7ee6b87495781019b9ff6a28cc67aa4c`

| Budget MiB | Tracked peak MiB | RSS peak MiB | pread bytes |
|---|---:|---:|---:|
| 8 | 7.55 | 11.25 | 583,742,736 |
| 16 | 15.55 | 13.38 | 541,024,272 |
| 32 | 31.55 | 19.12 | 137,412,252 |
| 64 | 62.05 | 20.25 | 31,648,824 |

These traced runs validate equality and cache behavior; their trace formatting
costs are excluded from the separate performance measurement above. Filesystem
cache can satisfy repeated reads, so the read figures are not NVMe throughput.

## Comparison with the original model

The original F32 checkpoint was executed with the upstream `llama2.c` reference.
On the 80-token robot prompt above, the greedy continuation matched this Q8
runtime's continuation byte for byte. This observation is **specific to this
prompt and token limit**. Q8 is lossy and exact source-F32 logits or all-prompts
text identity are not promised. Exact budget/backend equality applies to the
same Q8 weights, not to a comparison between quantized and original weights.

Reference source: https://github.com/karpathy/llama2.c/blob/master/run.c

Downloaded source SHA-256:
`9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd`

The reference needed `-include stdint.h` in this GCC environment:

```sh
cc -O3 -std=c99 -D_POSIX_C_SOURCE=200809L -include stdint.h \
  -ffp-contract=off run.c -o llama2-reference -lm
python3 tools/validate_pretrained.py examples/tinystories15m \
  --reference /absolute/path/llama2-reference \
  --original /absolute/path/stories15M.bin \
  --original-tokenizer /absolute/path/tokenizer.bin
```

The model, tokenizer and reference binary hashes, complete equality checks and
raw statistics are in `pretrained-validation.json`. The optional reference
check needs the original F32 file; normal execution and regression need only
this package's bundled Q8 model.

## Reproduce conversion

```sh
python3 tools/import_llama2c.py downloads/stories15M.bin downloads/tokenizer.bin \
  examples/tinystories15m --fetch-tiny15m
```

This uses Python standard-library I/O and math, checks download SHA-256 values,
then streams source rows through the Q8 packer. No training framework or original
PyTorch pickle is loaded. See `provenance.json` and `THIRD_PARTY.md` for attribution.
The dense model maps to one expert with router weight 1; no trained expert
routing or quality claim is made for this checkpoint.

## Limits

The model is trained for simple English story continuation. It is not a reliable
question-answering assistant, Portuguese instruction model, coding model, or
replacement for Kimi K3. It has 256-token context in this checkpoint. K3's KDA,
MLA, Attention Residuals, latent MoE, SiTU, MXFP4 and tokenizer remain unsupported.
What is achieved here is real prompt-conditioned CPU generation with trained
weights, bounded application allocations and a matching tokenizer.
