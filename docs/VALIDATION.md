# Validation record — Portuguese release

Updated 2026-11-06, America/Sao_Paulo. Linux x86-64, AMD EPYC 9V45,
GCC 13.3.0, Python 3.12.14. Only this host and compiler were executed.

- `make -B test`: **35 regression tests passed**, plus exact scalar/AVX2 dot
  kernel checks for input lengths 1..32.
- `ASAN_OPTIONS=detect_leaks=0 make sanitize`: **35 tests and kernel checks
  passed** with AddressSanitizer and UBSan. Leak detection was disabled because
  this runner blocks its `/proc` thread inspection. Leaks were not established
  absent by a leak detector.
- Prior synthetic/reference checks, legacy TinyStories import checks, large
  offsets, insufficient budgets and malformed-file rejection remain covered.
- New checks cover Qwen2 QKV biases, tied-head v2 aliasing, BF16 row conversion,
  rotary row permutation, NFC, invalid UTF-8, byte-BPE and ChatML tokenization.
- Qwen tokenizer matched Hugging Face tokenizers 0.21.4 in **11 recorded cases**,
  including Portuguese accents, decomposed Unicode, multilingual text and the
  complete conversation template. That library was development-only and is not
  required by the program or its offline regression suite.
- The real Qwen2.5-0.5B-Instruct Q8 model generated Portuguese outputs. The full
  response `12` and FNV-1a64 checksum of every raw logit matched under 128,256,512
  MiB allocation budgets, with dispatched AVX2 and forced scalar execution.
  This is a regression hash comparison, not a cryptographic all-input proof.
- A final optimized `make portugues-demo` run completed. Its full stdout and
  statistics are included under `examples/qwen-portugues/demo-atual*`.

The current release includes the trained Qwen weights/tokenizer and runs offline.
See `LEIA-ME-PORTUGUES.md` for commands and observed quality limitations;
`resultados-portugues.json`, `qwen-memory-validation.json`, and
`qwen-tokenizer-validation.json` for raw evidence. The small model sometimes
mixes languages or misses requested formatting; the failing examples were kept.

Older TinyStories/synthetic logs and measurements are historical results, not
current Qwen throughput figures. The original 21/27-test release counts are
superseded by this 35-test record. The bundled binary is optimized, not sanitized.

No full Kimi K3 model, physical 8 GB machine, trained large sparse MoE, cold-NVMe
benchmark, alternate compiler or non-x86 host was tested. Application budget,
process RSS and filesystem cache are distinct. These results establish local
Portuguese generation with the supported small Qwen model, not Kimi equivalence.
