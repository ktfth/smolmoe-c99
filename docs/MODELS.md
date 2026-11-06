# Modelos

Os pesos não ficam no repositório (somam 560 MB). Cada modelo mora em `models/<nome>/`, com quatro arquivos: os
pesos (`model.smoe`), o tokenizer, a `config.ini` e a `provenance.json`. Os dois últimos são versionados.

| Diretório | Modelo | Uso | Tamanho |
|---|---|---|---|
| `models/qwen2.5-0.5b-instruct` | Qwen2.5-0.5B-Instruct, Q8 | Chat em português (`--chat`) | 531 MiB |
| `models/tinystories-15m` | TinyStories 15M (llama2.c), Q8 | Histórias curtas em inglês; testes rápidos | 26 MiB |

## Opção 1: copiar os pesos já convertidos

Copie `model.smoe` e o tokenizer para o diretório do modelo e confira os hashes:

| Arquivo | SHA-256 |
|---|---|
| `qwen2.5-0.5b-instruct/model.smoe` | `9f6de8916fcfe841245a0ae7593ba66b5d2cac98c4f8d13fa4ad696889055260` |
| `qwen2.5-0.5b-instruct/tokenizer.qbp` | `12512c1c37f89ca0e9341a19ee8cba3c23b81c33678fcdff9b8aa0720bc4edac` |
| `tinystories-15m/model.smoe` | `df99487f97b88103ca7e44703870ee5909fef71b663a3c2af25390d1d065a9dc` |
| `tinystories-15m/tokenizer.spb` | `9f439ef87c7bef9c873a70a89cb8f4386f4cec246b69d488b8f45cadca8988f2` |

```sh
sha256sum models/*/model.smoe models/*/tokenizer.*                      # Linux / Git Bash
Get-FileHash models\*\model.smoe -Algorithm SHA256                      # PowerShell
```

## Opção 2: converter a partir dos originais

Os conversores ainda estão em Python e usam só a biblioteca padrão, sem PyTorch e sem NumPy. A conversão de linhas
usa `tools/q8pack.c`, que é POSIX: rode no Linux, no macOS ou no WSL. A migração para TypeScript está no roadmap.

```sh
make tools                                   # compila tools/q8pack
python3 tools/import_qwen2.py downloads/qwen2.5-0.5b models/qwen2.5-0.5b-instruct --fetch
python3 tools/import_llama2c.py downloads/stories15M.bin downloads/tokenizer.bin models/tinystories-15m --fetch-tiny15m
```

`--fetch` baixa a revisão fixada no Hugging Face e confere o SHA-256 do original antes de converter. A conversão é com
perda (Q8 com escala F32 por bloco de 32) e permuta as linhas de Q/K para o RoPE de pares adjacentes. Os detalhes de
cada modelo ficam na sua `provenance.json`, e o formato está em [FORMAT.md](FORMAT.md).

## Licenças

- Qwen2.5-0.5B-Instruct: Apache-2.0 (`models/qwen2.5-0.5b-instruct/LICENSE-Qwen.txt`).
- TinyStories 15M e o tokenizer do llama2.c: MIT (`licenses/llama2.c-MIT.txt`).

Ver também [THIRD_PARTY.md](../THIRD_PARTY.md).
