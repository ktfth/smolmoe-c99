# Modelos

Os pesos não ficam no git (somam 560 MB; o GitHub recusa arquivos acima de 100 MB). Eles ficam no Release
`models-v1`. Cada modelo mora em `models/<nome>/`, com quatro arquivos: os
pesos (`model.smoe`), o tokenizer, a `config.ini` e a `provenance.json`. Os dois últimos são versionados.

| Diretório | Modelo | Uso | Tamanho |
|---|---|---|---|
| `models/qwen2.5-0.5b-instruct` | Qwen2.5-0.5B-Instruct, Q8 | Chat em português (`--chat`) | 531 MiB |
| `models/tinystories-15m` | TinyStories 15M (llama2.c), Q8 | Histórias curtas em inglês; testes rápidos | 26 MiB |
| `models/phi-4-mini-instruct` | Phi-4-mini-instruct (Microsoft), Q8 | Chat de mais qualidade, mais lento; só por conversão (opção 2) | 4,0 GiB |

## Opção 1: baixar os pesos já convertidos

```sh
bun scripts/fetch-models.ts                 # todos os modelos (ou: make models)
bun scripts/fetch-models.ts tinystories-15m # só um
```

O script baixa os arquivos listados em `models/manifest.json` do Release `models-v1` do repositório e confere tamanho
e SHA-256 antes de colocar cada um no lugar. Um download corrompido ou interrompido nunca fica em `models/`. Arquivos
já presentes e verificados são pulados. O repositório é privado, então o script precisa de credencial: usa `GH_TOKEN`
ou `GITHUB_TOKEN` e, sem eles, o login do `gh` (`gh auth login`). Em um fork, aponte para o Release certo com
`SMOLMOE_MODELS_REPO=dono/nome`.

Para copiar à mão (de outra máquina, por exemplo), coloque `model.smoe` e o tokenizer no diretório do modelo e
confira os hashes:

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

O conversor é `scripts/convert.ts` (TypeScript/Bun) e roda em qualquer sistema, inclusive no Windows nativo. Cada
modelo tem a revisão do Hugging Face e o SHA-256 dos originais fixados no código: um original diferente é recusado.

```sh
bun scripts/convert.ts fetch-qwen2 downloads/qwen2.5-0.5b-instruct-src
bun scripts/convert.ts qwen2 downloads/qwen2.5-0.5b-instruct-src models/qwen2.5-0.5b-instruct

bun scripts/convert.ts fetch-phi3 downloads/phi-4-mini-instruct      # 7,7 GB
bun scripts/convert.ts phi3 downloads/phi-4-mini-instruct models/phi-4-mini-instruct
```

- **Qwen:** a saída é idêntica byte a byte aos arquivos do Release (os hashes da tabela acima). Leva cerca de 9 s.
- **Phi-4-mini:** leva cerca de 40 s com 8 threads e gera 4,3 GB. O Release não o hospeda, porque o GitHub limita
  cada arquivo a 2 GB. A saída é determinística: `model.smoe` tem SHA-256
  `83c24b62192add141c230cf30d92d5c0a14d0f36b4642b3249801277419d3468` e `tokenizer.qbp`,
  `773973f89c1882343167260c0ef227da4237ecf448d19b34ae3821843780d36f`.
- `--container v3` gera um arquivo único, com config e tokenizer embutidos. `bun scripts/convert.ts check MODELO`
  confere um modelo com as mesmas regras do carregador em C.

A conversão é com perda (Q8 com escala F32 por bloco de 32). Ela permuta as linhas de Q/K para o RoPE de pares
adjacentes e reordena os IDs dos tokens para a convenção do runtime. Os detalhes de cada modelo ficam na sua
`provenance.json`, e o formato está em [FORMAT.md](FORMAT.md).

O TinyStories ainda usa o conversor antigo em Python (`tools/import_llama2c.py`); a migração está no roadmap.

## Phi-4-mini-instruct (Microsoft, 3,8B)

```sh
build/smolmoe --model models/phi-4-mini-instruct/model.smoe --config models/phi-4-mini-instruct/config.ini \
  --tokenizer models/phi-4-mini-instruct/tokenizer.qbp
```

- **Contexto:** até 4096 tokens. É o limite em que o modelo não precisa da escala de contexto longo.
- **Memória:** cerca de 4,4 GB, quase todos pesos mapeados.
- **Velocidade neste notebook:** cerca de 4 tokens/s escrevendo e 12 tokens/s lendo.
- **Ferramentas:** o chat roda sem elas, porque o formato de chamada de função do Phi ainda não foi validado.

## Variante Q5 (menos memória)

```sh
build/smolmoe-quantize --in models/qwen2.5-0.5b-instruct/model.smoe --out models/qwen2.5-0.5b-instruct/model-q5.smoe
```

Use `--model .../model-q5.smoe` com a mesma `config.ini` e o mesmo tokenizer. Custos e ganhos medidos estão em
[DECISIONS.md](DECISIONS.md) (iteração 2).

## Licenças

- Qwen2.5-0.5B-Instruct: Apache-2.0 (`models/qwen2.5-0.5b-instruct/LICENSE-Qwen.txt`).
- Phi-4-mini-instruct: MIT (`models/phi-4-mini-instruct/LICENSE-Phi.txt`).
- TinyStories 15M e o tokenizer do llama2.c: MIT (`licenses/llama2.c-MIT.txt`).

Ver também [THIRD_PARTY.md](../THIRD_PARTY.md).
