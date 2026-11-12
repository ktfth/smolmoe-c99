# Modelos

Os pesos não ficam no git (somam 560 MB; o GitHub recusa arquivos acima de 100 MB). Eles ficam no Release
`models-v1`. Cada modelo mora em `models/<nome>/`, com quatro arquivos: os
pesos (`model.smoe`), o tokenizer, a `config.ini` e a `provenance.json`. Os dois últimos são versionados.

| Diretório | Modelo | Uso | Tamanho |
|---|---|---|---|
| `models/qwen2.5-0.5b-instruct` | Qwen2.5-0.5B-Instruct, Q8 | Chat em português (`--chat`) | 531 MiB |
| `models/tinystories-15m` | TinyStories 15M (llama2.c), Q8 | Histórias curtas em inglês; testes rápidos | 26 MiB |
| `models/gemma-3-1b-it` | Gemma 3 1B Instruct (Google), Q8 | Chat equilibrado; por conversão (`make gemma-weights`) | ~1,0 GiB |
| `models/gemma-3-270m-it` | Gemma 3 270M Instruct (Google), Q8 | Conversão disponível; pesos ainda sem validação ponta a ponta | ~280 MiB |
| `models/phi-4-mini-instruct` | Phi-4-mini-instruct (Microsoft), Q8 | Chat de mais qualidade, mais lento; só por conversão (opção 2) | 4,0 GiB |
| `models/dolphin3.0-llama3.2-1b` | Dolphin 3.0 Llama 3.2 1B (Cognitive Computations sobre Meta Llama 3.2), Q8 | Chat em inglês (português razoável), arquitetura Llama 3; por conversão (`make dolphin-weights`) | 1,3 GiB (Q5: 810 MiB) |

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

O conversor é `scripts/convert.ts` (TypeScript/Bun) e roda em qualquer sistema, inclusive no Windows nativo. Os
comandos `fetch-*` usam revisões fixas e conferem SHA-256 dos arquivos declarados; para Gemma, isso cobre o checkpoint
1B usado pelo atalho `make gemma`. Conversões manuais gravam os hashes de entrada em `provenance.json`.

```sh
bun scripts/convert.ts fetch-qwen2 downloads/qwen2.5-0.5b-instruct-src
bun scripts/convert.ts qwen2 downloads/qwen2.5-0.5b-instruct-src models/qwen2.5-0.5b-instruct

bun scripts/convert.ts fetch-gemma3 downloads/gemma-3-1b-it          # requer HF_TOKEN
bun scripts/convert.ts gemma3 downloads/gemma-3-1b-it models/gemma-3-1b-it

bun scripts/convert.ts fetch-phi3 downloads/phi-4-mini-instruct      # 7,7 GB
bun scripts/convert.ts phi3 downloads/phi-4-mini-instruct models/phi-4-mini-instruct

bun scripts/convert.ts fetch-llama downloads/dolphin3.0-llama3.2-1b  # 2,5 GB (ou: make dolphin-weights)
bun scripts/convert.ts llama downloads/dolphin3.0-llama3.2-1b models/dolphin3.0-llama3.2-1b
```

- **Llama 3.x:** `llama` converte qualquer checkpoint `LlamaForCausalLM` com o tokenizer do Llama 3 (Dolphin 3.0
  Llama 3.2 1B e 3B, Dolphin 3.0 Llama 3.1 8B, Llama 3.2 e 3.1 da Meta). Só o 1B tem `fetch-llama` com hashes fixos;
  para os outros, baixe o diretório do Hugging Face e rode `llama` nele: os hashes vão para `provenance.json`. A
  cabeça de saída separada do 8B é gravada à parte. O Q8 do 8B tem cerca de 8,5 GB e não cabe em 8 GB de RAM; use
  `smolmoe-quantize` para o Q5 (cerca de 5,5 GB). O cache KV é F32: no 8B são 128 KB por posição de contexto
  (`--context 4096` = 512 MB). Os modelos Instruct da Meta saem sem template de chat (só texto puro), porque o
  template nativo do Llama 3 ainda não foi implementado; os Dolphin usam ChatML e têm chat.
- **Dolphin 3.0 Llama 3.2 1B, medido** (i7 com AVX-512 VNNI, 4 threads, logits contra uma referência float32
  independente com a semântica do Hugging Face num trecho de 810 tokens): KL médio 0,0045 e top-1 igual em 96% das
  posições; perplexidade 21,79 (referência), 22,02 em Q8 (+1,1%) e 23,61 em Q5 (+8,4%). Tokenizer igual ao do
  Hugging Face em 411 de 412 textos; o caso diferente é `<|im_start|>` digitado pelo usuário, que o runtime trata
  como texto de propósito. Decode de cerca de 14 a 16 tok/s.

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
- **Ferramentas:** o runtime reconhece o formato de chamada do tokenizer Phi e oferece as ferramentas nativas do chat (`data_atual`, `dias_ate`, `calcular`, `converter`, `contar`).

## Variante Q5 (menos memória)

```sh
build/smolmoe-quantize --in models/qwen2.5-0.5b-instruct/model.smoe --out models/qwen2.5-0.5b-instruct/model-q5.smoe
```

Use `--model .../model-q5.smoe` com a mesma `config.ini` e o mesmo tokenizer. Custos e ganhos medidos estão em
[DECISIONS.md](DECISIONS.md) (iteração 2).

## Licenças

- Qwen2.5-0.5B-Instruct: Apache-2.0 (`models/qwen2.5-0.5b-instruct/LICENSE-Qwen.txt`).
- Gemma 3 1B Instruct: [Gemma Terms of Use](https://ai.google.dev/gemma/terms/); o acesso ao checkpoint no Hugging
  Face exige aceitar os termos. Antes de redistribuir pesos ou executáveis empacotados com Gemma, siga as condições
  de distribuição dos termos oficiais.
- Phi-4-mini-instruct: MIT (`models/phi-4-mini-instruct/LICENSE-Phi.txt`).
- Dolphin 3.0 Llama 3.2 1B: [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/) (derivado do
  Meta Llama 3.2). Redistribuir pesos ou executáveis empacotados exige seguir a licença: incluir uma cópia dela,
  manter o aviso "Built with Llama" e respeitar a política de uso aceitável da Meta.
- TinyStories 15M e o tokenizer do llama2.c: MIT (`licenses/llama2.c-MIT.txt`).

Ver também [THIRD_PARTY.md](../THIRD_PARTY.md).
