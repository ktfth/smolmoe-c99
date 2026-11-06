# smolmoe

Runtime de inferência de LLMs em CPU, escrito em C99, sem frameworks. É um executável pequeno que roda modelos
quantizados em Q8 (formato `SMOEQ8`) no Windows, no Linux e no macOS. O caso principal é o
**Qwen2.5-0.5B-Instruct gerando português do Brasil localmente**, sem GPU e sem serviço remoto.

| Medido no mesmo i7-10610U (Linux/WSL, prompt de 88 tokens) | v0.1 | **v0.2** |
|---|---|---|
| Tempo até o primeiro token | 29,0 s | **1,3 s** |
| Prefill | 3,0 tok/s | **67–71 tok/s** |
| Decode | 2,0 tok/s | **22–26 tok/s** |
| Heap do processo | 1024 MiB | **22 MiB** |
| Perplexidade (texto PT, 164 tokens) | 5,2750 | 5,2774 (+0,05%) |

Os números, o método e o que foi descartado no caminho estão em [docs/DECISIONS.md](docs/DECISIONS.md).

## Início rápido

### Windows (MSYS2 UCRT64 ou qualquer MinGW-w64 GCC/Clang)

```sh
make            # gera build/smolmoe.exe (estático em `make release`)
build/smolmoe.exe --model models/qwen2.5-0.5b-instruct/model.smoe \
  --config models/qwen2.5-0.5b-instruct/config.ini \
  --tokenizer models/qwen2.5-0.5b-instruct/tokenizer.qbp \
  --chat --prompt "Quanto é 7 mais 5? Responda apenas com o número."
```

### Linux / macOS

```sh
make
build/smolmoe --model models/qwen2.5-0.5b-instruct/model.smoe \
  --config models/qwen2.5-0.5b-instruct/config.ini \
  --tokenizer models/qwen2.5-0.5b-instruct/tokenizer.qbp \
  --chat --prompt "Escreva uma saudação curta para quem está aprendendo programação."
```

Os pesos não ficam no git. Veja [docs/MODELS.md](docs/MODELS.md) para obtê-los ou convertê-los.

O texto gerado vai para o stdout, em streaming. As estatísticas saem no stderr como uma linha JSON.
Saída real, sem edição:

```text
12
{"version":"0.2.0","backend":"avx2","threads":4,"batch":32,...,"ttft_seconds":0.858,"prefill_tokens_per_second":79.7,...}
```

## Opções principais

| Opção | Padrão | Efeito |
|---|---|---|
| `--chat` / `--system TEXTO` | system em PT-BR | Template ChatML de um turno (Qwen) |
| `--tokens N` | 64 | Máximo de tokens novos; para no EOS |
| `--temperature X` | 0 (guloso) | Amostragem quando > 0 |
| `--top-k N` / `--top-p X` | 40 / 0,95 | Truncamento da amostragem |
| `--seed N` | 1 | Amostragem reproduzível |
| `--threads N` | núcleos físicos | Threads de trabalho |
| `--batch N` | 32 | Tokens do prompt por passada (prefill) |
| `--context N` | min(modelo, 2048) | Posições do cache KV |
| `--memory-mib N` | 512 | Teto do heap; pesos mapeados não contam |
| `--stats`, `--trace`, `--hash-logits`, `--token-ids` | — | Diagnóstico e regressão |

A lista completa está em `smolmoe --help`.

## Como funciona

- **Pesos mapeados em memória** (`mmap` / `MapViewOfFile`): sem cópia, sem cache duplicado, e o SO gerencia as
  páginas. Antes, um cache LRU próprio sobre `pread` mantinha os pesos duas vezes na RAM e era o maior gargalo.
- **Produto escalar inteiro exato**: as ativações são quantizadas em blocos de 32 (Q8) e a soma de produtos int8 é
  exata. A acumulação em float segue uma ordem fixa de 8 lanes, que os kernels escalar e AVX2 reproduzem bit a bit.
- **Prefill em lote**: o mesmo código processa N tokens por passada. Cada linha de pesos é lida uma vez para 4 tokens.
- **Pool de threads com partição estática**: o resultado é o mesmo com qualquer número de threads.
- **Validação única na carga**: escalas finitas e não negativas, extents e shapes. O laço de inferência não tem checagens
  por acesso.

**Contrato de determinismo:** numa mesma plataforma, os logits são idênticos bit a bit para qualquer `--threads`,
`--batch` e kernel (AVX2/escalar). A suíte de testes garante isso. Entre sistemas operacionais ainda não há essa
garantia, porque `expf`/`cos`/`sin` vêm da libm local (ver o roadmap).

## Desenvolvimento

```sh
make test        # testes unitários em C + suíte de integração (bun test)
make sanitize    # ASan/UBSan (Linux/macOS)
make release     # binário otimizado e enxuto (estático no Windows)
bun scripts/bench.ts --model-dir models/qwen2.5-0.5b-instruct   # mediana de prefill/decode
```

Requisitos: compilador C99 (GCC ou Clang), `make` e [Bun](https://bun.sh) para os testes e scripts.
O runtime não tem dependências além da libc e da libm (e `pthread` em POSIX).

```text
src/        runtime C99 (plataforma, kernels, modelo, tokenizers, CLI)
tests/      testes unitários C + integração TypeScript (fixtures MoE sintéticas)
scripts/    benchmark, comparação de qualidade, geração de fixtures (TypeScript/Bun)
tools/      conversores de checkpoints (Python legado, POSIX)
docs/       formato, decisões, modelos, roadmap
models/     pesos locais (fora do git)
```

## Escopo honesto

- Arquiteturas suportadas: `smolmoe_v1` (GQA, RoPE, RMSNorm, SwiGLU, roteamento top-k sigmoide, experts
  compartilhados) e `smolmoe_qwen2` (com viés em QKV). Qualquer outra é recusada explicitamente.
- O roteamento MoE é testado só com checkpoints sintéticos. **Nenhum modelo MoE treinado é suportado ainda**; os modelos
  incluídos são densos, representados como um expert sempre ativo.
- Não é compatível com Kimi K3 nem com GGUF, e não há servidor HTTP nem modo multi-turno (estão no
  [roadmap](docs/ROADMAP.md)).
- O Qwen 0.5B é pequeno: às vezes mistura idiomas ou ignora o formato pedido. Revise as respostas.
- O checkpoint precisa ficar imutável durante a execução. No Windows o arquivo é aberto sem permitir escrita; no
  Linux/macOS, truncar ou reescrever o `.smoe` enquanto ele roda derruba o processo (`SIGBUS`), como em qualquer
  programa que usa `mmap`.

Licença MIT. Os pesos convertidos mantêm suas licenças originais (Apache-2.0 para Qwen, MIT para TinyStories); ver
[THIRD_PARTY.md](THIRD_PARTY.md).
