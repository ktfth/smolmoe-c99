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

## Um arquivo só: o executável já é o modelo

```sh
make qwen-pt                      # gera dist/qwen-pt.exe (Windows) ou dist/qwen-pt: runtime + Qwen, 560 MB
dist/qwen-pt.exe                  # duplo clique ou terminal: abre o chat em português
dist/qwen-pt.exe --chat --prompt "..."   # uma resposta e sai (scripts, pipes)
```

Como o llamafile, o binário carrega os próprios pesos: não precisa de instalação, DLL extra, Python nem arquivo ao
lado. O chat lembra a conversa (o cache KV é reaproveitado entre turnos); `/limpar` reinicia e `/sair` encerra. Para um
executável de 340 MB, empacote a variante Q5 (veja abaixo) com `build/smolmoe-bundle --model ...-q5.smoe
--config ... --tokenizer ... --exe build/smolmoe.exe --out dist/qwen-pt-q5.exe`. O formato do pacote está em
[docs/FORMAT.md](docs/FORMAT.md).

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

Os pesos não ficam no git: baixe com `bun scripts/fetch-models.ts` (ou `make models`), que confere o SHA-256 de cada
arquivo. Detalhes e conversão a partir dos originais em [docs/MODELS.md](docs/MODELS.md).

**Modelo maior: Phi-4-mini (Microsoft, 3,8B).** É 17% melhor em português e acerta contas em que o Qwen 0,5B erra,
mas escreve a cerca de 4 tokens/s neste notebook e usa 4,4 GB. Ele é obtido pela conversão:

```sh
bun scripts/convert.ts fetch-phi3 downloads/phi-4-mini-instruct     # 7,7 GB, SHA-256 conferido
bun scripts/convert.ts phi3 downloads/phi-4-mini-instruct models/phi-4-mini-instruct
build/smolmoe --model models/phi-4-mini-instruct/model.smoe --config models/phi-4-mini-instruct/config.ini \
  --tokenizer models/phi-4-mini-instruct/tokenizer.qbp
```

O texto gerado vai para o stdout, em streaming. As estatísticas saem no stderr como uma linha JSON.
Saída real, sem edição:

```text
12
{"version":"0.2.0","backend":"avx2","threads":4,"batch":32,...,"ttft_seconds":0.858,"prefill_tokens_per_second":79.7,...}
```

## Opções principais

| Opção | Padrão | Efeito |
|---|---|---|
| (sem `--prompt`) / `-i` | — | Chat multi-turno no terminal (modelos de chat) |
| `--chat` / `--system TEXTO` | system em PT-BR | Template ChatML de um turno com `--prompt` |
| `--no-tools` | ferramentas ligadas no chat | Chat sem as funções de data e de conta (abre cerca de 5,5 s mais rápido) |
| `--now AAAA-MM-DDTHH:MM` | relógio local | Fixa a hora que as ferramentas usam (execuções reproduzíveis) |
| `--tokens N` | 64 (512 no chat) | Máximo de tokens novos; para no EOS |
| `--temperature X` | 0 (guloso) | Amostragem quando > 0 |
| `--top-k N` / `--top-p X` | 40 / 0,95 | Truncamento da amostragem |
| `--seed N` | 1 | Amostragem reproduzível |
| `--threads N` | núcleos físicos | Threads de trabalho |
| `--batch N` | 32 | Tokens do prompt por passada (prefill) |
| `--context N` | min(modelo, 2048) | Posições do cache KV |
| `--memory-mib N` | 512 + cache KV (até 2048 posições) | Teto do heap; pesos mapeados não contam |
| `--perplexity` | — | Mede a perplexidade do prompt (sem gerar) |
| `--stats`, `--trace`, `--hash-logits`, `--token-ids` | — | Diagnóstico e regressão |

A lista completa está em `smolmoe --help`.

### Funções no chat: datas exatas

O modelo não tem relógio e não sabe fazer conta com datas. No chat, ele chama funções do runtime, e o C faz a conta:

```text
> Quanto tempo falta para o natal?
[ferramenta] dias_ate: {"name": "dias_ate", "arguments": {"data": "01/12/2023"}} -> {"resumo": "Faltam 49 dias para 25 de dezembro de 2026 (sexta-feira).", ...}
Para o natal, você tem 49 dias para o dia 25 de dezembro de 2026 (sexta-feira).
```

Esta é a saída real, com `--now 2026-11-06T10:30`. Nela, o modelo passou uma data errada ("01/12/2023"), e o runtime
usou a data da própria pergunta ("natal"). Para isso existe o aterramento.

- `data_atual` informa a data, o dia da semana e a hora.
- `dias_ate` aceita "25 de dezembro", "25/12" ou o nome de um feriado brasileiro, inclusive os móveis (Páscoa,
  Carnaval, Dia das Mães).
- As ferramentas só são oferecidas quando a mensagem fala de data ou de tempo. A data escrita pelo usuário vale mais
  que o argumento do modelo. Uma chamada que não combina com a mensagem é descartada, e o modelo responde sem ela.
- Em perguntas novas, que não foram usadas no desenho, **12 de 15 saem certas** (`bun scripts/eval-tools.ts --set
  holdout2`). Quando a ferramenta é chamada, o número de dias sai exato.
- Os pontos fracos medidos:
  - Em conversas longas, o modelo deixa de chamar as ferramentas e inventa a hora. Use `/limpar`.
  - Não entende dia da semana ("até sexta").
  - Às vezes escreve "atrás" em vez de "faltam".

  Os detalhes estão em [docs/DECISIONS.md](docs/DECISIONS.md), iteração 4.

### Função no chat: contas (`calcular`)

O modelo escreve a conta, e o programa a calcula em double e responde no formato brasileiro:

```text
> Quanto é 15% de 230?
[ferramenta] calcular: {"name": "calcular", "arguments": {"expressao": "15% de 230"}} -> {"resumo": "O resultado é 34,5.", ...}
O resultado é 34,5.
```

- Entende símbolos e palavras ("mais", "vezes", "dividido por", "15% de 230", "raiz(144)") e números como "1.234,56"
  ou "dois mil e quinhentos".
- Todo número da conta tem que vir da mensagem (exceto constantes pequenas e porcentagens). Uma conta com número
  inventado é descartada.
- **Limite medido, e importante:** o programa calcula certo a conta que recebe, mas quem escreve a conta é o modelo.
  No teste às cegas, o Qwen 0,5B acertou 12 de 40 contas com a ferramenta e 11 de 40 sem ela. Ela acerta conta direta
  e porcentagem simples, mas faz errar quando o modelo monta a fórmula errada (troco como "48,20 − 50" dá R$ −1,80).
  O aviso `[ferramenta]` mostra a conta escrita. Os detalhes estão em [docs/DECISIONS.md](docs/DECISIONS.md),
  iteração 7.

### Q5: 39% menos memória

```sh
build/smolmoe-quantize --in models/qwen2.5-0.5b-instruct/model.smoe --out models/qwen2.5-0.5b-instruct/model-q5.smoe
```

O arquivo Q5 tem 340 MB em vez de 556 MB, com perplexidade +1,7% (+1,1% mantendo o embedding em Q8 com
`--keep embed --keep lm_head`). Neste notebook ele **não** é mais rápido: o desempacotamento de 5 bits custa mais
computação do que a banda economizada. Use-o quando a memória for o limite.

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
`--batch` e kernel (AVX2/escalar). A suíte de testes garante isso. A exponencial é própria e idêntica em todo sistema
operacional; entre sistemas ainda não há garantia formal só porque as tabelas de RoPE usam `cos`/`sin` da libm local.

## Desenvolvimento

```sh
make test        # testes unitários em C + suíte de integração (bun test)
make sanitize    # ASan/UBSan (Linux/macOS)
make release     # binário otimizado e enxuto (estático no Windows)
bun scripts/bench.ts                                    # medianas de leitura/escrita e tempo por fase
bun scripts/bench.ts --ab build/smolmoe-base.exe        # A/B alternado contra um binário anterior
```

Para julgar uma otimização: copie o binário atual (por exemplo, para `build/smolmoe-base.exe`), faça a mudança,
recompile e rode o modo `--ab`. Os dois binários rodam alternados na mesma sessão, e o calor do processador afeta os
dois por igual. O script sai com erro se alguma métrica piorar além do limiar (`--threshold`, padrão 5%).
`--profile` no runtime mostra para onde vai o tempo de cada passada.

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

- Arquiteturas suportadas: `smolmoe_v1` (GQA, RoPE com rotação parcial opcional, RMSNorm, SwiGLU, roteamento top-k
  sigmoide, experts compartilhados) e `smolmoe_qwen2` (com viés em QKV). Qualquer outra é recusada explicitamente.
- Modelos convertidos e validados: Qwen2.5-0.5B-Instruct, Phi-4-mini-instruct (até 4096 tokens de contexto, chat
  sem ferramentas) e TinyStories. O Gemma está no [roadmap](docs/ROADMAP.md).
- O roteamento MoE é testado só com checkpoints sintéticos. **Nenhum modelo MoE treinado é suportado ainda**; os modelos
  incluídos são densos, representados como um expert sempre ativo.
- Não é compatível com Kimi K3 nem com GGUF, e ainda não há servidor HTTP (está no [roadmap](docs/ROADMAP.md)).
- O Qwen 0.5B é pequeno: às vezes mistura idiomas ou ignora o formato pedido. Revise as respostas.
- O checkpoint precisa ficar imutável durante a execução. No Windows o arquivo é aberto sem permitir escrita; no
  Linux/macOS, truncar ou reescrever o `.smoe` enquanto ele roda derruba o processo (`SIGBUS`), como em qualquer
  programa que usa `mmap`.

Licença MIT. Os pesos convertidos mantêm suas licenças originais (Apache-2.0 para Qwen, MIT para Phi e TinyStories); ver
[THIRD_PARTY.md](THIRD_PARTY.md).
