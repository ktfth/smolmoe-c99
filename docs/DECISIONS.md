# Decisões: o que caiu e o que sustenta o projeto

Este documento registra a revisão da v0.1 para a v0.2, feita com um critério simples: cada ideia precisa provar,
com medição, que serve ao objetivo **"modelo com alta eficiência, em produção"**. O que não provou foi removido.

Ambiente de medição: Intel i7-10610U (4 núcleos / 8 threads), 32 GB de RAM, Windows 11 e WSL2 (Linux 6.18), GCC 15.1
(MinGW UCRT64) e GCC 13 (Linux), modelo Qwen2.5-0.5B-Instruct Q8 (531 MiB).

## Ideias derrubadas

### 1. Cache LRU próprio de "tiles" sobre `pread`, para "memória limitada"

**Tese original:** ler os pesos sob demanda em blocos de 64 KB, com um cache associativo de 4 vias dentro de um
orçamento de memória.

**Por que caiu:**

- **Os pesos ficavam duas vezes na RAM:** uma no page cache do kernel (que o `pread` preenche) e outra no cache da
  aplicação. O "orçamento" excluía justamente a cópia maior, então a promessa de memória limitada não se cumpria.
- Havia lookup com hash e LRU **para cada pedaço de linha**, e uma chamada `tile_get` **para cada elemento** de norma e
  de viés.
- Com o modelo inteiro cabendo no cache (orçamento de 1 GiB), o decode continuava em 2 tok/s. Portanto o gargalo nunca
  foi o disco.

**Substituto:** `mmap` / `MapViewOfFile`. Leitura sem cópia, uma única cópia na RAM, páginas limpas que o SO pode
descartar sob pressão, e readahead do kernel. O `--memory-mib` passou a ter um significado honesto: o teto do heap (KV,
ativações, tokenizer). Heap medido: 22 MiB em vez de 1024 MiB.

### 2. "AVX2" que soma na ordem escalar

**Tese original:** garantir resultado idêntico entre escalar e AVX2 somando os produtos um a um, na ordem das colunas.

**Por que caiu:** o AVX2 só convertia e multiplicava; a soma continuava sequencial. Era SIMD apenas no nome. Para
manter a identidade bit a bit, o código abria mão de quase todo o ganho do SIMD.

**Substituto:** quantizar as ativações em Q8 (bloco de 32) e fazer o **produto inteiro exato** (`maddubs`/`madd`). A
soma inteira não depende de ordem; a acumulação em float segue uma ordem fixa de 8 lanes, que o escalar reproduz. A
identidade bit a bit continua garantida, agora sem custo de desempenho.

**Custo medido em qualidade:** perplexidade 5,2750 → 5,2774 (**+0,046%**), top-1 idêntico em 96,3% das posições e KL
médio de 0,0015 nats, em 164 tokens de texto em português com teacher forcing contra o motor da v0.1
(`scripts/compare-traces.ts`).

### 3. Uma única thread

Decode é limitado por banda de memória; uma thread não satura o barramento. Agora há um pool com partição estática
de linhas, e o resultado é idêntico com 1 a 256 threads (testado).

### 4. Prefill sequencial, um token por vez

Cada token do prompt relia todos os pesos. Agora o mesmo caminho de código processa lotes (padrão de 32 tokens), e o
kernel `x4` lê cada linha de pesos uma vez para 4 tokens, com 4 acumuladores independentes que escondem a latência de
soma em ponto flutuante. Prefill numa thread: 22 → 43 tok/s; com 4 threads: 67–95 tok/s.

### 5. Recalcular `powf`/`cosf`/`sinf` do RoPE por cabeça, por camada

Eram 24 camadas × 16 cabeças × 32 pares × 3 funções transcendentais por token. Agora a tabela é calculada uma vez
por token (em double) e reaproveitada em todas as camadas e cabeças.

### 6. Validar escalas Q8 a cada acesso

Um `isfinite` por bloco no laço mais quente. Agora todas as escalas e todos os valores F32 são validados **uma vez, na
carga, em paralelo**. A segurança é a mesma (checkpoints corrompidos continuam sendo recusados, e há testes para
isso), sem custo no laço.

### 7. Limite do prompt em bytes igual ao contexto em tokens

Um texto em português de 600 bytes (cerca de 160 tokens) era recusado com contexto de 512 tokens. O limite agora é de
4 bytes por posição de contexto, e a checagem real é feita sobre os tokens.

### 8. Só Linux/POSIX

O código original nem compilava no Windows. Agora existe uma camada de plataforma (`src/platform.c`,
`src/threadpool.c`) para Win32 e POSIX, com argv em UTF-8 no Windows, stdout binário e saída que nunca corta um
caractere UTF-8 no meio (importante em streaming no console).

### 9. Narrativa "inspirado no Kimi K3" e "MoE"

Nenhum modelo MoE treinado roda nesta base; os dois modelos reais são densos. A documentação deixou de vender MoE e
diz exatamente o que existe. O código de roteamento continua, porque é pequeno e coberto por testes sintéticos, mas
**suportar um MoE treinado de verdade** passou a ser item explícito do roadmap. Sem isso, o nome do projeto é uma
promessa e não uma capacidade.

### 10. Higiene do repositório

- Binário Linux pré-compilado sem procedência reproduzível: removido (o CI compila).
- 531 MiB de pesos dentro da árvore de código: movidos para `models/` e ignorados pelo git.
- Logs e JSONs de validação da versão antiga: removidos da árvore (continuam no primeiro commit). Eles descreviam um
  motor que não existe mais.
- Um arquivo C de 800 linhas com vários comandos por linha virou módulos coesos de 100 a 450 linhas.
- Testes em Python (que dependiam de recursos removidos) deram lugar a testes C (kernels, pool) e a uma suíte
  TypeScript/Bun (integração, validação de entradas hostis, goldens do tokenizer, modelos reais).


## Iteração 2: quantização abaixo de 8 bits

Ferramentas novas para decidir com dados: `--perplexity` (NLL com teacher forcing no próprio runtime) e
`smolmoe-quantize --simulate-bits N` (restringe os pesos Q8 a uma grade de N bits sem escrever kernels). Avaliação:
`tests/goldens/eval-pt.txt`, 1038 tokens de português em vários registros (narrativa, notícia, receita, ciência,
diálogo, código). **Limitação:** o texto foi escrito para o teste e é uma amostra única; diferenças abaixo de ~1% estão
dentro do ruído.

### Correção de uma afirmação minha

Na iteração 1 eu escrevi que o decode estava "no teto de banda de memória". Isso só vale para o Q8 com 4 threads neste
notebook (~15 GB/s). Um formato menor desloca o gargalo para a computação, porque o i7-10610U (15 W) baixa o clock com
AVX2 em todos os núcleos. A frase "basta ler menos bytes" estava errada da forma como a escrevi.

### Q4 (bloco 32, escala f16): derrubado

| Variante | Bytes/token | Perplexidade | Δ vs Q8 |
|---|---|---|---|
| Q8 | 556 MB | 11,39 | — |
| Q4 só no FFN | 380 MB | 12,86 | +12,9% |
| Q4 exceto embedding | 355 MB | 13,36 | +17,3% |
| Q4 em tudo | 278 MB | 13,76 | +20,8% |

O critério de aceite era menos de 3%. Num modelo de 0,5B, 4 bits destrói qualidade; a simulação de 4 bits (13,80)
reproduz o Q4 real, o que valida o método de simulação. Os kernels Q4 foram removidos. Um Q4 melhor (escalas por
sub-bloco, estilo K-quant) só volta a ser considerado com modelos de 1,5B ou mais.

### Q5 (bloco 32, escala f16, 22 bytes): mantido como opção de memória, não de velocidade

| | Q8 | Q5 | Q5 com embedding Q8 |
|---|---|---|---|
| Bytes por token | 556 MB | 340 MB (−39%) | 399 MB |
| Perplexidade | 11,39 | 11,58 (+1,7%) | 11,51 (+1,1%) |
| Decode, 1 thread | 15,5 tok/s | 11,9 tok/s | — |
| Decode, 4 threads | 27,3 tok/s | 17–22 tok/s | 19,2 tok/s |
| Decode, 8 threads | 25,5 tok/s | 25,4 tok/s | — |

Microbenchmark do kernel com dados no L1: Q8 1,7–2,2 ns por bloco, Q5 4,6 ns por bloco. O Q5 gasta cerca de 26
micro-ops por bloco (desempacotar 5 bits) contra cerca de 12 do Q8, num núcleo que despacha 4 por ciclo. Duas
otimizações foram medidas:

- Conversão f16 por F16C (instrução exata): decode de 15,7 para 22,1 tok/s. **Ficou.**
- Desempacotamento sem pressão na porta 5 e escalas pré-multiplicadas: nenhum ganho mensurável. Ficou por ser
  equivalente e mais simples de ler.

**Veredito:** neste hardware o Q5 não acelera; ele economiza 39% de RAM e disco por +1,7% de perplexidade. É útil em
máquinas apertadas e decisivo para modelos maiores (um Qwen 1,5B cai de 1,7 GB para 1,05 GB). O padrão continua sendo
Q8. Em máquinas com mais computação por byte de banda (desktops, servidores), o Q5 deve ganhar velocidade, mas isso
**não foi medido**.

## O que sustenta o projeto (o "pó" que ficou)

1. **C99, zero dependências, um executável.** Roda em qualquer lugar e é fácil de auditar e de embarcar.
2. **Formato `SMOEQ8` validado de forma estrita.** Diretório ordenado, extents sem sobreposição e shapes checados
   contra a config. Entradas hostis falham com erro claro, e há testes para cada caso.
3. **Tokenizers corretos sem bibliotecas.** BPE do Qwen2 com NFC do Unicode 15 e as regras de pré-tokenização,
   validado contra o `tokenizers` do Hugging Face em 11 casos.
4. **Determinismo como contrato testado.** Ele não depende mais de abrir mão de desempenho.
5. **Português local, de verdade.** A resposta "12" sai em 0,86 s no Windows nativo, num notebook.

## Próximas demolições candidatas

Ver [ROADMAP.md](ROADMAP.md). As suspeitas atuais, ainda sem medição:

- `expf`/`cos`/`sin` da libm impedem hashes idênticos entre Windows e Linux.
- O kernel Q5 custa 2,6× o Q8 por bloco; uma variante com desempacotamento por tabela ou 2 blocos por iteração pode
  torná-lo vantajoso também em velocidade.
- Os conversores ainda estão em Python e o `q8pack.c` é só POSIX.
