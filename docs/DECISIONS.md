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
- O decode está perto do teto de banda de memória (cerca de 13 GB/s neste notebook), e só ler menos bytes (Q4) ajuda.
- Os conversores ainda estão em Python e o `q8pack.c` é só POSIX.
