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


## Iteração 3: o modelo é um arquivo, e o arquivo é um programa

### Ideia derrubada: "modelo = três arquivos + flags, sem conversa"

Para usar o Qwen era preciso apontar `--model`, `--config` e `--tokenizer`, montar um `--prompt` e rodar um processo por
pergunta. Qualquer pessoa fora do terminal desistiria antes da primeira resposta.

### O que entrou

| Medida (i7-10610U, Windows) | Três arquivos | `qwen-pt.exe` |
|---|---|---|
| Arquivos necessários | 3 + executável | 1 |
| Tamanho | 556 MB + 4 MB + 200 KB | 560 MB (340 MB com Q5) |
| Partida (`load_seconds`) | 0,19 s | 0,15 s |
| Hash dos logits | referência | idêntico (Windows e Linux, testado) |
| Dependências | — | só DLLs do sistema (KERNEL32, UCRT, SHELL32) |

- **Contêiner v3 com blobs** (`__config__`, `__tokenizer__`). O tokenizer e a config passaram a ser lidos de memória,
  num leitor único que serve tanto arquivo quanto blob.
- **`smolmoe-bundle`**: valida a consistência entre config, tokenizer e pesos construindo o modelo antes de escrever,
  e revalida a saída depois. Ele recusa, por exemplo, uma config com o número de camadas errado.
- **Chat multi-turno** com cache KV reaproveitado. A memória entre turnos foi testada ("Meu nome é Ana", depois "Qual é o
  meu nome?" responde "Ana") e o `/limpar` também ("Ana" some).

### Achados durante a iteração

- **Bug real no Windows em português:** as mensagens de erro do sistema saíam como "n�o", porque `FormatMessageA` usa
  a codepage local. Corrigido para `FormatMessageW` com conversão para UTF-8.
- **Qualidade do modelo, não do runtime:** perguntado sobre a capital do Brasil, o Qwen 0,5B respondeu "São Paulo".
  Um executável fácil de usar torna esses erros mais visíveis, e o README diz para revisar as respostas.
- **Revisão independente do código da iteração:** nada de segurança de memória, overflow ou posição de contexto. Os
  5 achados menores foram corrigidos:
  - Linha longa no console do Windows derrubava o chat; agora é cortada na fronteira de caractere e avisada.
  - Linhas longas viravam vários turnos.
  - Pares surrogate (emojis) partidos entre leituras viravam lixo.
  - Falha ao mapear o próprio executável era fatal.
  - `remove` + `rename` não era atômico no Windows (agora `MoveFileExW`).

  Um teste de regressão novo falha no binário antigo e passa no novo. A leitura via `ReadConsoleW` (console
  interativo do Windows) não é coberta por teste automático, porque um pipe não exercita esse caminho.

## Iteração 4: o modelo chama funções

A pergunta que motivou a iteração: "quanto tempo falta para o Natal?". O modelo não tem relógio e inventava ("10 dias
antes do Natal"). Com a data no prompt de sistema, errava a conta ("2026 − 2026 = 0 dias"). Um modelo de 0,5B não faz
aritmética de datas.

### Ideia derrubada: "o Qwen 0,5B não sabe chamar funções"

Ele sabe. Quem impedia era o runtime, de duas formas:

- O conversor marca todo token extra do Qwen como de controle, inclusive `<tool_call>`, e o sampler banía todos.
- O texto `<tool_call>` no prompt de sistema virava caracteres comuns, e no treino ele é um token só.

Corrigida a codificação, a probabilidade do primeiro token da resposta ser `<tool_call>` subiu da **posição 3183 para
a posição 1**, com p = 0,93 (pergunta em inglês) e 0,73 (em português). O texto do usuário continua codificado como
texto puro, então quem digita `<tool_call>` não injeta uma chamada (há teste para isso).

### Ideia derrubada: "basta o modelo extrair os argumentos"

A primeira versão de `dias_ate` recebia mês e dia como inteiros. O modelo escolhia a ferramenta certa, mas convertia
mal: "Natal" virava 30/11, e "12 de outubro" virava o mês 12. Converter nome de mês em número é raciocínio, e um 0,5B não
faz isso.

O que entrou foi o princípio de que **o modelo decide e copia, e o C calcula**:

- `dias_ate` recebe a data como texto, do jeito que o usuário escreveu. O `calendar.c` entende "25 de dezembro",
  "25/12", "2026-12-25" e feriados, inclusive os móveis (Páscoa, Carnaval, Corpus Christi, Dia das Mães e dos Pais).
- **Aterramento:** se a mensagem do usuário tem uma data, ela vale mais que o argumento do modelo. Um ano passado
  inventado pelo modelo é descartado, e vale a próxima ocorrência.
- **Portão por mensagem:** as ferramentas ficam declaradas no prompt de sistema, mas o sampler só pode produzir
  `<tool_call>` quando a mensagem fala de data ou de tempo ("hoje", "horas", "falta", uma data ou um feriado). Antes do
  portão, o modelo chamava `dias_ate` para "qual é a capital da França?".
- O resultado traz uma frase pronta (`resumo`) como **primeiro** campo. O modelo a copia, e para de escrever
  "49 dias atrás".

### Medição (`bun scripts/eval-tools.ts`)

Cada pergunta roda num chat novo, com o relógio fixo em 2026-11-06 10:30 (uma sexta-feira) e decodificação gulosa,
portanto o resultado é determinístico. Uma resposta só conta como certa com o número exato e sem "atrás"/"passou"
para datas futuras.

**Conjunto de desenvolvimento** (`--set dev`): 12 perguntas sobre datas e 5 de controle. Foi com ele que escolhi o
desenho. Por isso o número dele é otimista:

| Variante | Ferramenta certa | Resposta certa | As duas |
|---|---|---|---|
| A. `dias_ate(mes, dia)` inteiros, sem portão | 15/16 | 10/14 | 12/16 |
| B. data como texto, `calendar.c`, campo `resumo` | 13/16 | 13/14 | 13/16 |
| C. B sem os exemplos na descrição da ferramenta | 14/17 | 12/15 | 12/17 |
| D. B com portão por mensagem e aterramento | 17/17 | 13/15 | 15/17 |
| E. D com o `resumo` como primeiro campo | 17/17 | 14/15 | 16/17 |
| F. E com "Hoje é ..." no início do `resumo` (derrubada) | — | — | sessão: 4/17, era 10/17 |
| **G. E com validação da chamada e resposta nunca vazia (atual)** | **17/17** | **14/15** | **16/17** |

As variantes A e B ainda aceitavam "49 dias atrás" como certo, e a pergunta de carnaval só entrou na C. Por isso os
números de A e B estão otimistas.

**Perguntas novas.** Um revisor apontou o problema certo: medir no mesmo conjunto usado para escolher o desenho é
estudar com a prova. Fiz então dois conjuntos de perguntas novas, cada um escrito antes de rodar:

| Conjunto | Versão | Ferramenta certa | Resposta certa | As duas |
|---|---|---|---|---|
| `holdout`: 17 perguntas, inclusive controles com "hora", "horas" e "feriado" | E | 13/17 | 13/15 | 12/17 |
| `holdout`, depois de corrigir o que ele mostrou (já não é inédito) | G | 16/17 | 13/15 | 15/17 |
| **`holdout2`: 15 perguntas, rodado uma única vez** | **G** | **14/15** | **9/12** | **12/15** |

**O número honesto é o do `holdout2`: 12 de 15 (80%).** As perguntas sobre datas acertam bem mais que as outras: o
número de dias sai exato sempre que a ferramenta é chamada. O que o `holdout` revelou e foi corrigido:

- **Chamadas indevidas.** O portão abre com "horas" ou "feriado", e o modelo chamava `dias_ate` em "Quantas horas tem
  um dia?" ("Um dia tem 24 horas. Para 1 de janeiro de 2027 ... há 56 dias").
  - Correção: o runtime **valida a chamada antes de executar**. `dias_ate` só vale se a mensagem nomeia uma data ou um
    feriado.
  - Se nenhuma chamada da resposta vale, ela é desfeita (o cache KV é sobrescrito a partir do início da resposta), e o
    modelo responde de novo com as ferramentas banidas.
- **Respostas vazias.** Com o `<tool_call>` banido, a segunda escolha do modelo costumava ser o fim do texto. Agora o
  primeiro token de uma resposta nunca pode ser `<|im_end|>`.
- **Encadeamento de chamadas.** Agora há uma rodada de ferramenta por mensagem: depois do resultado, o modelo responde
  sem poder chamar de novo.

O que o `holdout2` mostrou, e fica **sem corrigir** de propósito (corrigir agora exigiria um terceiro conjunto inédito
para medir):

- "Falta muito para o réveillon?" sai com o número certo, mas como "55 dias atrás".
- "Estamos em que ano?" não abre o portão, porque "ano" não está na lista de palavras.
- Em "Quantos dias tem uma semana?", a chamada foi descartada. Na segunda tentativa, sem ferramentas, o modelo
  escreveu o JSON da chamada como texto, e ele aparece para o usuário.
- Dia da semana ("Quantos dias até sexta?") não é suportado por `dias_ate`.
- Perguntas de conhecimento falham com e sem ferramentas: "Dom Casmurro" sai como "José Saramago", e o maior planeta
  sai como "Mercúrio". O modelo tem 0,5B.

**Conversa longa** (`--session`): todas as perguntas numa conversa só. A resposta da pergunta *i* é o trecho de saída
que aparece quando a conversa passa de *i−1* para *i* perguntas. O script confere essa propriedade de prefixo, então ela
também testa o determinismo.

- Nas versões E e G, a escolha de ferramenta foi 11/17 e a resposta, 10/15. A validação não muda a conversa longa.
- No meio da conversa, o modelo deixa de chamar `data_atual` e inventa a hora ("14h") e o mês ("outubro").
- Esse continua sendo o ponto mais fraco. `/limpar` volta ao comportamento de chat novo.

### Custo

As declarações levam o prompt de sistema de 31 para 279 tokens. **Abrir o chat passou de 0,6 s para 5,9 s** neste
notebook (mediana de 3). Elas ficam ligadas por padrão no chat, e `--no-tools` desliga. Num contexto pequeno demais
(as declarações passariam de metade dele), o chat as desliga sozinho e avisa. O próximo passo para esconder esse custo
está no roadmap.

### Escopo honesto

- Só há ferramentas embutidas (`data_atual` e `dias_ate`), e só no chat interativo.
- Ferramentas definidas pelo usuário e o modo `--chat --prompt` com ferramentas ficam para o servidor compatível com
  OpenAI, que tem o campo `tools` e o contrato certo para isso.

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
