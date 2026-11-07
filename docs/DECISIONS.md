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

## Iteração 5: esteira de desempenho e o trabalho fora das matrizes

Desempenho virou a prioridade. Antes de otimizar, o projeto ganhou como medir.

### A esteira

- `--profile`: o tempo de cada passada dividido por fase (normalização, QKV, atenção, FFN, ativação etc.), nas
  estatísticas JSON. Não muda nenhum bit do resultado (testado).
- `scripts/bench.ts`: cenários fixos ("leitura", com 301 tokens de entrada, e "chat", com 88 de entrada e 64 gerados),
  medianas, tempo por fase e comparação com uma base. Sai com erro se uma métrica piorar além do limiar.
- **Duas armadilhas de medição encontradas e corrigidas:**
  - O `--hash-logits` calcula as 151 mil pontuações do vocabulário para todo token do prompt (18,6% do tempo). O chat
    nunca faz isso, então ele saiu do benchmark. A igualdade entre rodadas é conferida pelo texto gerado.
  - O calor do notebook mudou a escrita de 21,5 para 15,5 tok/s com o mesmo código. Comparar com uma base medida em
    outro momento não é confiável aqui. O modo `--ab BINARIO_ANTIGO` roda os dois binários alternados na mesma sessão.

### O diagnóstico

Prefill de 388 tokens, 4 threads, antes da mudança:

| Fase | 1 thread | 4 threads |
|---|---|---|
| FFN (gate/up/down) | 66,7% | 56,1% |
| Atenção | 14,2% | 13,5% |
| Quantização + ativação + normalização | 9,2% | **21,3%** |
| QKV + saída da atenção | 9,7% | 8,8% |

A quantização, a ativação (SiLU com `expf`) e a normalização rodavam numa thread só e **não encolheram com 4 threads**.
Com 1, 2, 4 e 8 threads, o prefill ia a 30,8, 47,3, 54,8 e 62,4 tok/s.

### A mudança

- **Exponencial própria**, idêntica bit a bit no caminho escalar e no AVX2. Usa uma tabela de 32 valores de 2^(j/32)
  (gerada com aritmética inteira exata em `scripts/gen-exp-table.ts`) e um polinômio de grau 4, tudo em double, com um
  único arredondamento para float.
- **SiLU e quantização em AVX2**, idênticas bit a bit às versões escalares, inclusive em infinitos, NaN e todos os
  tamanhos de cauda.
- **Trabalho por token dividido entre as threads**, com a linha de cada token inteira numa thread só. Por isso o
  resultado não depende do número de threads.

### Ideias derrubadas

- **Exponencial em float (polinômio do Cephes):** errava 1 ulp em 10% das entradas, contra 0,01% do `expf` do sistema.
  Isso já bastou para mudar uma resposta do modelo (abaixo). A versão em double acerta o arredondamento em mais de
  99,99% e ficou.
- **"A mudança de exponencial é a única mudança numérica":** isso era uma hipótese até ser testada. O código novo
  compilado com o `expf` do sistema dá logits **idênticos bit a bit** aos do binário antigo em todas as 63 posições do
  teste. Vetorizar e paralelizar a quantização, a ativação e a normalização não mudou nenhum bit.

### Um efeito que a medição revelou

O teste de histórico do chat passou a falhar: "Meu nome é Ana e eu moro em Recife." virou "Meu nome é Renato…" na
resposta do modelo. O rastreamento dos logits mostrou o mecanismo:

- As posições 0 a 33 continuaram idênticas bit a bit.
- Nas posições 34 a 38 houve diferenças de 10⁻⁵ (uma exponencial 1 ulp diferente).
- Na posição 39, a diferença saltou para 0,39. Um valor que caía na fronteira de arredondamento de 8 bits mudou de
  degrau, e isso se propagou pelas 24 camadas.
- Na posição 55, a versão antiga escolhia entre dois tokens com margem de **0,02**, e a escolha virou.

O teste verifica o histórico do runtime, não a sorte do modelo. A mensagem passou a ser "Meu nome é Ana.", que as duas
versões respondem igual e com folga. **Lição:** um modelo que quantiza as ativações amplifica diferenças mínimas, e a
saída gulosa em empates próximos não é estável sob nenhuma mudança numérica.

### Resultado

| | Antes | Depois | Variação |
|---|---|---|---|
| Leitura (301 tokens), tok/s | 45,1 / 44,4 | 57,6 / 53,9 | **+27,6% / +21,5%** |
| Leitura do chat (88 tokens), tok/s | 47,9 / 51,0 | 66,7 / 67,5 | **+39,3% / +32,2%** |
| Tempo até o 1º token no chat | 2,08 / 1,96 s | 1,54 / 1,51 s | **−26% / −23%** |
| Escrita, tok/s | 19,1 / 16,8 | 21,4 / 16,5 | +11,7% / −2,1% (não confirmado) |
| Perplexidade (1038 tokens) | 11,3885 | 11,3681 | −0,18% (ruído) |
| Datas: desenvolvimento / perguntas novas | 16/17 · 12/15 | 16/17 · 13/15 | igual (1 resposta virou) |

Duas sessões A/B, 5 rodadas cada. A fase "fora das matrizes" caiu de 23% para cerca de 3% do prefill. Agora o prefill
é 70% produto de matrizes e 13% atenção: são os próximos alvos.

**Ganho colateral:** sem `expf` da libm no caminho do modelo, Windows e Linux passam a produzir os mesmos números,
exceto pelas tabelas de RoPE, que ainda usam `cos`/`sin` em double.

### Determinismo bit a bit: continua

Com desempenho como prioridade, revimos a regra que exige resultado idêntico com qualquer número de threads, lote e
kernel. Ela continua:

- Tudo o que o diagnóstico apontou foi resolvido sem quebrá-la.
- Até a instrução de multiplicar e somar (FMA) cabe nela, porque o `fmaf` do C dá o mesmo resultado.
- É ela que torna a esteira barata: se o hash dos logits não mudou, a otimização não mudou nenhum número e a qualidade
  não precisa ser medida de novo.

**Critério de exceção:** se uma medição mostrar que a regra bloqueia um ganho acima de 15%, a regra cai de propósito e
o efeito na qualidade é medido (perplexidade e conjuntos de perguntas) antes de aceitar.

## Iteração 6: Phi-4-mini, o primeiro modelo fora da família Qwen

**Pedido:** suporte a Gemma e Phi. **Decisões do usuário:** começar pelo Phi-4-mini, validar sem referência externa
(nem Python, nem llama.cpp) e obter o Gemma pelo aceite dos termos no Hugging Face (pendente).

### O que o motor ganhou

- **Rotação parcial e fator do RoPE**, como chaves opcionais da config (`rotary_dim`, `rope_attention_factor`). O
  Phi-4-mini roda 96 das 128 dimensões de cada cabeça e multiplica cos/sin por √(17/12) ≈ 1,19024. Sem as chaves, o
  resultado é idêntico bit a bit ao anterior (testado).
- **Tokenizer do GPT-4o (`o200k`)** e **template de chat do Phi**, com o formato `.qbp` v2. O v1 continua valendo: as
  11 referências do Qwen e um corpus adversarial de 403 casos ficaram idênticos.
- **Tabela de classes Unicode no 16.0**, a mesma versão do motor de regex que o Hugging Face usa. O gerador foi
  portado para TypeScript, com os arquivos da Unicode fixados por SHA-256.
- **Orçamento de memória padrão** = 512 MiB + o cache de atenção de até 2048 posições. Com 2048 posições, o cache
  do Phi sozinho ocupa 512 MiB; antes era preciso passar `--memory-mib` à mão. O teto de 2048 vem da revisão: sem
  ele, um `--context` enorme pediria dezenas de GB sem a mensagem que orienta o usuário.
- **Conversor em TypeScript** (`scripts/convert.ts`) no lugar do Python. Ele reproduz os 5 arquivos do Qwen **byte a
  byte** (inclusive os hashes do Release) e converte o Phi em cerca de 40 s.

### Como validamos sem referência externa

1. **Especificação a partir do código oficial.** Toda fórmula foi citada com arquivo e linha do `modeling_phi3.py` e
   do `modeling_rope_utils.py`.
2. **Implementação de referência independente**, em TypeScript e em double. Ela lê os pesos bf16 originais e segue a
   semântica do Hugging Face (meia rotação, pesos fundidos, IDs originais). Não usa nada do conversor nem do runtime,
   e tem 57 testes com valores calculados à mão. Para "Qual é a capital do Brasil?", ela continua com "A capital do
   Brasil é Brasília." Um erro sutil de implementação dificilmente produziria português correto.
3. **Tokenizer:** 42 casos comparados com o `js-tiktoken` (`o200k_base`), mais 300 mil textos aleatórios na regra de
   divisão. Dois agentes escreveram, cada um por conta própria, o gerador do `.qbp` do Phi, e os arquivos saíram
   idênticos byte a byte.
4. **Motor contra a referência, posição por posição:**

| Texto | Mesma escolha (top-1) | KL típica | Onde diverge |
|---|---|---|---|
| Chat de 18 tokens | 18/18 | 10⁻³ a 10⁻² | — |
| Português corrido, 67 tokens | 64/67 | 10⁻³ a 5×10⁻³ | 3 quase empates na referência (margens 0,006, 0,034, 0,087) |

A diferença **não cresce com a posição**; um erro de RoPE cresceria. O maior desvio fica na posição 0 (KL 0,08 a
0,2), onde a rotação é a identidade e a atenção vê um único token. A hipótese, não medida, é o arredondamento para
8 bits das ativações grandes do primeiro token.

### Resultado

| | Qwen2.5-0.5B | Phi-4-mini (3,8B) |
|---|---|---|
| Bits por byte no texto em português (menor é melhor) | 0,985 | **0,818** (−17%) |
| Escrita / leitura neste notebook | ~17–21 / ~55 tok/s | ~3,9 / ~12 tok/s |
| Memória | 0,55 GB | 4,4 GB |
| 15% de 230; desconto de 25% em 80; 10 milhas em km | "030", 50, 1,61 | **34,5, 60, 16,093** |
| Comando PowerShell para renomear um arquivo | nome errado | **certo** |

A perplexidade bruta (11,4 contra 12,0) não serve para comparar: o Phi divide o mesmo texto em 843 tokens, e o Qwen
em 1038.

### O que ficou de fora

- **Ferramentas no chat do Phi:** as fontes divergem sobre o formato de chamada de função. O chat do Phi roda sem
  ferramentas até que isso seja testado no próprio modelo.
- **Contexto acima de 4096:** exigiria os fatores longos do "longrope", que mudam a partir da posição 4096. O
  conversor declara `max_context=4096`.
- **Release:** 4,3 GB passa do limite de 2 GB por arquivo do GitHub, então o Phi é obtido pela conversão.
- **Gemma:** aguarda o aceite dos termos. Exige mais peças no motor (GELU, 4 normalizações por camada, normalização
  de Q/K, atenção local de 512 tokens, duas frequências de RoPE, cabeça de tamanho próprio) e um tokenizer
  SentencePiece de 262 mil peças.

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

- `cos`/`sin` da libm (tabelas de RoPE) ainda podem impedir hashes idênticos entre Windows e Linux. O `expf` já saiu
  (iteração 5).
- O kernel Q8 de prefill usa cerca de metade do que o processador permite por bloco (estimativa: ~6 ciclos contra ~3) e
  só ganha 2,4× com 4 threads.
- O kernel Q5 custa 2,6× o Q8 por bloco; uma variante com desempacotamento por tabela ou 2 blocos por iteração pode
  torná-lo vantajoso também em velocidade.
- Os conversores ainda estão em Python e o `q8pack.c` é só POSIX.
