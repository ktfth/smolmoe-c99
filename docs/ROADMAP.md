# Roadmap

Ordem definida por impacto no objetivo (eficiência em produção) dividido pelo risco. Cada item só fecha com medição
antes/depois registrada em [DECISIONS.md](DECISIONS.md).

## Próximo: eficiência (prioridade principal)

Cada item é medido com `bun scripts/bench.ts --ab` contra o binário anterior. A ordem vem do perfil por fase
(DECISIONS, iteração 5): o prefill agora é 70% produto de matrizes e 13% atenção.

0. **Kernel de matrizes do prefill.** Ele usa cerca de metade do que o processador permite por bloco e só ganha 2,4×
   com 4 threads. Caminhos a medir:
   - blocos de registradores (várias linhas × vários tokens);
   - FMA com `fmaf` no escalar, o que mantém a identidade bit a bit;
   - threads lógicas no prefill (8 threads deram +13%) e físicas na escrita.
1. **Atenção vetorizada** (13% do prefill): produto escalar e soma ponderada em AVX2, e softmax com a exponencial
   própria em 8 lanes.
2. **Esconder o custo das ferramentas na abertura do chat.** As declarações custam cerca de 5 s de prefill a cada
   abertura (DECISIONS, iteração 4); os itens 0 e 1 já reduzem isso. Há dois caminhos:
   - Fazer o prefill do prompt de sistema numa thread enquanto o usuário digita a primeira mensagem. No uso
     interativo, o custo some.
   - Gravar o cache KV desse prefixo fixo junto com o executável (cerca de 7 MB) e mapeá-lo na partida.
3. **`sin`/`cos` próprios nas tabelas de RoPE.** É o último uso da libm no caminho do modelo (o `expf` saiu na
   iteração 5). Sem ele, os hashes ficam idênticos entre sistemas operacionais e a suíte pode fixar hashes-ouro.
4. **Kernel Q5 mais barato.** O Q5 já existe e economiza 39% de memória por +1,7% de perplexidade, mas custa 2,6× o Q8
   por bloco e não acelera neste notebook (ver DECISIONS, iteração 2). Opções: dois blocos por iteração
   compartilhando a expansão dos bits altos, ou um layout com os bits altos já intercalados para SIMD.
5. **Q4 com escalas por sub-bloco (estilo K-quant)**, só junto com modelos de 1,5B+. O Q4 simples custou +21% de
   perplexidade no 0,5B e foi removido.
6. **Kernels NEON (ARM64).** Mesma ordem de lanes. Abre Apple Silicon, Raspberry Pi e servidores Graviton.
7. **Cache KV em F16.** Metade da memória e da banda em contextos longos.

## Funções no chat: o que a medição deixou aberto

Detalhes em DECISIONS, iteração 4. Cada correção só conta com um **conjunto novo de perguntas, escrito antes de
rodar**: os dois atuais já serviram para decidir.

1. **Conversa longa.** No meio de uma sessão, o modelo deixa de chamar `data_atual` e inventa a hora.
2. **Chamada descartada que vira texto.** Na segunda tentativa, o modelo às vezes escreve o JSON da chamada para o
   usuário. Uma alternativa a medir é responder à chamada com um erro de ferramenta, em vez de desfazer a resposta.
3. **Lacunas do portão e do `dias_ate`.** Falta "ano" na lista de palavras, e falta entender dia da semana
   ("até sexta").
4. **Ferramentas do usuário** (`tools` no estilo OpenAI) junto com o servidor local.
5. **Ferramentas no chat do Phi-4-mini.** As fontes divergem sobre o formato de chamada (`<|tool_call|>` ou
   `functools[...]`); medir no próprio modelo antes de ligar.

## Ferramentas novas (plano combinado; cada uma medida com perguntas novas)

1. **`calcular`: reprovada no teste às cegas com o Qwen 0,5B.** Foram 12 de 40 contas certas com ela e 11 de 40 sem
   ela. O modelo escreve a fórmula errada, e o programa calcula exatamente a conta errada. O código está no ramo
   `feat/calcular`. Reavaliar quando o Phi-4-mini tiver ferramentas ([DECISIONS.md](DECISIONS.md), iteração 7).
2. **Ajudante de comandos para bash e PowerShell.** O modelo sugere um comando completo. No Windows o padrão é
   PowerShell; bash onde existir (Linux, macOS, Git Bash, WSL).
   - **Primeiro a hipótese, às cegas.** Antes de construir as conferências e a confirmação, medir num conjunto novo o
     modelo com os 3 exemplos recuperados contra o modelo sem eles, pergunta a pergunta. Sem ganho claro, a ideia cai
     como caiu a calculadora.
   - **Livro de exemplos fora do contexto** (tldr-pages em português, CC BY 4.0, mais receitas de PowerShell). Para
     cada pedido entram só os 3 exemplos mais parecidos, em cerca de 60 tokens, e eles saem do histórico depois.
   - **Primeira palavra restrita** aos comandos dos exemplos.
   - **Conferências antes de mostrar:**
     - marcador que sobrou;
     - arquivos citados que existem ou aparecem no pedido;
     - sintaxe verificada sem executar.

     Se nada passar, a resposta é "não consegui".
   - **Nada roda sem confirmação.** Comandos que apagam, movem ou usam a rede pedem "sim" por extenso e mostram antes
     os arquivos afetados. A função vem desligada por padrão (`--shell`).
   - Medido sem exemplos: o Qwen 0,5B acertou 1 de 9 comandos bash e 1 de 4 PowerShell; com 3 exemplos, 6 de 9. O
     Phi-4-mini acertou de primeira um `Rename-Item` que o Qwen errava.
3. **Calendário estendido:** dia da semana de qualquer data, "daqui a N dias" e a diferença entre datas.
4. **`converter`** (unidades) e **`contar`** (letras e palavras).

## Rumo ao "llamafile": um executável que já é o modelo

Objetivo: **um único arquivo** (`qwen-pt.exe` / `qwen-pt`) que contém runtime, config, tokenizer e pesos. Basta dar
duplo clique ou executar. O caminho é incremental, e cada etapa entrega valor sozinha.

### Etapas 1 e 2: concluídas na v0.2 (iteração 3)

- **Contêiner v3:** config e tokenizer viajam como blobs `__config__` e `__tokenizer__`, então `--model` sozinho
  basta. Leitores v1 e v2 recusam o v3 já no cabeçalho.
- **Executável autocontido:** `smolmoe-bundle --exe` (ou `make qwen-pt`) anexa o contêiner ao runtime com um trailer
  `SMOEBDL1`. Sem `--model`, o runtime se mapeia e roda. Testado no Windows e no Linux (ASan): o hash dos logits é
  idêntico ao da execução com três arquivos, e a partida é igual ou mais rápida.
- **Chat multi-turno**, que é o que torna o duplo clique útil: o cache KV é reaproveitado entre turnos, com
  `/limpar` e `/sair`. No Windows a entrada é lida em UTF-16 e convertida, para os acentos chegarem intactos.
- **Pendentes desta trilha:**
  - Assinatura de código no CI (Authenticode depois do bundle; no macOS, notarização com o payload numa seção Mach-O).
  - Fallback para `<nome>.smoe` ao lado do executável, para modelos acima de 4 GiB no Windows.

### Etapa 3: um binário para todos os sistemas (opcional)

O llamafile usa a [Cosmopolitan libc](https://github.com/jart/cosmopolitan) (formato APE): um mesmo arquivo roda em
Windows, Linux, macOS e BSDs, em x86-64 e ARM64. Como o runtime é C99 puro com a plataforma isolada em dois arquivos, a
porta para o `cosmocc` é viável. As duas peças que exigem cuidado:

- `src/platform.c`: o Cosmopolitan expõe POSIX mesmo no Windows, então o ramo POSIX deve servir quase sem mudança.
- Seleção de kernel em tempo de execução: o binário "fat" x86-64 + ARM64 precisa compilar AVX2 e NEON em fatias
  separadas.

Critério para entrar: a etapa 2 em uso real. Antes disso o Cosmopolitan é complexidade sem demanda comprovada.

### Etapa 4: servidor local compatível com OpenAI

`smolmoe serve --port 8080` com `/v1/chat/completions` (SSE em streaming), uma sessão por conexão, cache KV
reaproveitado entre turnos e multi-turno no template ChatML. É o que torna o executável único útil para outros
programas, como o llamafile faz.

## Capacidade de modelo

Feito: **Phi-4-mini** (iteração 6) e o **conversor em TypeScript** para Qwen2 e Phi-3. Falta portar o do TinyStories
(`tools/import_llama2c.py`).

1. **Gemma 3 (1B e 270M).** Aguarda o aceite dos termos no Hugging Face (`HF_TOKEN`). Exige no motor:
   - ativação GELU;
   - 4 normalizações por camada;
   - normalização de Q/K;
   - atenção local de 512 tokens em 5 de cada 6 camadas;
   - duas frequências de RoPE;
   - tamanho de cabeça desacoplado da largura.

   Exige também um tokenizer SentencePiece de 262 mil peças e o template de chat do Gemma.
2. **Um MoE treinado de verdade.** O candidato mais leve é o Granite 3.1 1B-A400M (IBM, Apache-2.0: 32 experts, 8
   ativos, cerca de 400M parâmetros ativos). Também OLMoE-1B-7B. Exigem roteamento softmax (hoje é só sigmoide),
   fatores de escala e tokenizer próprio. Sem esse item, o "MoE" do nome continua sem prova.
3. **Importador de GGUF** (o formato do llama.cpp). O Q8 de lá é quase o nosso (escala em 16 bits em vez de 32, sem
   perda) e o arquivo traz tokenizer e template. Abre milhares de modelos já publicados, para as arquiteturas que o
   motor já implementa.
4. **Qwen2.5-1.5B e Coder-1.5B.** O conversor só precisa liberar o tamanho, porque a arquitetura é a mesma. Com Q5, o
   1,5B cabe em ~1,05 GB.
5. **Phi-4-mini com contexto acima de 4096**, com os fatores longos do "longrope".

## Operação

- Releases assinadas no CI: Windows (x64), Linux (x64, ARM64) e macOS (universal), com SHA-256 publicado.
- Testes de regressão de qualidade no CI usando o TinyStories (27 MB, cabe no cache do CI).
