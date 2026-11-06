# Geração local em português, somente CPU

Esta versão inclui **Qwen2.5-0.5B-Instruct convertido para Q8**, seu tokenizer e
o aplicativo C99. A inferência acontece no computador, sem GPU, serviço remoto
ou framework de aprendizado de máquina. Os pesos estão no ZIP.

## Executar

No Linux, com compilador C e make:

```sh
unzip smolmoe-c99.zip
cd smolmoe
make portugues-demo
```

Para usar seu próprio prompt:

```sh
./smolmoe --model examples/qwen-portugues/model.smoe \
  --config examples/qwen-portugues/config.ini \
  --tokenizer examples/qwen-portugues/tokenizer.qbp \
  --chat --prompt 'Quanto é 7 mais 5? Responda apenas com o número.' \
  --tokens 32 --context 512 --memory-mib 256
```

`--chat` monta o ChatML com esta mensagem de sistema:

> Responda somente em português do Brasil. Não misture outros idiomas. Seja breve e direto.

Use `--system 'Sua instrução'` para substituí-la. É uma conversa de um turno por
processo, sem histórico interativo. Texto vai para stdout; estatísticas vão para
stderr. `--stats arquivo.json` salva as estatísticas.

## Saídas realmente executadas

| Prompt | Resposta do modelo, sem reescrita |
|---|---|
| Escreva uma saudação curta para alguém que está aprendendo programação. | "Olá, aprendiz! Estou aqui para te ajudar com suas tarefas no código." |
| Quanto é 7 mais 5? Responda apenas com o número. | 12 |
| Escreva duas frases sobre um robô que cuida de uma horta. | Um robô cuida de uma horta através de tecnologia, monitorando o clima, a fertilidade e a saúde do plantio. |

O último exemplo recebeu uma frase, apesar de pedir duas. Em uma execução com
outra mensagem de sistema, o modelo misturou a palavra chinesa `指导` em uma
definição; a saída foi preservada em `examples/qwen-portugues/primeira-resposta.txt`.
O prompt de idioma ajuda, mas não garante qualidade, precisão ou cumprimento
completo das instruções. O modelo é pequeno; revise suas respostas.

Prompts, saídas e medidas estão em `docs/resultados-portugues.json`. São textos
gerados pelos pesos treinados; não há traduções prontas ou tabela de respostas.

## Memória e desempenho

O Q8 convertido ocupa aproximadamente **531 MiB em disco**. O ZIP cresceu por
incluir os pesos. O app mantém um cache limitado e lê o restante por blocos.

Na validação da soma, com contexto 256 e limite de alocações de 128 MiB:

- Pico de alocações controladas: cerca de 127,83 MiB.
- Pico de RSS do processo: cerca de 123,26 MiB.
- Geração observada: cerca de 6,6 tokens/s em AVX2.
- Preparação do prompt: cerca de 5,8 s, nesse teste com hash de todos os logits.

Medido em Linux x86-64, AMD EPYC 9V45 e GCC 13.3.0, com possível cache de
arquivos aquecido. O host tem mais de 8 GB; não foi uma máquina física de 8 GB.
CPU, SSD e cache afetam o resultado. RSS, alocações e cache de arquivos são
medidas diferentes. O limite do app não inclui overhead de libc/stack/allocator
nem cache do Linux; deixe folga para o sistema. A primeira resposta leva segundos.

## Validação

```sh
make test
make portugues-test
ASAN_OPTIONS=detect_leaks=0 make sanitize
make clean
make
```

- **35 testes** passaram, incluindo atenção com vieses, compartilhamento de
  pesos, conversão BF16/Q8, Unicode e tokenizer.
- Passaram também com AddressSanitizer e UBSan. Leaks não foram verificados:
  o ambiente bloqueia a inspeção de threads exigida por LeakSanitizer.
- O tokenizer C coincidiu com `tokenizers` 0.21.4 em 11 casos, incluindo acentos
  decompostos e ChatML. A biblioteca foi apenas referência de desenvolvimento;
  não é necessária para executar o app, converter pesos ou rodar os testes.
- A resposta `12` e o checksum FNV-1a64 de todos os logits coincidiram com 128,
  256 e 512 MiB, em AVX2 e escalar. O checksum é uma verificação de regressão,
  não uma prova criptográfica para qualquer modelo ou prompt.

Evidências: `docs/qwen-memory-validation.json`, `docs/qwen-tokenizer-validation.json`
e `docs/validation-logs/`.

## Refazer a conversão

Os pesos estão incluídos. Para baixar os originais e converter novamente:

```sh
python3 tools/import_qwen2.py downloads/qwen2.5-0.5b \
  examples/qwen-portugues --fetch
```

O checkpoint oficial BF16 tem cerca de 988 MB e seu SHA-256 é verificado. O
Python lê metadados com a biblioteca padrão; `tools/q8pack.c` converte linha a
linha. Não carrega pickle nem usa PyTorch, NumPy ou Transformers. A conversão
Q8 é com perda; embedding e projeção de saída compartilham os mesmos dados.
Q/K e seus vieses são permutados para o RoPE de pares adjacentes do motor.

## Escopo e licenças

O modelo é Qwen2.5-0.5B-Instruct, não Kimi K3. É denso, representado como um
expert sempre ativo. Isso não valida um MoE treinado. KDA, MLA, MXFP4 e checkpoints
completos Kimi K3 continuam sem suporte. O importador não aceita qualquer modelo HF.

O código é MIT. Pesos/tokenizer Qwen são Apache-2.0, com licença e proveniência
em `examples/qwen-portugues/`. Dados Unicode têm licença em
`licenses/Unicode-LICENSE.txt`. TinyStories continua incluído (`make pretrained-demo`).
