# Changelog

## Não lançado

## 0.3.0 — 2026-11-07

### Adicionado
- **Release com binários prontos** para Windows x64, Linux x64, Linux ARM64 (também Termux, no Android) e macOS
  (Intel e Apple Silicon num binário só). Os binários de Linux são estáticos (musl).
  - Executáveis de arquivo único para o Qwen (chat em português) e o TinyStories.
  - Modelos num arquivo só: Qwen Q8 e Q5, TinyStories e Phi-4-mini. O Phi vai em partes, por causa do limite de
    2 GB do GitHub, e é convertido no CI e conferido byte a byte contra o `provenance.json`.
  - Cada binário passa por um teste rápido no próprio sistema antes de publicar (no ARM64, emulado), e o Release
    traz `SHA256SUMS`. Instruções em [docs/INSTALL.md](docs/INSTALL.md).
- **`!comando` no chat:** a linha roda no shell que abriu o smolmoe (PowerShell, pwsh, cmd, Git Bash, bash, zsh) e a
  saída vai direto para o terminal, sem passar pelo modelo.
  - `!cd` muda a pasta do chat, e Ctrl+C interrompe só o comando.
  - `--shell` escolhe o shell; `--no-shell-escape` desliga a função.
  - Com entrada por pipe, a função fica desligada, a menos que se use `--shell-escape`.
  - Texto colado com várias linhas vira uma mensagem só, e nenhuma linha dele roda como comando.
- Texto colado com várias linhas agora vai ao modelo como **uma** mensagem; antes, cada linha virava uma pergunta.
- **Chamada de funções no chat** (template do Qwen2.5). Há duas ferramentas embutidas: `data_atual` e `dias_ate`, que
  aceita data em texto e feriados brasileiros, inclusive os móveis. O modelo escolhe a ferramenta, e as datas são
  calculadas em C.
  - As ferramentas só são oferecidas quando a mensagem fala de data ou de tempo.
  - A data escrita pelo usuário vale mais que o argumento do modelo.
  - Uma chamada que não combina com a mensagem é descartada.
  - Medição: 12 de 15 certas em perguntas novas, que não foram usadas no desenho. Os limites conhecidos estão em
    [DECISIONS.md](docs/DECISIONS.md), iteração 4.
- **Ferramenta `calcular`:** o modelo escreve a conta e o programa a calcula, com símbolos ou palavras ("15% de
  230", "dois mil e quinhentos") e números no formato brasileiro.
  - Todo número da conta precisa vir da mensagem.
  - Quando a mensagem é só uma conta e o modelo escolhe outra ferramenta, o runtime abre a chamada de `calcular`.
  - **Medição às cegas:** 12 de 40 contas certas com ela e 11 de 40 sem ela. Ela também faz errar contas que o modelo
    acertava sozinho, quando ele monta a fórmula errada. Entrou por decisão do usuário; detalhes em
    [DECISIONS.md](docs/DECISIONS.md), iteração 7.
- `scripts/eval-tools.ts`: conjuntos da calculadora (`--set calcdev` e `--set calc`, às cegas), linha de base sem
  ferramentas (`--no-tools`), intervalo de confiança de 95% e leitura de números só no formato brasileiro.
- `--no-tools` desliga as ferramentas. `--now AAAA-MM-DDTHH:MM` fixa o relógio, para execuções reproduzíveis.
- Campo `tool_calls` nas estatísticas do chat.
- `scripts/eval-tools.ts`: mede a escolha de ferramenta e a resposta.
  - Há três conjuntos: o de desenvolvimento e dois de perguntas novas.
  - `--session` mede a conversa longa.
- `scripts/fetch-models.ts` (`make models`): baixa os pesos do Release `models-v1` e confere o SHA-256.
- `--profile`: tempo de cada passada dividido por fase, nas estatísticas.
- `scripts/bench.ts` refeito como esteira de desempenho:
  - cenários fixos, medianas e tempo por fase;
  - `--save` e `--compare` contra uma base, e `--ab` para comparar dois binários alternados na mesma sessão;
  - sai com erro se alguma métrica piorar.

- **Phi-4-mini-instruct (Microsoft, 3,8B)**, o primeiro modelo fora da família Qwen: 17% melhor em português (bits
  por byte) e certo em contas e comandos em que o Qwen 0,5B errava; cerca de 4 tokens/s neste notebook. Detalhes
  e validação em [DECISIONS.md](docs/DECISIONS.md), iteração 6.
  - Chaves opcionais `rotary_dim` e `rope_attention_factor` na config (RoPE parcial e escalado).
  - Tokenizer `o200k` (o do GPT-4o), `.qbp` v2 e template de chat do Phi; o v1 do Qwen segue idêntico.
- `scripts/convert.ts`: conversor em TypeScript (Qwen2 e Phi-3), no lugar do Python. Reproduz os arquivos do Qwen
  byte a byte. `fetch-qwen2` e `fetch-phi3` baixam os originais fixados e conferem o SHA-256.
- `scripts/ref/`: implementação de referência do Phi-4-mini em TypeScript (double), para validar o runtime.
- `scripts/gen-unicode.ts`: gerador das tabelas Unicode em TypeScript (classes no Unicode 16.0, NFC no 15.0).

### Alterado
- O limite de memória padrão passou a ser 512 MiB mais o cache de atenção de até 2048 posições, então modelos
  maiores abrem sem `--memory-mib`. Com a opção, o limite é exatamente o pedido.
- **Leitura do prompt de 21% a 39% mais rápida**, e primeiro token de 17% a 26% mais cedo, em duas medições A/B.
  Quantização, ativação e normalização passaram a usar AVX2 e todas as threads; antes eram 23% do tempo numa thread só.
  Detalhes em [DECISIONS.md](docs/DECISIONS.md), iteração 5.
- Exponencial própria (corretamente arredondada em mais de 99,99% das entradas), idêntica no escalar e no AVX2 e em
  qualquer sistema operacional. Os números mudam no último bit em casos raros. A perplexidade foi de 11,3885 para
  11,3681.
- Com as ferramentas ligadas, o chat leva cerca de 5,5 s a mais para abrir: o prompt de sistema vai de 31 para 397
  tokens (279 com as de data e mais 118 da `calcular`). `--no-tools` volta ao comportamento anterior.
- `.env` e `.env.*` ficam fora do Git (o token do Hugging Face mora ali).

## 0.2.0 — 2026-11-06

Reescrita do runtime para eficiência e para uso em produção. Números e justificativas em
[docs/DECISIONS.md](docs/DECISIONS.md).

### Adicionado
- **Executável único estilo llamafile**: `smolmoe-bundle` (contêiner v3 com config e tokenizer embutidos; `--exe` anexa ao
  runtime) e `make qwen-pt`. Sem `--model`, o runtime usa o modelo anexado a si mesmo.
- **Chat multi-turno** (`-i`, padrão sem `--prompt`) com cache KV reaproveitado, `/limpar` e `/sair`; entrada UTF-8 correta
  no console do Windows.
- Mensagens de erro do Windows em UTF-8 (antes saíam na codepage local).
- Formato Q5 (dtype 2, 22 bytes por 32 pesos) e a ferramenta `smolmoe-quantize`: 39% menos memória por +1,7% de
  perplexidade. Opcional; não acelera no hardware medido.
- `--perplexity`: NLL e perplexidade do prompt com teacher forcing; texto de avaliação em `tests/goldens/eval-pt.txt`.
- Kernels AVX2 agora exigem F16C (presente em todo CPU com AVX2 do mercado).
- Suporte nativo a Windows (MinGW-w64 GCC/Clang): argv em UTF-8, console em UTF-8 e build estático em `make release`.
- Pool de threads determinístico (`--threads`, padrão: núcleos físicos).
- Prefill em lote (`--batch`, padrão 32) com kernel AVX2 de 4 tokens por linha de pesos.
- Amostragem com `--top-k` (padrão 40) e `--top-p` (padrão 0,95).
- `--version`, e os campos `version`, `threads`, `batch` e `mapped_bytes` nas estatísticas.
- Suíte de testes: unitários em C (kernels, f16, pool) e integração em TypeScript/Bun (55 testes).
- Scripts `bench.ts`, `compare-traces.ts` e `make-fixture.ts`. CI para Linux, Windows e macOS.

### Alterado (incompatível)
- Os pesos agora são mapeados em memória. `--memory-mib` limita só o heap (KV, ativações, tokenizer); antes incluía um
  cache de pesos.
- As ativações são quantizadas em Q8 antes de cada produto matricial. Os logits diferem da v0.1 em +0,05% de
  perplexidade, e hashes antigos de `--hash-logits` não valem mais.
- O limite do prompt passou de `context-1` bytes para `4 × context` bytes; a checagem real é sobre os tokens.
- Os arquivos de saída (`--trace`, `--token-ids`, `--stats`) são escritos em modo binário (LF em todas as plataformas).
- Os modelos foram movidos de `examples/` para `models/` e saíram do controle de versão.

### Removido
- `--drop-cache` e os campos `cache_hits`, `cache_misses`, `read_bytes`, `expert_read_bytes` e `cache_bytes`, que só
  faziam sentido com o cache de tiles.
- O binário Linux pré-compilado, as evidências de validação da v0.1 e os testes em Python.

## 0.1.0 — 2026-11-05

Versão inicial: runtime de uma thread com cache de tiles sobre `pread`, só Linux.
