# Changelog

## Não lançado

### Adicionado
- **Chamada de funções no chat** (template do Qwen2.5). Há duas ferramentas embutidas: `data_atual` e `dias_ate`, que
  aceita data em texto e feriados brasileiros, inclusive os móveis. O modelo escolhe a ferramenta, e as datas são
  calculadas em C.
  - As ferramentas só são oferecidas quando a mensagem fala de data ou de tempo.
  - A data escrita pelo usuário vale mais que o argumento do modelo.
  - Uma chamada que não combina com a mensagem é descartada.
  - Medição: 12 de 15 certas em perguntas novas, que não foram usadas no desenho. Os limites conhecidos estão em
    [DECISIONS.md](docs/DECISIONS.md), iteração 4.
- `--no-tools` desliga as ferramentas. `--now AAAA-MM-DDTHH:MM` fixa o relógio, para execuções reproduzíveis.
- Campo `tool_calls` nas estatísticas do chat.
- `scripts/eval-tools.ts`: mede a escolha de ferramenta e a resposta.
  - Há três conjuntos: o de desenvolvimento e dois de perguntas novas.
  - `--session` mede a conversa longa.
- `scripts/fetch-models.ts` (`make models`): baixa os pesos do Release `models-v1` e confere o SHA-256.

### Alterado
- Com as ferramentas ligadas, o chat leva cerca de 5 s a mais para abrir (o prompt de sistema vai de 31 para 279
  tokens). `--no-tools` volta ao comportamento anterior.

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
