# Changelog

## 0.2.0 — 2026-11-06

Reescrita do runtime para eficiência e para uso em produção. Números e justificativas em
[docs/DECISIONS.md](docs/DECISIONS.md).

### Adicionado
- Suporte nativo a Windows (MinGW-w64 GCC/Clang): argv em UTF-8, console em UTF-8 e build estático em `make release`.
- Pool de threads determinístico (`--threads`, padrão: núcleos físicos).
- Prefill em lote (`--batch`, padrão 32) com kernel AVX2 de 4 tokens por linha de pesos.
- Amostragem com `--top-k` (padrão 40) e `--top-p` (padrão 0,95).
- `--version`, e os campos `version`, `threads`, `batch` e `mapped_bytes` nas estatísticas.
- Suíte de testes: unitários em C (kernels e pool) e integração em TypeScript/Bun (35 testes).
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
