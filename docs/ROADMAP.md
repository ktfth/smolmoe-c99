# Roadmap

Ordem definida por impacto no objetivo (eficiência em produção) dividido pelo risco. Cada item só fecha com medição
antes/depois registrada em [DECISIONS.md](DECISIONS.md).

## Próximo: eficiência

1. **Matemática determinística própria** (`expf`, `sin`/`cos`). Hoje os hashes de logits diferem entre Windows e Linux
   só por causa da libm. Polinômios próprios em float puro (sem FMA) dão bit-identidade entre sistemas operacionais,
   o que permite fixar hashes-ouro na suíte. Também permite vetorizar SiLU e softmax.
2. **Kernel Q5 mais barato.** O Q5 já existe e economiza 39% de memória por +1,7% de perplexidade, mas custa 2,6× o Q8
   por bloco e não acelera neste notebook (ver DECISIONS, iteração 2). Opções: dois blocos por iteração
   compartilhando a expansão dos bits altos, ou um layout com os bits altos já intercalados para SIMD.
3. **Q4 com escalas por sub-bloco (estilo K-quant)**, só junto com modelos de 1,5B+. O Q4 simples custou +21% de
   perplexidade no 0,5B e foi removido.
4. **Kernels NEON (ARM64).** Mesma ordem de lanes. Abre Apple Silicon, Raspberry Pi e servidores Graviton.
5. **Cache KV em F16.** Metade da memória e da banda em contextos longos.

## Rumo ao "llamafile": um executável que já é o modelo

Objetivo: **um único arquivo** (`qwen-pt.exe` / `qwen-pt`) que contém runtime, config, tokenizer e pesos. Basta dar
duplo clique ou executar. O caminho é incremental, e cada etapa entrega valor sozinha.

### Etapa 1: contêiner único `SMOE v3`

Hoje são três arquivos (`model.smoe`, `config.ini`, `tokenizer.*`). A v3 do contêiner guarda a config e o tokenizer
como blobs nomeados (`__config__`, `__tokenizer__`) no mesmo diretório ordenado que já existe. Assim, `--model` sozinho
basta. O parser atual já valida diretório e extents, então a mudança é pequena e retrocompatível (v1/v2 continuam
lendo).

### Etapa 2: pesos embutidos no executável (Windows, Linux e macOS)

- `smolmoe bundle --model m.smoe --out qwen-pt.exe` copia o próprio executável e anexa ao final o contêiner v3,
  alinhado em 64 KiB, mais um trailer fixo: `magic "SMOEBNDL" | offset u64 | size u64 | sha256`.
- Na partida, o runtime abre o próprio executável (`self_executable_path()` já existe em `src/platform.c`), lê o
  trailer, mapeia o arquivo e passa `base + offset` ao `modelfile_parse`, que já recebe um ponteiro e um tamanho. Nada
  muda no modelo.
- Assinatura de código: no Windows, o Authenticode cobre o executável inteiro, então é preciso assinar **depois** de
  anexar. No macOS, os dados anexados invalidam a assinatura e exigem notarização com o payload em
  `__DATA,__smoe` (seção Mach-O). Ficam documentados como passos de release.
- Limite: o Windows não executa PE acima de 4 GiB. Modelos maiores continuam com pesos externos, e o executável
  procura `<nome>.smoe` ao lado de si.

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

1. **Um MoE treinado de verdade.** Candidatos pequenos: OLMoE-1B-7B (64 experts, top-8, softmax) e Qwen1.5-MoE-A2.7B.
   Isso exige roteamento softmax (hoje é só sigmoide), qk-norm (OLMoE) e um importador. Sem esse item, o "MoE" do nome
   continua sem prova.
2. **Conversores em TypeScript/Bun.** Substituem os scripts Python e o `q8pack.c` (só POSIX). Leitura de safetensors
   em TS é direta (cabeçalho JSON e bytes crus).
3. **Qwen2.5-1.5B/3B.** Maior qualidade em português, ao custo de 3× a 6× a memória. Com Q5, o 1,5B cabe em ~1,05 GB.

## Operação

- Releases assinadas no CI: Windows (x64), Linux (x64, ARM64) e macOS (universal), com SHA-256 publicado.
- Testes de regressão de qualidade no CI usando o TinyStories (27 MB, cabe no cache do CI).
