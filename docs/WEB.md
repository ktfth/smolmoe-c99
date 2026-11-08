# smolmoe no navegador

A página `web/` roda o chat inteiro no navegador de quem a abre, em WebAssembly. O servidor só entrega arquivos
estáticos (a página, o `.wasm` de 240 KB e os pesos), e nenhuma mensagem sai do computador. É o mesmo código C do
binário nativo e o mesmo motor de conversa (`src/engine.c`), com as cinco ferramentas locais: `data_atual`, `dias_ate`,
`calcular`, `converter` e `contar`.

**Por enquanto, a página é para rodar no seu computador** (decisão de 2026-11-07): `make run-web` compila e inicia o
servidor local; o Qwen local fica disponível para abrir na página. Outros modelos podem ser abertos como `.smoe`
autocontidos. A seção sobre publicar fica para quando houver publicação. Os outros jeitos de rodar estão em
[CAMINHOS.md](CAMINHOS.md).

## Usar

1. Abra a página no Chrome ou no Edge recentes (testada no Chrome). O Firefox e o Safari também têm WebAssembly SIMD
   e threads, mas ainda não foram testados.
2. Escolha uma das opções:
   - **Abrir arquivo .smoe:** use o `qwen2.5-0.5b-instruct-q8.smoe` (560 MB) da página de releases.
   - **Baixar e abrir:** aparece quando o `models.json` servido tem uma URL para os pesos. O servidor de `make run-web`
     aponta para o Qwen local; numa hospedagem, configure `web/models.json`. Depois que o modelo
     abre, uma cópia vai para o cache do navegador, e só a primeira visita baixa. Uma cópia de tamanho errado, ou
     que o motor recusa, nunca fica no cache. A chave leva o SHA-256 do modelo, então uma versão nova é baixada de
     novo.
   - **Gemma 3 1B:** o conversor gera por padrão um diretório com `model.smoe`, config e tokenizer separados. Para abrir
     no navegador, gere um `.smoe` autocontido v3:

     ```sh
     bun scripts/convert.ts gemma3 downloads/gemma-3-1b-it models/gemma-3-1b-it --container v3
     ```

     Escolha `models/gemma-3-1b-it/model.smoe` em "Abrir arquivo .smoe". O checkpoint precisa estar baixado e os
     termos aceitos; veja [MODELS.md](MODELS.md).
3. A primeira mensagem demora mais porque o prompt do sistema, com as declarações das ferramentas, é processado a cada
   abertura.

O `!comando` não existe na página: o navegador não tem shell.

## Construir e testar localmente

```sh
make run-web                         # compila e inicia a página em localhost:8080
make ZIG=/caminho/do/zig web        # build/web: index.html, app.js, workers, smolmoe.wasm, vercel.json
bun scripts/serve-web.ts            # http://localhost:8080/, com os cabeçalhos COOP/COEP
```

O build precisa do zig 0.13, que traz a libc do WASI, e do bun. Os testes `tests/wasm.test.ts` e `tests/web.test.ts`
rodam com `make test` quando `build/wasi` e `build/web` existem. No CI, eles são obrigatórios (`SM_EXPECT_WASM=1`).

## Publicar (adiado: Vercel ou qualquer hospedagem estática)

Nada disto está em uso: a página não foi publicada. O que segue é o que falta para publicar quando for decidido.

- Publique a pasta `build/web`. O `vercel.json` dela já envia os cabeçalhos `Cross-Origin-Opener-Policy: same-origin`
  e `Cross-Origin-Embedder-Policy: require-corp`. Sem eles, o navegador não libera memória compartilhada, e a página
  avisa disso.
- **Pesos:** como o repositório é privado, eles precisam de outro endereço público, por exemplo o Hugging Face ou o
  Vercel Blob. Ponha a URL em `web/models.json` e rode `make web`. As licenças permitem redistribuir: Apache-2.0 no
  Qwen, MIT no TinyStories.
  - Requisitos do endereço: CORS, porque a página usa COEP, e o tamanho exato que está no `models.json`.
  - Ainda não testado: um endereço em outra origem. O download foi testado só com o servidor local, na mesma origem.
- Não ponha o `.smoe` dentro de `build/web`: a página não precisa dele ali, e a hospedagem pode recusar arquivos
  desse tamanho.

## Como funciona

| Peça | Arquivo | Papel |
|---|---|---|
| Motor | `src/engine.c` | Uma conversa: ferramentas, rollback de chamadas, limites de contexto. É o mesmo do terminal. |
| Entrada do navegador | `src/web.c` | Funções exportadas (`sm_open`, `sm_send`, `sm_reset`, `sm_stats`) e saída por funções importadas. |
| Host | `web/wasm-host.ts` | Memória compartilhada, threads, as poucas chamadas WASI do módulo, texto e avisos. |
| Worker do motor | `web/engine-worker.ts` | Põe os pesos direto na memória do módulo, abre o modelo e responde à página. |
| Origem dos pesos | `web/model-source.ts` | Cache ou download; o cache só recebe um modelo que abriu. |
| Threads | `web/pool-worker.ts` | Uma instância do módulo por thread, na mesma memória, cada uma com a sua pilha. |
| Página | `web/app.ts`, `web/index.html` | Só a interface. O texto do modelo entra sempre como texto, nunca como HTML. |

- **Threads sem Emscripten.** O plano previa o Emscripten, mas instalá-lo nesta máquina foi bloqueado. O zig já estava
  instalado e compila para WASI, só que sem pthreads. O pool não precisa deles:
  - o host cria as instâncias de trabalho antes do `main`;
  - cada uma ocupa uma vaga do pool e dorme em `memory.atomic.wait` até haver trabalho.
  - A vantagem: o módulo testado no CI é o mesmo que o navegador roda, compilado pelo mesmo compilador.
- **Kernels SIMD128.** Seguem a estrutura dos kernels NEON e a ordem de lanes do código escalar. O "relaxed SIMD" fica
  de fora, porque os resultados dele variam de máquina para máquina.
- **Determinismo.** Os testes provam que o WebAssembly dá exatamente os números do binário nativo:
  - o hash dos logits do TinyStories e do Qwen (Q8 e Q5), com 1, 2 e 4 threads, é o mesmo do nativo;
  - um chat com ferramenta sai idêntico ao do nativo, byte a byte.

## Velocidade (notebook i7-10610U)

| Qwen 0.5B Q8 | Escrita | Leitura do prompt |
|---|---|---|
| Nativo, AVX2, 4 threads | 27 tok/s | 60 tok/s |
| Chrome, SIMD128, 4 threads | 11 tok/s | ~18 tok/s |
| bun, SIMD128, 1 thread | 6,8 tok/s | 8,7 tok/s |
| bun, escalar, 1 thread | 1,6 tok/s | 1,7 tok/s |

No WebAssembly, o Q8 escreve 1,6× mais rápido que o Q5 (6,3 contra 4,0 tok/s numa thread). Por isso a página oferece o
Q8, mesmo sendo 220 MB maior. A diferença de tamanho só pesa no primeiro download.

## Limites atuais

- **Memória:** o módulo vai até 2 GiB. O zig 0.13 lê `--max-memory` em hexadecimal, e é esse o valor no Makefile.
  Cabem o Qwen e o TinyStories, o Phi-4-mini (4,4 GB) não.
- **Abertura:** o prompt do sistema é processado de novo a cada carga. Guardar esse KV cache (~10 MB) no navegador
  eliminaria os ~20 s.
- **Parar uma resposta:** não há como interromper uma resposta no meio; ela vai até o fim ou até 512 tokens.
- **Testes manuais:** download pelo servidor local, leitura do cache, ferramentas e interface foram testados à mão
  no Chrome. Ainda faltam o Firefox, o Safari e a largura de celular, conferida só pelo CSS.
- **Thread travada:** se uma thread do pool falhar depois de começar, o motor fica esperando por ela, e é preciso
  recarregar a página.
