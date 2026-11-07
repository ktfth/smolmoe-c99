# Todos os jeitos de rodar o smolmoe

São três interfaces (terminal, janela e navegador) e duas origens (arquivos prontos do Release ou compilados da
fonte). Esta página diz qual escolher e como começar em cada sistema. Os detalhes de cada uma ficam nos guias
indicados.

## Qual escolher

| Você quer... | Caminho | Sistemas |
|---|---|---|
| Clicar e conversar | [Janela com o modelo dentro](#1-janela) | Windows (no Release a partir da próxima versão); Linux e macOS compilando |
| Conversar no terminal, sem instalar nada | [Chat no terminal, arquivo único](#2-chat-no-terminal-arquivo-único) | Windows, Linux, macOS, Termux |
| Usar outro modelo (Phi-4-mini, Q5) | [Programa e modelo separados](#3-programa-e-modelo-separados) | Todos |
| Uma resposta para um script | [Uma resposta e sai](#4-uma-resposta-e-sai-scripts) | Todos |
| Conversar no navegador, sem binário nativo | [Navegador, no seu computador](#5-navegador-no-seu-computador) | Chrome ou Edge recentes (testado no Chrome) |
| Um executável seu, com outro modelo dentro | [Montar seu próprio arquivo único](#6-montar-seu-próprio-arquivo-único) | Todos |
| Mexer no código | [Compilar da fonte](#7-compilar-da-fonte) | Windows (MSYS2), Linux, macOS |

Em todos os caminhos é o mesmo motor de conversa: o mesmo português, as mesmas ferramentas de data e de conta, e os
mesmos números (o navegador dá exatamente as respostas do nativo). Nenhum deles manda texto para fora do computador.

### Qual modelo

| Modelo | Arquivo | Memória | Bom para |
|---|---|---|---|
| **Qwen2.5-0.5B Q8** (padrão) | `qwen2.5-0.5b-instruct-q8.smoe`, 560 MB | ~0,6 GB | Chat em português, com as ferramentas. É o que vem dentro do `qwen-pt`. |
| Qwen2.5-0.5B Q5 | `qwen2.5-0.5b-instruct-q5.smoe`, 340 MB | ~0,4 GB | O mesmo modelo, para quando falta memória. Não é mais rápido. |
| Phi-4-mini (3,8B) | `phi-4-mini-instruct.smoe.001` a `.003` (juntar) | 4,4 GB | Português e contas melhores, a ~4 tokens/s. Ainda sem as ferramentas. Não cabe no navegador. |
| TinyStories 15M | `tinystories-15m.smoe`, 28 MB | pouca | Testar se roda. Escreve historinhas em inglês e não conversa (a janela e o navegador o recusam). |

O repositório é privado: os arquivos do Release baixam com o GitHub CLI autenticado (`gh auth login`). Os comandos
de download de cada sistema estão em [INSTALL.md](INSTALL.md).

## 1. Janela

O chat numa janela, com a resposta chegando aos poucos, botão de parar e as ferramentas. Guia completo em
[DESKTOP.md](DESKTOP.md).

- **Windows:** baixe `qwen-pt-gui-windows-x64.exe` e dê dois cliques. Para abrir outro modelo, use o
  `smolmoe-gui-windows-x64.exe` e escolha o `.smoe` na janela, ou arraste o arquivo para cima dela.
- **Linux:** `sudo apt install g++ make pkg-config libglfw3-dev`, `make gui` e depois `build/smolmoe-gui modelo.smoe`.
- **macOS:** `brew install glfw`, `make gui` e depois `build/smolmoe-gui modelo.smoe`.
- **Termux:** não tem; use o chat no terminal.

A janela entra no Release a partir da primeira versão depois da 0.3.1.

## 2. Chat no terminal, arquivo único

O programa e o Qwen num executável só, como o llamafile: baixe, rode e converse.

| Sistema | Arquivo | Rodar |
|---|---|---|
| Windows | `qwen-pt-windows-x64.exe` | dois cliques, ou `.\qwen-pt-windows-x64.exe` no PowerShell |
| Linux x64 | `qwen-pt-linux-x64` | `chmod +x qwen-pt-linux-x64 && ./qwen-pt-linux-x64` |
| Linux ARM64 e Termux | `qwen-pt-linux-arm64` | o mesmo; no Termux, na pasta `~` (em `/sdcard` o Android não deixa executar) |
| macOS (Intel e Apple Silicon) | `qwen-pt-macos` | `chmod +x qwen-pt-macos && ./qwen-pt-macos` |

Os binários de Linux são estáticos e rodam em qualquer distribuição.

No chat:
- `/limpar` começa outra conversa, e `/sair` encerra;
- uma linha que começa com `!` roda no shell, como em `!dir` ou `!git status` (veja o README).

## 3. Programa e modelo separados

Para trocar de modelo sem baixar outro executável: o programa `smolmoe-<sistema>` com `--model`.

```sh
./smolmoe-linux-x64 --model qwen2.5-0.5b-instruct-q5.smoe          # chat com o Q5
./smolmoe-linux-x64 --model phi-4-mini-instruct.smoe               # Phi-4-mini, depois de juntar as partes
```

No Windows, o comando é `.\smolmoe-windows-x64.exe --model ...`. Juntar as partes do Phi-4-mini está em
[INSTALL.md](INSTALL.md#phi-4-mini-modelo-maior).

Os `.smoe` do Release são autocontidos. Um `model.smoe` sem configuração dentro, como os de `models/<nome>/`
baixados por `bun scripts/fetch-models.ts`, precisa também de `--config config.ini --tokenizer tokenizer.qbp`. Na
janela basta que esses arquivos estejam na mesma pasta.

## 4. Uma resposta e sai (scripts)

```sh
./qwen-pt-linux-x64 --chat --prompt "Quanto é 7 mais 5? Responda apenas com o número."   # escreve 12 e sai
echo "Resuma: ..." | ./qwen-pt-linux-x64                                                # chat lendo do pipe
```

- **A saída:** o texto vai para o stdout, e uma linha JSON com as estatísticas vai para o stderr.
- **No pipe,** o `!comando` fica desligado: um arquivo não roda nada.
- **Para repetir execuções:** `--now AAAA-MM-DDTHH:MM` fixa a hora das ferramentas, e `--temperature 0` (o padrão)
  dá sempre a mesma resposta.

As opções estão no README e em `--help`.

## 5. Navegador, no seu computador

A página `web/` roda o modelo inteiro no navegador, em WebAssembly. Ela fica para uso local por enquanto, sem
publicação na internet. Guia em [WEB.md](WEB.md).

```sh
make ZIG=/caminho/do/zig web     # precisa do zig 0.13 e do bun
bun scripts/serve-web.ts         # http://localhost:8080/
```

Abra o endereço no Chrome ou no Edge e use "Abrir arquivo .smoe" com o `qwen2.5-0.5b-instruct-q8.smoe`.
- **Por que o servidor próprio:** abrir o `index.html` direto do disco não funciona. A página precisa dos cabeçalhos
  COOP/COEP que o `serve-web.ts` envia; sem eles, não há threads.
- **Velocidade:** ~11 tokens/s no Chrome, contra ~27 no nativo.
- **Abertura:** a primeira mensagem demora uns 20 s a mais, porque as instruções do modelo são processadas a cada
  abertura.

## 6. Montar seu próprio arquivo único

O `smolmoe-bundle` (no `smolmoe-tools-<sistema>.tar.gz`) junta um programa e um modelo num executável:

```sh
smolmoe-bundle --model modelo.smoe --exe smolmoe-linux-x64 --out meu-chat                 # chat no terminal
smolmoe-bundle --model modelo.smoe --exe smolmoe-gui-windows-x64.exe --out meu-chat.exe   # janela (Windows)
```

Um `.smoe` sem configuração dentro também leva `--config` e `--tokenizer`. O formato está em [FORMAT.md](FORMAT.md).

## 7. Compilar da fonte

| Sistema | Ferramentas |
|---|---|
| Windows | MSYS2 UCRT64 com `make` e `mingw-w64-ucrt-x86_64-gcc` |
| Linux | `gcc` ou `clang`, `make`; para a janela, também `g++`, `pkg-config` e `libglfw3-dev` |
| macOS | Xcode Command Line Tools; para a janela, `brew install glfw` |

Em todos: o [Bun](https://bun.sh) para baixar os modelos e rodar os testes, e o zig 0.13 só para o WebAssembly.

```sh
bun scripts/fetch-models.ts      # pesos em models/, com SHA-256 conferido
make                             # build/smolmoe, smolmoe-bundle, smolmoe-quantize
make gui                         # build/smolmoe-gui
make qwen-pt                     # dist/qwen-pt: o arquivo único do Qwen
make ZIG=zig wasi web            # WebAssembly: módulos WASI para os testes e a página
make test                        # testes em C e de integração (bun test)
```

Para rodar o que foi compilado:

```sh
build/smolmoe --model models/qwen2.5-0.5b-instruct/model.smoe \
  --config models/qwen2.5-0.5b-instruct/config.ini --tokenizer models/qwen2.5-0.5b-instruct/tokenizer.qbp
build/smolmoe-gui models/qwen2.5-0.5b-instruct/model.smoe
```

O build WASI (`build/wasi/*.wasm`) existe para os testes e roda no bun, pelo host de `tests/support/`. Para conversar
no navegador, use a página (caminho 5).

## Matriz: o que roda onde

| | Windows | Linux x64 | Linux ARM64 | macOS | Termux |
|---|---|---|---|---|---|
| Chat no terminal (Release) | sim | sim | sim | sim | sim |
| Janela | Release | `make gui` (CI) | `make gui` (não testado) | `make gui` (CI) | não |
| Navegador local | testado no Chrome | não testado | não testado | não testado | não |
| `!comando` no chat | sim | sim | sim | sim | sim |

- **Release:** o arquivo vem pronto.
- **`make gui` (CI):** o CI compila a janela a cada mudança, mas ela ainda não foi aberta à mão nesse sistema.
- **Phi-4-mini:** precisa de 4,4 GB de memória livres. Foi validado no Windows. Nos outros sistemas o código é o
  mesmo, mas o modelo não foi conferido.
