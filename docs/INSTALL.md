## Como baixar e rodar

O repositório é privado, então os arquivos só baixam com acesso a ele. O jeito mais simples é o GitHub CLI já
autenticado (`gh auth login`). Sem tag, `gh release download` baixa a versão mais recente.

### Qual arquivo baixar

| Arquivo | Para quê |
|---|---|
| `qwen-pt-gui-windows-x64.exe` | **Chat numa janela**, num arquivo só: a janela e o Qwen2.5-0.5B. Dois cliques e pronto. |
| `smolmoe-gui-windows-x64.exe` | Só a janela, para abrir um arquivo `.smoe` (por exemplo, o Q5 ou o Phi-4-mini) |
| `qwen-pt-windows-x64.exe`, `qwen-pt-linux-x64`, `qwen-pt-linux-arm64`, `qwen-pt-macos` | Chat em português no terminal, num arquivo só: o programa e o Qwen2.5-0.5B (560 MB) |
| `tinystories-<sistema>` | Historinhas em inglês (28 MB), bom para testar se roda |
| `smolmoe-<sistema>` | Só o programa, para usar com um arquivo de modelo (`--model`) |
| `qwen2.5-0.5b-instruct-q8.smoe`, `qwen2.5-0.5b-instruct-q5.smoe`, `tinystories-15m.smoe` | Modelos num arquivo só (configuração e tokenizer dentro) |
| `phi-4-mini-instruct.smoe.001` a `.003` | Phi-4-mini (3,8B) em partes, porque o GitHub aceita até 2 GB por arquivo |
| `smolmoe-tools-<sistema>.tar.gz` | `smolmoe-bundle` (cria executáveis de arquivo único) e `smolmoe-quantize` |
| `SHA256SUMS` | Somas SHA-256 de tudo, inclusive do Phi-4-mini já juntado |

`linux-arm64` serve para Linux ARM e para o **Termux** (Android 64 bits). O `macos` roda em Mac Intel e Mac M.
No Linux e no macOS, a janela ainda não vem pronta: ela é compilada com `make gui`. O guia `docs/CAMINHOS.md`, no
repositório, compara todos os jeitos de rodar.

O Gemma 3 não vem nos arquivos do Release: baixe e converta os pesos localmente seguindo [MODELS.md](MODELS.md).
O acesso ao checkpoint do Hugging Face exige aceitar os termos do Gemma.

### Janela (Windows)

```powershell
gh release download --pattern qwen-pt-gui-windows-x64.exe
.\qwen-pt-gui-windows-x64.exe
```

- **Teclas:** Enter envia, Ctrl+Enter quebra a linha, Esc interrompe a resposta e Ctrl+N começa outra conversa.
- **Outro modelo:** com o `smolmoe-gui-windows-x64.exe`, escolha o `.smoe` no botão "Abrir modelo (.smoe)…" ou arraste o
  arquivo para a janela.
- **SmartScreen:** como todos os binários daqui, a janela não tem assinatura digital. O aviso da primeira vez se
  resolve como no terminal, logo abaixo.

### Windows (terminal)

```powershell
gh release download --pattern qwen-pt-windows-x64.exe
.\qwen-pt-windows-x64.exe
```

O binário não tem assinatura digital, então o SmartScreen pode avisar na primeira vez: "Mais informações" e
"Executar assim mesmo".

### Linux e macOS

```sh
gh release download --pattern qwen-pt-linux-x64     # ou qwen-pt-macos
chmod +x qwen-pt-linux-x64
./qwen-pt-linux-x64
```

No macOS, se o arquivo veio pelo navegador, tire a quarentena antes: `xattr -d com.apple.quarantine qwen-pt-macos`.

### Termux (Android)

```sh
pkg install gh && gh auth login
cd ~                                   # na pasta do Termux: em /sdcard o Android não deixa executar
gh release download --pattern qwen-pt-linux-arm64
chmod +x qwen-pt-linux-arm64
./qwen-pt-linux-arm64
```

O Qwen usa cerca de 0,6 GB de memória. Desde a v0.3.1, o ARM usa os kernels NEON: no Mac do CI, eles escrevem 3,7×
mais rápido que o código sem eles. Num celular a velocidade ainda não foi medida.

### Phi-4-mini (modelo maior)

Ele é melhor em português e em contas, mas usa 4,4 GB de memória e escreve cerca de 4 tokens por segundo num
notebook. Baixe as partes e o programa do seu sistema, junte as partes e rode:

```sh
gh release download --pattern 'phi-4-mini-instruct.smoe.*' --pattern smolmoe-linux-x64
cat phi-4-mini-instruct.smoe.0* > phi-4-mini-instruct.smoe          # Linux, macOS, Termux
chmod +x smolmoe-linux-x64
./smolmoe-linux-x64 --model phi-4-mini-instruct.smoe
```

No Windows, junte as partes no `cmd`:

```bat
copy /b phi-4-mini-instruct.smoe.001 + phi-4-mini-instruct.smoe.002 + phi-4-mini-instruct.smoe.003 phi-4-mini-instruct.smoe
smolmoe-windows-x64.exe --model phi-4-mini-instruct.smoe
```

Na janela, a última linha vira `smolmoe-gui-windows-x64.exe phi-4-mini-instruct.smoe`. A janela usa contexto 4096, o
dobro do terminal, e por isso o Phi pede cerca de 0,5 GB a mais; `--context 2048` iguala ao terminal. O Phi ainda não
foi testado na janela.

O chat do Phi oferece as cinco ferramentas locais: `data_atual`, `dias_ate`, `calcular`, `converter` e `contar`.
Use `--no-tools` para desativá-las; veja os detalhes no [README](../README.md).

### Conferir os arquivos

```sh
gh release download --pattern SHA256SUMS
sha256sum -c --ignore-missing SHA256SUMS        # Linux e Termux; no macOS: shasum -a 256 ARQUIVO
```

No PowerShell, `(Get-FileHash .\qwen-pt-windows-x64.exe).Hash` mostra a soma em maiúsculas; compare com a linha do
arquivo em `SHA256SUMS`.
