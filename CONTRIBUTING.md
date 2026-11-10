# Como contribuir

O runtime é **C99 com extensões do GCC e do Clang**, não C99 estrito. `__atomic_*`, a detecção de AVX2/F16C e o atributo `target` ficam em `src/compat.h`, que também tem o ramo do MSVC (`Interlocked*`, `__cpuid`). O MSVC ainda não compila o projeto. Uma extensão nova entra nesse cabeçalho, com os dois ramos, em vez de espalhar builtin pelo resto do código.

## Compilar e testar

No Linux: `gcc` ou `clang` e `make`. A janela pede `g++`, `pkg-config` e `libglfw3-dev`. No macOS, `brew install glfw`. No Windows, MSYS2 UCRT64. Os testes de integração, o conversor e os scripts usam [Bun](https://bun.sh). O WebAssembly usa `zig` 0.13.

```sh
make            # build/smolmoe, smolmoe-bundle, smolmoe-quantize
make test       # testes em C e a suíte no Bun
make sanitize   # ASan + UBSan (Linux e macOS)
make fuzz       # libFuzzer por alguns segundos (precisa de Clang)
```

`make test` é o mínimo antes de um pull request. O build trata `-Wconversion` como erro.

## Estilo

- C99 no runtime. A janela é C++17 e não entra no binário do terminal.
- Sem dependência nova de runtime. A GLFW fica só na janela, no Linux e no macOS.
- O número não pode mudar com `--threads`, `--batch` ou com o kernel. O Makefile já passa `-ffp-contract=off`.
- Entrada inválida termina em `die()`, com uma frase curta. Um arquivo quebrado não segue em frente.
- Comentário de código em inglês, como o resto de `src/`. O que a pessoa lê no chat fica em português.

## Arquivo de modelo

O parser lê arquivo de terceiro. O caminho feliz não basta: `fuzz/` e os testes de validação cobrem o arquivo quebrado. Se mudar `modelfile.c`, `tokenizer*.c` ou `config.c`, rode `make fuzz`.
