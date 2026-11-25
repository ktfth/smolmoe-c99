# Builds isolados com Gentoo e Podman

Este guia documenta como compilar cada parte do `smolmoe` em contêineres **Gentoo** isolados via **Podman**, com foco principal em cross-compile para **Windows x64** (`dev-util/mingw64-toolchain`).

## Imagem de toolchain

A imagem `localhost/smolmoe-gentoo-mingw:latest` é definida em [container/gentoo/Containerfile](../container/gentoo/Containerfile) a partir das imagens oficiais `docker.io/gentoo/stage3:latest` e `docker.io/gentoo/portage:latest`, instalando `dev-util/mingw64-toolchain` diretamente do binhost oficial do Gentoo (`--getbinpkg --usepkgonly`):

```sh
bun scripts/podman-gentoo.ts image
```

## Partes isoladas

Cada parte roda em seu próprio contêiner descartável (`podman run --rm`). O repositório é montado somente-leitura (`/src:ro`) e copiado para `/work` dentro do contêiner — **incluindo alterações locais não commitadas** (e excluindo `build/`, `dist/`, `models/`, `downloads/`, `.git/`, `.claude/`). Os artefatos finais são gravados em `build/gentoo/<parte>/`:

| Parte | Alvo | Artefatos |
|---|---|---|
| `win-cli` | Windows x64 (MinGW-w64, estático) | `smolmoe.exe`, `smolmoe-quantize.exe`, `smolmoe-bundle.exe` |
| `win-scalar` | Windows x64 (`-DSM_NO_SIMD`, estático) | `smolmoe-scalar.exe` |
| `win-gui` | Windows x64 (Dear ImGui + Win32 + D3D11, estático) | `smolmoe-gui.exe` |
| `win-tests` | Windows x64 (testes unitários C + helpers, estático) | `test-*.exe`, `console-input.exe` |
| `win-inventario` | Windows x64 (inventário defensivo, estático) | `smolmoe-inventario.exe`, `test-inventory.exe` |
| `linux-cli` | Linux x64 (GCC nativo do Gentoo + `test-unit`) | `smolmoe`, `smolmoe-scalar`, `smolmoe-quantize`, `smolmoe-bundle`, `smolmoe-inventario` |
| `wasm` | WebAssembly (`dev-lang/zig-bin` no Gentoo, imagem isolada `smolmoe-gentoo-wasm`) | `wasi/{smolmoe,smolmoe-mt,test-kernels}.wasm`, `web/smolmoe.wasm` (também sincroniza `build/web/` pronto para `bun scripts/serve-web.ts`) |

## Comandos

```sh
# Listar partes disponíveis
bun scripts/podman-gentoo.ts list

# Compilar todas as partes de Windows (padrão)
bun scripts/podman-gentoo.ts build

# Compilar partes específicas
bun scripts/podman-gentoo.ts build win-cli win-gui

# Compilar todas as partes (Windows + Linux)
bun scripts/podman-gentoo.ts build all
```
