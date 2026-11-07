# smolmoe build. Linux, macOS and Windows (MSYS2/MinGW-w64 GCC or Clang), also
# from PowerShell or cmd. `make help` lists the common targets in Portuguese.
#   make                 optimized binaries -> build/smolmoe[.exe], build/smolmoe-quantize[.exe]
#   make test            kernel unit tests + bun integration suite
#   make sanitize        ASan/UBSan build and tests (Linux/macOS)
#   make release         stripped, statically linked where supported (release-gui: the same for the window)
#   make qwen, qwen-gui, qwen-q5, qwen-q5-gui, phi, phi-gui, stories, all-models
#                        dist/: one executable per model, runtime or window + weights
#   make wasi            build/wasi/*.wasm: WebAssembly (WASI) runtime and kernel tests, for bun (needs zig)
#   make web             build/web: the browser page (needs zig and bun)
#   make gui             build/smolmoe-gui[.exe]: the desktop window (C++17; GLFW on Linux/macOS)

# PowerShell and cmd: the recipes need a POSIX shell, and native Windows make
# (winget's, mingw32-make) falls back to cmd.exe when PATH has none. Use the
# one that ships with the compiler, in its folder (w64devkit) or in
# <MSYS2>/usr/bin, and put that folder on PATH for echo, mkdir and the rest.
ifeq ($(OS),Windows_NT)
  ifeq ($(notdir $(SHELL)),sh.exe)
    ifeq ($(wildcard $(SHELL)),)
      POSIX_CC := $(subst \,/,$(firstword $(shell where $(firstword $(CC)) 2>nul)))
      POSIX_SH := $(firstword $(wildcard $(dir $(POSIX_CC))sh.exe $(dir $(POSIX_CC))../../usr/bin/sh.exe))
      ifeq ($(POSIX_SH),)
        $(error no POSIX shell next to the compiler ($(or $(POSIX_CC),$(firstword $(CC)) not on PATH)): install MSYS2 UCRT64 and put <msys2>\ucrt64\bin on PATH, or build from the MSYS2 shell)
      endif
      POSIX_DIR := $(abspath $(dir $(POSIX_SH)))
      SHELL := $(POSIX_DIR)/sh.exe
      # The compiler's folder goes first: a folder earlier on PATH can hold an
      # incompatible DLL of the same name, and gcc then fails without a message.
      export PATH := $(subst /,\,$(abspath $(dir $(POSIX_CC))));$(PATH);$(subst /,\,$(POSIX_DIR))
    endif
  endif
endif

CC      ?= cc
OPT     ?= -O3
STD     := -std=c99 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wstrict-prototypes
# Determinism contract: never allow FMA contraction or reassociation.
FPFLAGS := -ffp-contract=off -fno-fast-math
CFLAGS  ?=
LDFLAGS ?=
BUILD   ?= build

SOURCES := util platform process threadpool kernels vecmath config tokenizer tokenizer_qwen modelfile container bundle model sampler output textfold calendar calc tools shell engine chat main
LIB     := $(filter-out main,$(SOURCES))

ifeq ($(OS),Windows_NT)
  EXE        := .exe
  SYSLIBS    := -lshell32 -lpsapi
  PLATFORM   := -D_WIN32_WINNT=0x0601 -D__USE_MINGW_ANSI_STDIO=1
  STATIC     := -static
  # Test helper that types into a real console (tests/shell.test.ts)
  CONSOLE_INPUT := $(BUILD)/console-input.exe
else
  EXE        :=
  SYSLIBS    := -lpthread
  PLATFORM   := -pthread
  STATIC     :=
endif

ALLFLAGS = $(OPT) $(STD) $(FPFLAGS) $(PLATFORM) $(CFLAGS) -MMD -MP
LDLIBS   = -lm $(SYSLIBS)

BIN        := $(BUILD)/smolmoe$(EXE)
BIN_SCALAR := $(BUILD)/smolmoe-scalar$(EXE)
TEST_KERN  := $(BUILD)/test-kernels$(EXE)
TEST_TOOLS := $(BUILD)/test-tools$(EXE)
TEST_TOK   := $(BUILD)/test-tokenizer$(EXE)
TEST_CALC  := $(BUILD)/test-calc$(EXE)
TEST_SHELL := $(BUILD)/test-shell$(EXE)
TEST_DESKTOP := $(BUILD)/test-desktop$(EXE)
QUANTIZE   := $(BUILD)/smolmoe-quantize$(EXE)
BUNDLE     := $(BUILD)/smolmoe-bundle$(EXE)
Q8PACK     := tools/q8pack$(EXE)

OBJ        := $(SOURCES:%=$(BUILD)/obj/%.o)
OBJ_SCALAR := $(SOURCES:%=$(BUILD)/scalar/%.o)
OBJ_LIB    := $(LIB:%=$(BUILD)/obj/%.o)

.PHONY: all scalar tools test test-unit test-integration sanitize release release-gui clean bench models qwen-pt wasi web gui \
	qwen qwen-gui qwen-q5 qwen-q5-gui phi phi-gui stories all-models phi-weights help

all: $(BIN) $(QUANTIZE) $(BUNDLE)
scalar: $(BIN_SCALAR)
tools: $(Q8PACK)

$(BUILD)/obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(ALLFLAGS) -c $< -o $@

$(BUILD)/scalar/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(ALLFLAGS) -DSM_NO_SIMD -c $< -o $@

$(BUILD)/tests/%.o: tests/%.c
	@mkdir -p $(dir $@)
	$(CC) $(ALLFLAGS) -Isrc -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_SCALAR): $(OBJ_SCALAR)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(QUANTIZE): $(BUILD)/obj/quantize_main.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUNDLE): $(BUILD)/obj/bundle_main.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_TOOLS): $(BUILD)/tests/test_tools.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_KERN): $(BUILD)/tests/test_kernels.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_TOK): $(BUILD)/tests/test_tokenizer.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_CALC): $(BUILD)/tests/test_calc.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_SHELL): $(BUILD)/tests/test_shell.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(TEST_DESKTOP): $(BUILD)/tests/test_desktop.o $(BUILD)/obj/desktop.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(BUILD)/console-input.exe: tests/support/console_input.c
	@mkdir -p $(dir $@)
	$(CC) $(OPT) $(STD) $(PLATFORM) $< -o $@

# Q8 row converter of the old Python Qwen importer; kept as the reference for tests/goldens/convert-q8.json (POSIX only).
$(Q8PACK): tools/q8pack.c
	@mkdir -p $(dir $@)
	$(CC) $(OPT) -std=c99 -Wall -Wextra -Werror $(FPFLAGS) $< -o $@ -lm

test-unit: $(TEST_KERN) $(TEST_TOOLS) $(TEST_TOK) $(TEST_CALC) $(TEST_SHELL) $(TEST_DESKTOP)
	$(TEST_KERN)
	$(TEST_TOOLS)
	$(TEST_TOK)
	$(TEST_CALC)
	$(TEST_SHELL)
	$(TEST_DESKTOP)

# Per-test limit; bun ignores a timeout set in bunfig.toml. SLOW_TEST_MS covers the real-model tests.
TEST_TIMEOUT_MS ?= 30000
SLOW_TEST_MS ?= 180000

test-integration: $(BIN) $(BIN_SCALAR) $(QUANTIZE) $(BUNDLE) $(CONSOLE_INPUT) $(TEST_DESKTOP)
	SMOLMOE_BUILD=$(BUILD) SMOLMOE_SLOW_MS=$(SLOW_TEST_MS) bun test --timeout $(TEST_TIMEOUT_MS)

test: test-unit test-integration

sanitize:
	$(MAKE) BUILD=build/asan OPT="-O1 -g -fno-omit-frame-pointer" \
		CFLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all" \
		LDFLAGS="-fsanitize=address,undefined" SLOW_TEST_MS=600000 test

# RELEASE_DIR and RELEASE_LDFLAGS let the release workflow build a static musl
# binary (-static) or one slice of the macOS universal binary (-arch arm64).
RELEASE_DIR ?= build/release
RELEASE_LDFLAGS ?=
release:
	$(MAKE) BUILD=$(RELEASE_DIR) OPT="-O3" LDFLAGS="-s $(STATIC) $(RELEASE_LDFLAGS)" all
release-gui: release
	$(MAKE) BUILD=$(RELEASE_DIR) OPT="-O3" LDFLAGS="-s $(STATIC) $(RELEASE_LDFLAGS)" gui

# Converted weights from the GitHub release, SHA-256 verified (docs/MODELS.md).
models:
	bun scripts/fetch-models.ts

# One alias per model (docs/CAMINHOS.md): the release runtime or window plus
# the weights in one executable under dist/. `make help` lists them. Missing
# Qwen and TinyStories weights are downloaded and checked; the Q5 file is made
# from the Q8 one; Phi-4-mini is converted only on request (make phi-weights).
DIST        := dist
QWEN        := models/qwen2.5-0.5b-instruct
PHI         := models/phi-4-mini-instruct
STORIES     := models/tinystories-15m
REL_CLI     := $(RELEASE_DIR)/smolmoe$(EXE)
REL_GUI     := $(RELEASE_DIR)/smolmoe-gui$(EXE)
REL_BUNDLE  := $(RELEASE_DIR)/smolmoe-bundle$(EXE)
MODEL_ALIASES := qwen qwen-gui qwen-q5 qwen-q5-gui phi phi-gui stories

# $(call bundle,OUT,MODEL,MODEL_DIR,TOKENIZER,RUNTIME): DIST/OUT[.exe]
define bundle
	@mkdir -p $(DIST)
	$(REL_BUNDLE) --model $(2) --config $(3)/config.ini --tokenizer $(3)/$(4) --exe $(5) --out $(DIST)/$(1)$(EXE)
endef

qwen qwen-pt: release $(QWEN)/model.smoe $(QWEN)/tokenizer.qbp
	$(call bundle,qwen-pt,$(QWEN)/model.smoe,$(QWEN),tokenizer.qbp,$(REL_CLI))
qwen-gui: release-gui $(QWEN)/model.smoe $(QWEN)/tokenizer.qbp
	$(call bundle,qwen-pt-gui,$(QWEN)/model.smoe,$(QWEN),tokenizer.qbp,$(REL_GUI))
qwen-q5: release $(QWEN)/model-q5.smoe $(QWEN)/tokenizer.qbp
	$(call bundle,qwen-pt-q5,$(QWEN)/model-q5.smoe,$(QWEN),tokenizer.qbp,$(REL_CLI))
qwen-q5-gui: release-gui $(QWEN)/model-q5.smoe $(QWEN)/tokenizer.qbp
	$(call bundle,qwen-pt-q5-gui,$(QWEN)/model-q5.smoe,$(QWEN),tokenizer.qbp,$(REL_GUI))
phi: release $(PHI)/model.smoe
	$(call bundle,phi-4-mini,$(PHI)/model.smoe,$(PHI),tokenizer.qbp,$(REL_CLI))
phi-gui: release-gui $(PHI)/model.smoe
	$(call bundle,phi-4-mini-gui,$(PHI)/model.smoe,$(PHI),tokenizer.qbp,$(REL_GUI))
stories: release $(STORIES)/model.smoe $(STORIES)/tokenizer.spb
	$(call bundle,tinystories,$(STORIES)/model.smoe,$(STORIES),tokenizer.spb,$(REL_CLI))

# Every alias whose weights can be had without a long conversion; Phi-4-mini too
# once it is in models/.
all-models: qwen qwen-gui qwen-q5 qwen-q5-gui stories $(if $(wildcard $(PHI)/model.smoe),phi phi-gui)
	$(if $(wildcard $(PHI)/model.smoe),,$(info Phi-4-mini ficou de fora: os pesos não estão em $(PHI) (make phi-weights).))
	@ls -l $(DIST)

# One download per model, also under make -j: the tokenizer waits for the model's fetch.
$(QWEN)/model.smoe:
	bun scripts/fetch-models.ts $(notdir $(QWEN))
$(QWEN)/tokenizer.qbp: | $(QWEN)/model.smoe
	@test -f $@ || bun scripts/fetch-models.ts $(notdir $(QWEN))
$(STORIES)/model.smoe:
	bun scripts/fetch-models.ts $(notdir $(STORIES))
$(STORIES)/tokenizer.spb: | $(STORIES)/model.smoe
	@test -f $@ || bun scripts/fetch-models.ts $(notdir $(STORIES))
# Q5: 39% smaller, perplexity +1.7% (README).
$(QWEN)/model-q5.smoe: $(QWEN)/model.smoe | release
	$(RELEASE_DIR)/smolmoe-quantize$(EXE) --in $< --out $@
$(PHI)/model.smoe:
	$(info Os pesos do Phi-4-mini não estão em $(PHI).)
	$(info Para convertê-los (7,7 GB de download e cerca de 13 GB de disco): make phi-weights)
	$(info Ou use o Phi do Release com o programa: docs/INSTALL.md, seção Phi-4-mini.)
	@exit 1
phi-weights:
	bun scripts/convert.ts fetch-phi3 downloads/phi-4-mini-instruct
	bun scripts/convert.ts phi3 downloads/phi-4-mini-instruct $(PHI)

EMPTY :=
# What can be built, in Portuguese; printed by make itself, which keeps the
# accents that native Windows make would garble on the way to the shell.
help:
	$(info smolmoe: o que dá para compilar. Funciona no PowerShell, no cmd, no Git Bash e nos terminais do Linux e do macOS.)
	$(info )
	$(info $(EMPTY)  make                 programa de terminal: build/smolmoe, smolmoe-bundle, smolmoe-quantize)
	$(info $(EMPTY)  make gui             janela: build/smolmoe-gui)
	$(info $(EMPTY)  make test            testes (em C e com o bun))
	$(info )
	$(info Um executável por modelo, com o programa e os pesos num arquivo só, em $(DIST)/:)
	$(info $(EMPTY)  make qwen            $(DIST)/qwen-pt             chat no terminal, Qwen2.5-0.5B Q8 (560 MB))
	$(info $(EMPTY)  make qwen-gui        $(DIST)/qwen-pt-gui         janela, Qwen Q8)
	$(info $(EMPTY)  make qwen-q5         $(DIST)/qwen-pt-q5          terminal, Qwen Q5: 340 MB, menos memória)
	$(info $(EMPTY)  make qwen-q5-gui     $(DIST)/qwen-pt-q5-gui      janela, Qwen Q5)
	$(info $(EMPTY)  make phi             $(DIST)/phi-4-mini          terminal, Phi-4-mini 3,8B (pesos: make phi-weights))
	$(info $(EMPTY)  make phi-gui         $(DIST)/phi-4-mini-gui      janela, Phi-4-mini)
	$(info $(EMPTY)  make stories         $(DIST)/tinystories         historinhas em inglês, para testar)
	$(info $(EMPTY)  make all-models      todos acima; o Phi só quando os pesos já estão em models/)
	$(info )
	$(info Os pesos do Qwen e do TinyStories são baixados e conferidos na primeira vez.)
	@:

# WebAssembly (WASI) build that the tests run under bun (tests/support/wasi-run.ts):
# the runtime with SIMD128 kernels, and the kernel tests. Needs zig, whose
# `zig cc` ships the WASI libc. Single-threaded: WASI has no threads.
ZIG        ?= zig
WASI       := build/wasi
WASI_CC     = $(ZIG) cc -target wasm32-wasi
WASI_FLAGS  = -O3 $(STD) $(FPFLAGS) -msimd128 -D_WASI_EMULATED_MMAN -MMD -MP
WASI_LINK  := -s -lwasi-emulated-mman -Wl,-z,stack-size=8388608
WASI_OBJ   := $(SOURCES:%=$(WASI)/obj/%.o)
WASI_LIB   := $(LIB:%=$(WASI)/obj/%.o)

$(WASI)/obj/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(WASI_CC) $(WASI_FLAGS) -c $< -o $@

$(WASI)/tests/%.o: tests/%.c
	@mkdir -p $(dir $@)
	$(WASI_CC) $(WASI_FLAGS) -Isrc -c $< -o $@

$(WASI)/smolmoe.wasm: $(WASI_OBJ)
	$(WASI_CC) $^ $(WASI_LINK) -o $@

$(WASI)/test-kernels.wasm: $(WASI)/tests/test_kernels.o $(WASI_LIB)
	$(WASI_CC) $^ $(WASI_LINK) -o $@

# The same with threads: shared memory imported from the host, which starts
# worker instances on it (src/threadpool.c). zig 0.13 reads --max-memory as
# hexadecimal: 7FFF0000 is 2 GiB - 64 KiB.
WASI_MT_FLAGS = $(WASI_FLAGS) -mcpu=generic+atomics+bulk_memory+simd128
WASI_MT_LINK := $(WASI_LINK) -Wl,--shared-memory -Wl,--import-memory -Wl,--max-memory=7FFF0000 \
	-Wl,--export=__stack_pointer
WASI_MT_OBJ  := $(SOURCES:%=$(WASI)/mt/%.o)

$(WASI)/mt/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(WASI_CC) $(WASI_MT_FLAGS) -c $< -o $@

$(WASI)/smolmoe-mt.wasm: $(WASI_MT_OBJ)
	$(WASI_CC) -mcpu=generic+atomics+bulk_memory+simd128 $^ $(WASI_MT_LINK) -o $@

wasi: $(WASI)/smolmoe.wasm $(WASI)/smolmoe-mt.wasm $(WASI)/test-kernels.wasm

# The browser page (web/): the threaded objects with src/web.c instead of
# main, as a reactor (exported functions, no _start), plus the page's
# TypeScript bundled by bun. Serve build/web with `bun scripts/serve-web.ts`.
WEB        := build/web
WEB_OBJ    := $(LIB:%=$(WASI)/mt/%.o) $(WASI)/mt/web.o
WEB_TS     := $(wildcard web/*.ts)

$(WEB)/smolmoe.wasm: $(WEB_OBJ)
	@mkdir -p $(dir $@)
	$(WASI_CC) -mcpu=generic+atomics+bulk_memory+simd128 -mexec-model=reactor $^ $(WASI_MT_LINK) -o $@

$(WEB)/app.js: $(WEB_TS)
	bun build web/app.ts web/engine-worker.ts web/pool-worker.ts --outdir $(WEB) --target browser --minify

web: $(WEB)/smolmoe.wasm $(WEB)/app.js web/index.html web/vercel.json web/models.json
	cp web/index.html web/vercel.json web/models.json $(WEB)/

# The desktop window (gui/, docs/DESKTOP.md): Dear ImGui (third_party/imgui, MIT)
# on Win32 + Direct3D 11, or on GLFW + OpenGL 3 (the GLFW development package:
# libglfw3-dev, brew install glfw). The runtime objects are the C ones above.
GUI        := $(BUILD)/smolmoe-gui$(EXE)
IMGUI      := third_party/imgui
CXXSTD     := -std=c++17
GUI_FLAGS   = $(OPT) $(CXXSTD) -Wall -Wextra -Wpedantic -Werror -Isrc -I$(IMGUI) -I$(IMGUI)/backends $(GUI_CFLAGS) \
	$(PLATFORM) $(CXXFLAGS) -MMD -MP
IMGUI_FLAGS = $(OPT) $(CXXSTD) -I$(IMGUI) $(GUI_CFLAGS) $(PLATFORM) $(CXXFLAGS) -MMD -MP
IMGUI_CORE := imgui imgui_draw imgui_tables imgui_widgets misc/cpp/imgui_stdlib
ifeq ($(OS),Windows_NT)
  GUI_MAIN       := platform_win32
  IMGUI_BACKENDS := backends/imgui_impl_win32 backends/imgui_impl_dx11
  # Static, also in development builds: an older libstdc++ earlier on PATH would
  # stop the window from starting (0xC0000139); the terminal program has no C++.
  GUI_LIBS        = -static -mwindows -ld3d11 -ld3dcompiler -ldxgi -ldwmapi -lgdi32 -lcomdlg32 -ladvapi32
else ifeq ($(shell uname -s),Darwin)
  GUI_MAIN       := platform_glfw
  IMGUI_BACKENDS := backends/imgui_impl_glfw backends/imgui_impl_opengl3
  # Homebrew's GLFW, also without pkg-config.
  GUI_CFLAGS      = $(shell pkg-config --cflags glfw3 2>/dev/null || echo -I$$(brew --prefix)/include)
  GUI_LIBS        = $(shell pkg-config --libs glfw3 2>/dev/null || echo -L$$(brew --prefix)/lib -lglfw) \
	-framework OpenGL -framework Cocoa -framework IOKit
else
  GUI_MAIN       := platform_glfw
  IMGUI_BACKENDS := backends/imgui_impl_glfw backends/imgui_impl_opengl3
  GUI_CFLAGS      = $(shell pkg-config --cflags glfw3)
  GUI_LIBS        = $(shell pkg-config --libs glfw3) -lGL -ldl
endif
GUI_OBJ    := $(addprefix $(BUILD)/gui/,$(GUI_MAIN).o session.o view.o args.o) \
	$(addprefix $(BUILD)/imgui/,$(addsuffix .o,$(IMGUI_CORE) $(IMGUI_BACKENDS)))

$(BUILD)/gui/%.o: gui/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(GUI_FLAGS) -c $< -o $@

$(BUILD)/imgui/%.o: $(IMGUI)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(IMGUI_FLAGS) -c $< -o $@

$(GUI): $(GUI_OBJ) $(BUILD)/obj/desktop.o $(OBJ_LIB)
	$(CXX) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(GUI_LIBS) $(LDLIBS)

gui: $(GUI)

bench: $(BIN)
	bun scripts/bench.ts

clean:
	rm -rf build dist tools/q8pack tools/q8pack.exe

-include $(OBJ:.o=.d) $(OBJ_SCALAR:.o=.d) $(BUILD)/tests/test_kernels.d $(BUILD)/tests/test_tokenizer.d $(BUILD)/tests/test_calc.d \
	$(BUILD)/tests/test_desktop.d $(BUILD)/obj/desktop.d $(GUI_OBJ:.o=.d) \
	$(BUILD)/obj/quantize_main.d $(BUILD)/obj/bundle_main.d $(WASI_OBJ:.o=.d) $(WASI_MT_OBJ:.o=.d) $(WASI)/mt/web.d $(WASI)/tests/test_kernels.d
