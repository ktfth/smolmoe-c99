# smolmoe build. Linux, macOS and Windows (MSYS2/MinGW-w64 GCC or Clang).
#   make                 optimized binaries -> build/smolmoe[.exe], build/smolmoe-quantize[.exe]
#   make test            kernel unit tests + bun integration suite
#   make sanitize        ASan/UBSan build and tests (Linux/macOS)
#   make release         stripped, statically linked where supported (release-gui: the same for the window)
#   make qwen-pt         dist/qwen-pt[.exe]: runtime + Qwen weights in one file
#   make wasi            build/wasi/*.wasm: WebAssembly (WASI) runtime and kernel tests, for bun (needs zig)
#   make web             build/web: the browser page (needs zig and bun)
#   make gui             build/smolmoe-gui[.exe]: the desktop window (C++17; GLFW on Linux/macOS)

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

.PHONY: all scalar tools test test-unit test-integration sanitize release release-gui clean bench models qwen-pt wasi web gui

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

# Self-contained Portuguese chat executable; downloads the Qwen weights if missing.
QWEN := models/qwen2.5-0.5b-instruct
qwen-pt: release
	bun scripts/fetch-models.ts $(notdir $(QWEN))
	@mkdir -p dist
	build/release/smolmoe-bundle$(EXE) --model $(QWEN)/model.smoe --config $(QWEN)/config.ini \
		--tokenizer $(QWEN)/tokenizer.qbp --exe build/release/smolmoe$(EXE) --out dist/qwen-pt$(EXE)

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
  GUI_LIBS        = -mwindows -ld3d11 -ld3dcompiler -ldxgi -ldwmapi -lgdi32 -lcomdlg32 -ladvapi32
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
