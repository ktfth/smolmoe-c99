# smolmoe build. Linux, macOS and Windows (MSYS2/MinGW-w64 GCC or Clang).
#   make                 optimized binaries -> build/smolmoe[.exe], build/smolmoe-quantize[.exe]
#   make test            kernel unit tests + bun integration suite
#   make sanitize        ASan/UBSan build and tests (Linux/macOS)
#   make release         stripped, statically linked where supported
#   make qwen-pt         dist/qwen-pt[.exe]: runtime + Qwen weights in one file
#   make wasi            build/wasi/*.wasm: WebAssembly (WASI) runtime and kernel tests, for bun (needs zig)

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
QUANTIZE   := $(BUILD)/smolmoe-quantize$(EXE)
BUNDLE     := $(BUILD)/smolmoe-bundle$(EXE)
Q8PACK     := tools/q8pack$(EXE)

OBJ        := $(SOURCES:%=$(BUILD)/obj/%.o)
OBJ_SCALAR := $(SOURCES:%=$(BUILD)/scalar/%.o)
OBJ_LIB    := $(LIB:%=$(BUILD)/obj/%.o)

.PHONY: all scalar tools test test-unit test-integration sanitize release clean bench models qwen-pt wasi

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

$(BUILD)/console-input.exe: tests/support/console_input.c
	@mkdir -p $(dir $@)
	$(CC) $(OPT) $(STD) $(PLATFORM) $< -o $@

# Q8 row converter of the old Python Qwen importer; kept as the reference for tests/goldens/convert-q8.json (POSIX only).
$(Q8PACK): tools/q8pack.c
	@mkdir -p $(dir $@)
	$(CC) $(OPT) -std=c99 -Wall -Wextra -Werror $(FPFLAGS) $< -o $@ -lm

test-unit: $(TEST_KERN) $(TEST_TOOLS) $(TEST_TOK) $(TEST_CALC) $(TEST_SHELL)
	$(TEST_KERN)
	$(TEST_TOOLS)
	$(TEST_TOK)
	$(TEST_CALC)
	$(TEST_SHELL)

# Per-test limit; bun ignores a timeout set in bunfig.toml. SLOW_TEST_MS covers the real-model tests.
TEST_TIMEOUT_MS ?= 30000
SLOW_TEST_MS ?= 180000

test-integration: $(BIN) $(BIN_SCALAR) $(QUANTIZE) $(BUNDLE) $(CONSOLE_INPUT)
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

wasi: $(WASI)/smolmoe.wasm $(WASI)/test-kernels.wasm

bench: $(BIN)
	bun scripts/bench.ts

clean:
	rm -rf build dist tools/q8pack tools/q8pack.exe

-include $(OBJ:.o=.d) $(OBJ_SCALAR:.o=.d) $(BUILD)/tests/test_kernels.d $(BUILD)/tests/test_tokenizer.d $(BUILD)/tests/test_calc.d \
	$(BUILD)/obj/quantize_main.d $(BUILD)/obj/bundle_main.d $(WASI_OBJ:.o=.d) $(WASI)/tests/test_kernels.d
