# smolmoe build. Linux, macOS and Windows (MSYS2/MinGW-w64 GCC or Clang).
#   make                 optimized binaries -> build/smolmoe[.exe], build/smolmoe-quantize[.exe]
#   make test            kernel unit tests + bun integration suite
#   make sanitize        ASan/UBSan build and tests (Linux/macOS)
#   make release         stripped, statically linked where supported
#   make qwen-pt         dist/qwen-pt[.exe]: runtime + Qwen weights in one file

CC      ?= cc
OPT     ?= -O3
STD     := -std=c99 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wstrict-prototypes
# Determinism contract: never allow FMA contraction or reassociation.
FPFLAGS := -ffp-contract=off -fno-fast-math
CFLAGS  ?=
LDFLAGS ?=
BUILD   ?= build

SOURCES := util platform threadpool kernels vecmath config tokenizer tokenizer_qwen modelfile container bundle model sampler output textfold calendar calc tools chat main
LIB     := $(filter-out main,$(SOURCES))

ifeq ($(OS),Windows_NT)
  EXE        := .exe
  SYSLIBS    := -lshell32 -lpsapi
  PLATFORM   := -D_WIN32_WINNT=0x0601 -D__USE_MINGW_ANSI_STDIO=1
  STATIC     := -static
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
QUANTIZE   := $(BUILD)/smolmoe-quantize$(EXE)
BUNDLE     := $(BUILD)/smolmoe-bundle$(EXE)
Q8PACK     := tools/q8pack$(EXE)

OBJ        := $(SOURCES:%=$(BUILD)/obj/%.o)
OBJ_SCALAR := $(SOURCES:%=$(BUILD)/scalar/%.o)
OBJ_LIB    := $(LIB:%=$(BUILD)/obj/%.o)

.PHONY: all scalar tools test test-unit test-integration sanitize release clean bench models qwen-pt

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

# Q8 row converter of the old Python Qwen importer; kept as the reference for tests/goldens/convert-q8.json (POSIX only).
$(Q8PACK): tools/q8pack.c
	@mkdir -p $(dir $@)
	$(CC) $(OPT) -std=c99 -Wall -Wextra -Werror $(FPFLAGS) $< -o $@ -lm

test-unit: $(TEST_KERN) $(TEST_TOOLS) $(TEST_TOK) $(TEST_CALC)
	$(TEST_KERN)
	$(TEST_TOOLS)
	$(TEST_TOK)
	$(TEST_CALC)

# Per-test limit; bun ignores a timeout set in bunfig.toml. SLOW_TEST_MS covers the real-model tests.
TEST_TIMEOUT_MS ?= 30000
SLOW_TEST_MS ?= 180000

test-integration: $(BIN) $(BIN_SCALAR) $(QUANTIZE) $(BUNDLE)
	SMOLMOE_BUILD=$(BUILD) SMOLMOE_SLOW_MS=$(SLOW_TEST_MS) bun test --timeout $(TEST_TIMEOUT_MS)

test: test-unit test-integration

sanitize:
	$(MAKE) BUILD=build/asan OPT="-O1 -g -fno-omit-frame-pointer" \
		CFLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all" \
		LDFLAGS="-fsanitize=address,undefined" SLOW_TEST_MS=600000 test

release:
	$(MAKE) BUILD=build/release OPT="-O3" LDFLAGS="-s $(STATIC)" all

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

bench: $(BIN)
	bun scripts/bench.ts

clean:
	rm -rf build dist tools/q8pack tools/q8pack.exe

-include $(OBJ:.o=.d) $(OBJ_SCALAR:.o=.d) $(BUILD)/tests/test_kernels.d $(BUILD)/tests/test_tokenizer.d $(BUILD)/tests/test_calc.d \
	$(BUILD)/obj/quantize_main.d $(BUILD)/obj/bundle_main.d
