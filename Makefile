# smolmoe build. Linux, macOS and Windows (MSYS2/MinGW-w64 GCC or Clang).
#   make                 optimized binaries -> build/smolmoe[.exe], build/smolmoe-quantize[.exe]
#   make test            kernel unit tests + bun integration suite
#   make sanitize        ASan/UBSan build and tests (Linux/macOS)
#   make release         stripped, statically linked where supported

CC      ?= cc
OPT     ?= -O3
STD     := -std=c99 -Wall -Wextra -Wpedantic -Werror -Wshadow -Wstrict-prototypes
# Determinism contract: never allow FMA contraction or reassociation.
FPFLAGS := -ffp-contract=off -fno-fast-math
CFLAGS  ?=
LDFLAGS ?=
BUILD   ?= build

SOURCES := util platform threadpool kernels config tokenizer tokenizer_qwen modelfile model sampler main
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
QUANTIZE   := $(BUILD)/smolmoe-quantize$(EXE)
Q8PACK     := tools/q8pack$(EXE)

OBJ        := $(SOURCES:%=$(BUILD)/obj/%.o)
OBJ_SCALAR := $(SOURCES:%=$(BUILD)/scalar/%.o)
OBJ_LIB    := $(LIB:%=$(BUILD)/obj/%.o)

.PHONY: all scalar tools test test-unit test-integration sanitize release clean bench

all: $(BIN) $(QUANTIZE)
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

$(TEST_KERN): $(BUILD)/tests/test_kernels.o $(OBJ_LIB)
	$(CC) $(OPT) $(PLATFORM) $(LDFLAGS) $^ -o $@ $(LDLIBS)

# Row converter used by tools/import_qwen2.py (POSIX only: Linux, macOS, WSL).
$(Q8PACK): tools/q8pack.c
	@mkdir -p $(dir $@)
	$(CC) $(OPT) -std=c99 -Wall -Wextra -Werror $(FPFLAGS) $< -o $@ -lm

test-unit: $(TEST_KERN)
	$(TEST_KERN)

test-integration: $(BIN) $(BIN_SCALAR)
	SMOLMOE_BUILD=$(BUILD) bun test

test: test-unit test-integration

sanitize:
	$(MAKE) BUILD=build/asan OPT="-O1 -g -fno-omit-frame-pointer" \
		CFLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all" \
		LDFLAGS="-fsanitize=address,undefined" test

release:
	$(MAKE) BUILD=build/release OPT="-O3" LDFLAGS="-s $(STATIC)" all

bench: $(BIN)
	bun scripts/bench.ts

clean:
	rm -rf build tools/q8pack tools/q8pack.exe

-include $(OBJ:.o=.d) $(OBJ_SCALAR:.o=.d) $(BUILD)/tests/test_kernels.d $(BUILD)/obj/quantize_main.d
