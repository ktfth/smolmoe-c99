CC = cc
CPPFLAGS ?=
CFLAGS ?= -O3 -std=c99 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off
LDLIBS = -lm

.PHONY: all clean test sanitize demo pretrained-demo pretrained-test portugues-demo portugues-test
all: smolmoe
smolmoe: src/smolmoe.c src/qwen_tokenizer.h src/unicode.h src/unicode_data.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)
smolmoe-scalar: src/smolmoe.c src/qwen_tokenizer.h src/unicode.h src/unicode_data.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSM_NO_AVX2 $< -o $@ $(LDLIBS)
tests/kernels: tests/kernels.c src/smolmoe.c src/qwen_tokenizer.h src/unicode.h src/unicode_data.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)
tools/q8pack: tools/q8pack.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)
test: smolmoe smolmoe-scalar tests/kernels tools/q8pack
	./tests/kernels
	python3 -m unittest discover -s tests -v
sanitize:
	$(MAKE) clean
	$(MAKE) test CFLAGS='-O1 -g -std=c99 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer -fno-pie -no-pie'
demo: smolmoe
	python3 tools/checkpoint.py fixture examples/demo --echo
	./smolmoe --model examples/demo/model.smoe --config examples/demo/config.ini --tokenizer examples/demo/tokenizer.bpe --prompt Hi --tokens 32 --memory-mib 4
pretrained-demo: smolmoe
	./smolmoe --model examples/tinystories15m/model.smoe --config examples/tinystories15m/config.ini --tokenizer examples/tinystories15m/tokenizer.spb --prompt 'Once upon a time, there was a little robot' --tokens 80 --context 256 --memory-mib 16
pretrained-test: smolmoe
	python3 tools/validate_pretrained.py examples/tinystories15m
portugues-demo: smolmoe
	./smolmoe --model examples/qwen-portugues/model.smoe --config examples/qwen-portugues/config.ini --tokenizer examples/qwen-portugues/tokenizer.qbp --chat --prompt 'Escreva uma saudação curta para alguém que está aprendendo programação.' --tokens 64 --context 512 --memory-mib 256
portugues-test: smolmoe
	python3 tools/validate_portuguese.py examples/qwen-portugues
clean:
	rm -f smolmoe smolmoe-scalar tests/kernels tools/q8pack
