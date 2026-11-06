/* Tokenizers. Three on-disk formats are supported:
 *   SMBPE1  text, byte fallback + ranked merges (synthetic fixtures)
 *   SMSPB1  binary score-BPE (llama2.c / TinyStories)
 *   SMQBP1  binary Qwen2 byte-level BPE with NFC and pre-tokenization
 * IDs 0..255 are bytes, 256 is BOS and 257 is EOS in every format. */
#ifndef SMOLMOE_TOKENIZER_H
#define SMOLMOE_TOKENIZER_H

#include <stdint.h>

#include "util.h"

typedef struct Tokenizer Tokenizer;
typedef void (*ByteSink)(void *ctx, const unsigned char *bytes, uint32_t n);

Tokenizer *tokenizer_load(const char *path, uint32_t vocab, Memory *mem);

/* Allocates encoding workspace for prompts up to max_bytes raw bytes and
 * returns the token-ID capacity that tokenizer_encode() may write. */
uint32_t tokenizer_prepare(Tokenizer *t, uint32_t max_bytes, Memory *mem);

/* Encodes raw bytes. `chat` wraps the prompt in a single-turn ChatML template
 * (Qwen only). Returns the number of IDs written. */
uint32_t tokenizer_encode(Tokenizer *t, const unsigned char *s, uint32_t n, uint32_t *ids,
                          int chat, const char *system);

/* Writes the bytes of one generated token into buf (tokenizer_max_piece bytes). */
uint32_t tokenizer_decode(Tokenizer *t, uint32_t id, unsigned char *buf);
uint32_t tokenizer_max_piece(const Tokenizer *t);

/* Renders an encoded prompt back to text, as --tokenize-only prints it. */
void tokenizer_render_prompt(Tokenizer *t, const uint32_t *ids, uint32_t n, unsigned char *buf,
                             ByteSink sink, void *ctx);

int tokenizer_supports_chat(const Tokenizer *t);
/* Control tokens that must never be sampled (EOS excepted). */
int tokenizer_is_control(const Tokenizer *t, uint32_t id);

#endif
