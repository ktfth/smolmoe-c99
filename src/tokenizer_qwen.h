/* Qwen2 byte-level BPE, Unicode 15 NFC and Qwen2 pre-tokenization rules. */
#ifndef SMOLMOE_TOKENIZER_QWEN_H
#define SMOLMOE_TOKENIZER_QWEN_H

#include <stdint.h>

#include "util.h"

typedef struct QwenTokenizer QwenTokenizer;

QwenTokenizer *qwen_load(ByteReader *r, uint32_t vocab, Memory *mem);
uint32_t qwen_prepare(QwenTokenizer *q, uint32_t max_bytes, Memory *mem);
uint32_t qwen_encode(QwenTokenizer *q, const unsigned char *s, uint32_t bytes, uint32_t *ids,
                     int chat, const char *system);
/* Multi-turn chat: one complete ChatML message, optionally opening the
 * assistant reply; and the "<|im_end|>\n" that closes a generated reply. */
uint32_t qwen_chat_turn(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                        int open_assistant, uint32_t *ids);
uint32_t qwen_chat_close(QwenTokenizer *q, uint32_t *ids);
uint32_t qwen_decode(const QwenTokenizer *q, uint32_t id, unsigned char *buf);
uint32_t qwen_max_piece(const QwenTokenizer *q);
int qwen_is_special(const QwenTokenizer *q, uint32_t id);
int qwen_tool_markers(const QwenTokenizer *q, uint32_t *open, uint32_t *close);

#endif
