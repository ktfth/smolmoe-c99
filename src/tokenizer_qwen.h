/* SMQBP1 byte-level BPE, first written for Qwen2 and now shared by every
 * byte-level BPE model: v1 header (Qwen2.5: NFC, Qwen2 pre-tokenizer, ChatML)
 * or v2 header naming the normalizer, pre-tokenizer and chat template
 * (Phi-4-mini: none, o200k, Phi). See docs/FORMAT.md. */
#ifndef SMOLMOE_TOKENIZER_QWEN_H
#define SMOLMOE_TOKENIZER_QWEN_H

#include <stdint.h>

#include "util.h"

/* v2 header enums; any other value is rejected at load. */
enum { QBP_NORM_NONE = 0, QBP_NORM_NFC = 1 };
enum { QBP_PRETOK_QWEN2 = 1, QBP_PRETOK_O200K = 2, QBP_PRETOK_GEMMA = 3 };
enum { QBP_CHAT_NONE = 0, QBP_CHAT_CHATML = 1, QBP_CHAT_PHI = 2, QBP_CHAT_GEMMA = 3 };

typedef struct QwenTokenizer QwenTokenizer;

QwenTokenizer *qwen_load(ByteReader *r, uint32_t vocab, Memory *mem);
uint32_t qwen_prepare(QwenTokenizer *q, uint32_t max_bytes, Memory *mem);
uint32_t qwen_encode(QwenTokenizer *q, const unsigned char *s, uint32_t bytes, uint32_t *ids,
                     int chat, const char *system);
/* Multi-turn chat: one complete message in the tokenizer's template, optionally
 * opening the assistant reply; and the tokens that close a generated reply
 * (ChatML "<|im_end|>\n", Phi "<|end|>"). */
uint32_t qwen_chat_turn(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                        int open_assistant, uint32_t *ids);
uint32_t qwen_chat_close(QwenTokenizer *q, uint32_t *ids);
uint32_t qwen_decode(const QwenTokenizer *q, uint32_t id, unsigned char *buf);
uint32_t qwen_max_piece(const QwenTokenizer *q);
int qwen_is_special(const QwenTokenizer *q, uint32_t id);
int qwen_supports_chat(const QwenTokenizer *q);
/* ChatML's <tool_call> and </tool_call>; 0 for templates without a tool-call protocol. */
int qwen_tool_markers(const QwenTokenizer *q, uint32_t *open, uint32_t *close);

#endif
