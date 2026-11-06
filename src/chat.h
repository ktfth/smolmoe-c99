/* Interactive multi-turn chat (ChatML) over stdin/stdout.
 * The KV cache is kept across turns, so each turn only processes new tokens. */
#ifndef SMOLMOE_CHAT_H
#define SMOLMOE_CHAT_H

#include <stdint.h>

#include "model.h"
#include "output.h"
#include "sampler.h"
#include "tokenizer.h"

typedef struct ChatContext {
    Model *model;
    Tokenizer *tok;
    Sampler *sampler;
    Output *out;
    uint32_t *ids;          /* tokenizer_prepare() capacity */
    char *line;
    uint32_t line_cap;
    unsigned char *piece;   /* tokenizer_max_piece() bytes */
    const char *system;
    uint32_t max_reply;
} ChatContext;

typedef struct ChatStats {
    uint32_t turns, prompt_tokens, generated;
    double prefill_seconds, decode_seconds, first_token_seconds;
} ChatStats;

/* Runs until EOF or "/sair". "/limpar" restarts the conversation, keeping the
 * already-processed system prompt. */
void chat_run(const ChatContext *c, ChatStats *st);

#endif
