/* Interactive multi-turn chat (ChatML) over stdin/stdout.
 * The KV cache is kept across turns, so each turn only processes new tokens.
 * With tools on, calls the model writes between <tool_call> markers are run by
 * the runtime and their results fed back before the model answers. */
#ifndef SMOLMOE_CHAT_H
#define SMOLMOE_CHAT_H

#include <stdint.h>

#include "model.h"
#include "output.h"
#include "sampler.h"
#include "tokenizer.h"
#include "tools.h"

typedef struct ChatContext {
    Model *model;
    Tokenizer *tok;
    Sampler *sampler;
    Output *out;
    uint32_t *ids;          /* tokenizer_prepare() capacity */
    char *line;
    uint32_t line_cap;
    unsigned char *piece;   /* tokenizer_max_piece() bytes */
    const char *system;     /* already including the tool declarations when tools are on */
    uint32_t max_reply;
    int tools;              /* tool declarations are in the system prompt */
    uint32_t tool_open, tool_close;
    unsigned char *banned;  /* the sampler's ban list: the markers are allowed per turn */
    const ToolClock *fixed_clock; /* NULL: the tools read the local clock at each call */
} ChatContext;

typedef struct ChatStats {
    uint32_t turns, prompt_tokens, generated, tool_calls;
    double prefill_seconds, decode_seconds, first_token_seconds;
} ChatStats;

/* Runs until EOF or "/sair". "/limpar" restarts the conversation, keeping the
 * already-processed system prompt. */
void chat_run(const ChatContext *c, ChatStats *st);

#endif
