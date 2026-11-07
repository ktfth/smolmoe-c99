/* One conversation with a chat model, independent of where its text goes: the
 * terminal chat (chat.c) and the browser build drive the same engine.
 * The KV cache is kept across messages, so each one only processes new tokens.
 * With tools on, calls the model writes between <tool_call> markers are run by
 * the runtime and their results fed back before the model answers. */
#ifndef SMOLMOE_ENGINE_H
#define SMOLMOE_ENGINE_H

#include <stdint.h>

#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "tools.h"

/* The chat's system prompt unless the user gives one (--system). */
#define CHAT_DEFAULT_SYSTEM "Responda somente em português do Brasil. Não misture outros idiomas. Seja breve e direto."
/* Room for the "# Tools" section the chat appends to the system prompt. */
#define TOOLS_PROMPT_BYTES 4096u

/* Where the engine's output goes. The pointers passed to the callbacks are
 * valid only during the call. */
typedef struct EngineSink {
    void *ctx;
    /* Reply bytes as they are generated; a UTF-8 character may be split
     * across calls, and n may be 0. */
    void (*text)(void *ctx, const unsigned char *bytes, uint32_t n);
    /* After each reply: anything held back can be written out now. */
    void (*reply_end)(void *ctx);
    /* One line about what the runtime did (a tool call, a restart), without
     * the line ending. The tool name and call the model wrote have their
     * control bytes replaced; a tool result is JSON. */
    void (*notice)(void *ctx, const char *line);
} EngineSink;

typedef struct ChatStats {
    uint32_t turns, prompt_tokens, generated, tool_calls;
    double prefill_seconds, decode_seconds, first_token_seconds;
} ChatStats;

typedef struct Engine {
    /* Set by the caller before engine_start(). */
    Model *model;
    Tokenizer *tok;
    Sampler *sampler;
    uint32_t *ids;          /* tokenizer_prepare() capacity */
    unsigned char *piece;   /* tokenizer_max_piece() bytes */
    const char *system;     /* already including the tool declarations when tools are on */
    uint32_t max_reply;
    int tools;              /* tool declarations are in the system prompt */
    uint32_t tool_open, tool_close;
    unsigned char *banned;  /* the sampler's ban list: the markers are allowed per message */
    const ToolClock *fixed_clock; /* NULL: the tools read the local clock at each call */
    EngineSink sink;
    /* State. */
    uint32_t pos, system_len;
    const char *message;    /* the user message being answered */
    ChatStats stats;
} Engine;

/* The system prompt with the tool declarations, allocated from mem, when the
 * vocabulary has the call markers (filled in) and the declarations take at
 * most half of `context` tokens. Otherwise NULL: the chat runs with `system`
 * and the markers banned; *needed is then the context the declarations would
 * need, or 0 when the model has no tools at all. ids: tokenizer_prepare() room
 * for system + TOOLS_PROMPT_BYTES bytes. */
const char *engine_tools_prompt(Tokenizer *tok, const char *system, uint32_t context, uint32_t *ids,
                                uint32_t *tool_open, uint32_t *tool_close, uint32_t *needed, Memory *mem);

/* Processes the system prompt; dies when it does not fit in the context. */
void engine_start(Engine *e);

/* Answers one user message: len bytes followed by a NUL (the tools read it as a
 * string). When the conversation no longer fits, it restarts from the system
 * prompt first. Returns 0, with a notice, when the message alone does not fit. */
int engine_send(Engine *e, const char *text, uint32_t len);

/* Forgets the conversation, keeping the processed system prompt. */
void engine_reset(Engine *e);

#endif
