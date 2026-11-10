/* One conversation with a chat model, independent of where its text goes: the
 * terminal chat (chat.c), the browser build (web.c) and the desktop window
 * (desktop.c) drive the same engine.
 * The KV cache is kept across messages, so each one only processes new tokens.
 * With tools on, calls the model writes between <tool_call> markers are run by
 * the runtime and their results fed back before the model answers. */
#ifndef SMOLMOE_ENGINE_H
#define SMOLMOE_ENGINE_H

#include <stdint.h>

#include "engine_sink.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"
#include "tools.h"
#include "util.h"

/* The chat's system prompt unless the user gives one (--system). */
#define CHAT_DEFAULT_SYSTEM "Responda somente em português do Brasil. Não misture outros idiomas. Seja breve e direto."
/* Room for the "# Tools" section the chat appends to the system prompt. */
#define TOOLS_PROMPT_BYTES 4096u

typedef struct ChatStats {
    uint32_t turns, prompt_tokens, generated, tool_calls;
    double prefill_seconds, decode_seconds, first_token_seconds;
    /* 0: no disk cache, 1: the system prompt was prefilled, 2: loaded from disk. */
    int system_kv;
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
    Memory *mem; /* turn log; set before engine_start() */
    /* State. */
    uint32_t pos, system_len;
    uint32_t *hist, *turns; /* tokens after the system prompt, and each committed turn's length */
    uint32_t hist_cap, hist_len, nturns;
    uint32_t anchor; /* history index where the current message starts; older turns may be dropped */
    int stopped;            /* the last engine_send() was ended by sink.stop */
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
 * string). When the conversation no longer fits, the oldest turns are dropped
 * and the rest is replayed after the system prompt. Returns 0, with a notice,
 * when the message alone does not fit.
 * When sink.stop ends a reply, e->stopped is set and the reply stays in the
 * history as far as it got, unless it was inside a tool call: then the whole
 * message leaves the history, so it never holds a call without its result. */
int engine_send(Engine *e, const char *text, uint32_t len);

/* Forgets the conversation, keeping the processed system prompt. */
void engine_reset(Engine *e);

#endif
