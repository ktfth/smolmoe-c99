/* Interactive multi-turn chat (ChatML) over stdin/stdout: reads the user's
 * messages and drives the conversation engine (engine.h). Replies go to
 * stdout, the runtime's notices to stderr. */
#ifndef SMOLMOE_CHAT_H
#define SMOLMOE_CHAT_H

#include <stdint.h>

#include "engine.h"
#include "output.h"
#include "shell.h"

typedef struct ChatContext {
    Engine *engine;         /* set up by the caller; chat_run starts it and wires its sink */
    Output *out;
    char *line;
    uint32_t line_cap;
    const Shell *shell;     /* where a typed "!comando" line runs; NULL: "!" is plain text */
    int terminal;           /* stdin is a terminal: lines pasted together form one message */
} ChatContext;

/* Runs until EOF or "/sair". "/limpar" restarts the conversation, keeping the
 * already-processed system prompt. A line typed on its own as "!comando" runs
 * in c->shell and never reaches the model. Statistics end up in c->engine. */
void chat_run(const ChatContext *c);

#endif
