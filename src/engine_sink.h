/* Where a conversation's output goes (engine.h), in a header of its own so a
 * front end can include it without the model's headers (desktop.h). */
#ifndef SMOLMOE_ENGINE_SINK_H
#define SMOLMOE_ENGINE_SINK_H

#include <stdint.h>

/* The pointers passed to the callbacks are valid only during the call. */
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
    /* Polled before each generated token, possibly from another thread's
     * request: nonzero ends the reply there. NULL: replies are never stopped. */
    int (*stop)(void *ctx);
} EngineSink;

#endif
