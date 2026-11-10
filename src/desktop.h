/* A chat model opened for the desktop window (gui/), with the terminal chat's
 * defaults: greedy decoding, the Portuguese system prompt, the tools when the
 * model has them. The model comes from a self-contained .smoe, from a .smoe
 * with config.ini and tokenizer.qbp (or .spb) beside it (the layout of
 * models/), from the container appended to the running executable
 * (smolmoe-bundle), or from <stem>.smoe beside that executable when the
 * weights do not fit in a Windows executable.
 *
 * Any thread may call these functions, one call at a time. Expected problems
 * (a missing file, a file that is not a model, a model without chat) come back
 * as messages; anything else goes through die(), so a window installs a die
 * handler (util.h) to show it. This header needs no other runtime header, so
 * C++ includes it directly. */
#ifndef SMOLMOE_DESKTOP_H
#define SMOLMOE_DESKTOP_H

#include <stddef.h>
#include <stdint.h>

#include "engine_sink.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DESKTOP_DEFAULT_CONTEXT 4096u
#define DESKTOP_REPLY_TOKENS 512u
#define DESKTOP_ERROR_BYTES 512u

/* The slow parts of desktop_open(), reported as they start. */
typedef enum DesktopStage {
    DESKTOP_CHECKING,  /* every weight is validated once: seconds for a large model */
    DESKTOP_PREPARING  /* buffers, KV cache and the system prompt */
} DesktopStage;

typedef struct DesktopOptions {
    const char *path;  /* UTF-8; NULL: the model appended to this executable */
    uint32_t context;  /* 0: DESKTOP_DEFAULT_CONTEXT; capped at the model's */
    uint32_t threads;  /* 0: the physical cores */
    EngineSink sink;   /* the replies; text, reply_end and notice are required */
    void (*stage)(void *ctx, DesktopStage stage); /* called with sink.ctx; may be NULL */
} DesktopOptions;

typedef struct DesktopInfo {
    const char *backend; /* kernels: "avx2", "neon", "scalar"... */
    uint32_t threads, context;
    int tools;           /* the date and calculation tools are on */
    int stopped;         /* the last message's reply was stopped */
    /* Totals since the model opened (the system prompt included). */
    uint32_t turns, prompt_tokens, generated, tool_calls;
    double prefill_seconds, decode_seconds;
} DesktopInfo;

typedef struct Desktop Desktop;

/* Whether this executable carries a model, appended or as a sibling .smoe. */
int desktop_has_bundle(void);

/* Opens a model and processes the system prompt. Returns NULL with a message
 * in Portuguese in err when the file cannot be a chat model. */
Desktop *desktop_open(const DesktopOptions *o, char err[DESKTOP_ERROR_BYTES]);

/* Answers one message (UTF-8, len bytes). Returns 0, with a notice, when it
 * does not fit in the context. */
int desktop_send(Desktop *d, const char *text, size_t len);

/* Starts a new conversation, keeping the processed system prompt. */
void desktop_reset(Desktop *d);

void desktop_info(const Desktop *d, DesktopInfo *out);
void desktop_close(Desktop *d);

/* A notice as a window shows it: "[ferramenta] calcular: call -> {result}"
 * becomes "ferramenta calcular: <resumo>" (or "erro: <erro>") when the result
 * has one; any other line is copied. Returns the length written. */
size_t desktop_notice_text(const char *line, char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif
