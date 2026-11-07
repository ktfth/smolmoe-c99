/* Entry points of the browser build: a WebAssembly reactor (no main). The
 * page's worker (web/engine-worker.ts) copies a self-contained .smoe into
 * memory from sm_alloc(), opens it with sm_open() and sends messages with
 * sm_send(); the engine's output comes back through the imported smolmoe.*
 * functions. Same defaults as the terminal chat: greedy decoding, the default
 * system prompt, tools when the model has them. Fatal errors go through die()
 * (stderr, then proc_exit), which the worker reports to the page. */
#include "config.h"
#include "engine.h"
#include "kernels.h"
#include "modelfile.h"
#include "util.h"

#include <stdlib.h>
#include <string.h>

#define WEB_IMPORT(name) __attribute__((import_module("smolmoe"), import_name(name)))
#define WEB_EXPORT(name) __attribute__((export_name(name)))

#define WEB_MEMORY_LIMIT (1024u * 1048576u) /* weights are outside the budget: they are the caller's */
#define WEB_BATCH 32u
#define WEB_REPLY_TOKENS 512u
#define WEB_MAX_CONTEXT 8192u
#define WEB_BYTES_PER_TOKEN 4u

WEB_IMPORT("text") void host_text(const unsigned char *bytes, uint32_t n);
WEB_IMPORT("reply_end") void host_reply_end(void);
WEB_IMPORT("notice") void host_notice(const char *line);

/* One chat per module instance. */
static struct {
    int open;
    Memory mem;
    ModelFile mf;
    Config c;
    Model model;
    Sampler sampler;
    Tokenizer *tok;
    Pool *pool;
    Engine engine;
    ToolClock clock;
    char *message;
    uint32_t message_cap;
    const char *backend;
    double stats[6];
} web;

static void sink_text(void *ctx, const unsigned char *bytes, uint32_t n) {
    (void)ctx;
    host_text(bytes, n);
}

static void sink_reply_end(void *ctx) {
    (void)ctx;
    host_reply_end();
}

static void sink_notice(void *ctx, const char *line) {
    (void)ctx;
    host_notice(line);
}

static const Tensor *required_blob(const char *name, const char *what) {
    const Tensor *t = modelfile_blob(&web.mf, name);
    if (!t) die("this .smoe has no embedded %s: use a self-contained model (smolmoe-bundle)", what);
    return t;
}

WEB_EXPORT("sm_alloc") void *sm_alloc(uint32_t size) { return malloc(size); }

/* Opens the model in data[0..size), which must stay allocated, with a context
 * of `context` tokens (0: the model's, capped) and up to `threads` threads.
 * Returns 0 when a model is already open. */
WEB_EXPORT("sm_open") int sm_open(const unsigned char *data, uint32_t size, uint32_t context, uint32_t threads) {
    const Tensor *blob;
    unsigned char *banned;
    uint32_t i, ids, needed, open_marker = 0, close_marker = 0, encode_cap;
    const char *tools;
    if (web.open) return 0;
    web.mem.limit = WEB_MEMORY_LIMIT;
    modelfile_parse(&web.mf, data, size, &web.mem);
    blob = required_blob(BLOB_CONFIG, "config");
    web.c = config_parse(blob->data, (size_t)blob->bytes);
    blob = required_blob(BLOB_TOKENIZER, "tokenizer");
    web.tok = tokenizer_parse(blob->data, (size_t)blob->bytes, web.c.vocab, &web.mem);
    if (!tokenizer_supports_chat(web.tok)) die("this model has no chat template");
    if (!context || context > web.c.max_context) context = web.c.max_context;
    if (context > WEB_MAX_CONTEXT) context = WEB_MAX_CONTEXT;
    web.backend = kernels_select(0);
    web.pool = pool_create(threads, &web.mem);
    modelfile_verify_payload(&web.mf, web.pool);
    model_init(&web.model, &web.c, &web.mf, context, WEB_BATCH < context ? WEB_BATCH : context, 1, web.pool, &web.mem);

    web.message_cap = context * WEB_BYTES_PER_TOKEN;
    encode_cap = (uint32_t)strlen(CHAT_DEFAULT_SYSTEM) + TOOLS_PROMPT_BYTES;
    ids = tokenizer_prepare(web.tok, encode_cap > web.message_cap ? encode_cap : web.message_cap, &web.mem);
    web.message = (char *)mem_alloc(&web.mem, (size_t)web.message_cap + 1, 1);
    web.engine.ids = (uint32_t *)mem_alloc(&web.mem, ids, sizeof(uint32_t));
    web.engine.piece = (unsigned char *)mem_alloc(&web.mem, tokenizer_max_piece(web.tok), 1);
    banned = (unsigned char *)mem_alloc(&web.mem, web.c.vocab, 1);
    for (i = 0; i < web.c.vocab; ++i) banned[i] = (unsigned char)tokenizer_is_control(web.tok, i);
    tools = engine_tools_prompt(web.tok, CHAT_DEFAULT_SYSTEM, context, web.engine.ids, &open_marker, &close_marker,
                                &needed, &web.mem);
    sampler_init(&web.sampler, web.c.vocab, banned, 0.0f, 40, 0.95f, 1, &web.mem);

    web.engine.model = &web.model;
    web.engine.tok = web.tok;
    web.engine.sampler = &web.sampler;
    web.engine.system = tools ? tools : CHAT_DEFAULT_SYSTEM;
    web.engine.max_reply = WEB_REPLY_TOKENS;
    web.engine.tools = tools != NULL;
    web.engine.tool_open = open_marker;
    web.engine.tool_close = close_marker;
    web.engine.banned = banned;
    web.engine.fixed_clock = &web.clock;
    web.engine.sink.text = sink_text;
    web.engine.sink.reply_end = sink_reply_end;
    web.engine.sink.notice = sink_notice;
    engine_start(&web.engine);
    web.open = 1;
    return 1;
}

/* Where the page writes a message (UTF-8) before sm_send, and its room. */
WEB_EXPORT("sm_message") char *sm_message(void) { return web.message; }
WEB_EXPORT("sm_message_cap") uint32_t sm_message_cap(void) { return web.message_cap; }

/* Answers the message in sm_message()[0..len). The tools answer from the
 * page's local time (WebAssembly has no time zone). Returns 0 when nothing
 * is open or the message alone does not fit in the context. */
WEB_EXPORT("sm_send") int sm_send(uint32_t len, int year, int month, int day, int hour, int minute) {
    if (!web.open) return 0;
    if (len > web.message_cap) len = web.message_cap;
    web.message[len] = 0;
    web.clock.year = year;
    web.clock.month = month;
    web.clock.day = day;
    web.clock.hour = hour;
    web.clock.minute = minute;
    return engine_send(&web.engine, web.message, len);
}

WEB_EXPORT("sm_reset") void sm_reset(void) {
    if (web.open) engine_reset(&web.engine);
}

/* [generated tokens, decode seconds, prompt tokens, prefill seconds, threads,
 * context] so far. */
WEB_EXPORT("sm_stats") const double *sm_stats(void) {
    const ChatStats *s = &web.engine.stats;
    web.stats[0] = s->generated;
    web.stats[1] = s->decode_seconds;
    web.stats[2] = s->prompt_tokens;
    web.stats[3] = s->prefill_seconds;
    web.stats[4] = web.pool ? pool_threads(web.pool) : 0;
    web.stats[5] = web.open ? web.model.context : 0;
    return web.stats;
}

WEB_EXPORT("sm_backend") const char *sm_backend(void) { return web.backend ? web.backend : ""; }
