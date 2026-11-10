#include "desktop.h"
#include "bundle.h"
#include "config.h"
#include "engine.h"
#include "kernels.h"
#include "modelfile.h"
#include "platform.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DESKTOP_MEMORY_MIB 512u     /* as the terminal chat, plus the KV cache */
#define DESKTOP_BATCH 32u
#define DESKTOP_MAX_THREADS 256u
#define DESKTOP_BYTES_PER_TOKEN 4u  /* room for a message, as the terminal chat */
#define MAGIC "SMOEQ8"
#define MAGIC_BYTES 7u              /* "SMOEQ8", NUL; the version byte follows */
#define MAX_VERSION 3u

struct Desktop {
    Memory mem;
    MappedFile file;
    ModelFile mf;
    Config c;
    Model model;
    Sampler sampler;
    Tokenizer *tok;
    Pool *pool;
    Engine engine;
    const char *backend;
    char *message;
    uint32_t message_cap;
};

static int fail(char *err, const char *fmt, const char *arg) {
    snprintf(err, DESKTOP_ERROR_BYTES, fmt, arg);
    return 0;
}

static int fail_msg(char *err, const char *message) {
    snprintf(err, DESKTOP_ERROR_BYTES, "%s", message ? message : "");
    return 0;
}

/* The container appended to this executable, mapped; 0 when there is none. */
static int map_bundle(MappedFile *file, uint64_t *offset, uint64_t *length) {
    char err[512];
    const char *self = self_executable_path();
    if (!self || !try_map_file(self, file, err, sizeof(err))) return 0;
    if (bundle_find(file->data, file->size, offset, length)) return 1;
    unmap_file(file);
    return 0;
}

int desktop_has_bundle(void) {
    MappedFile file;
    uint64_t offset, length;
    if (!map_bundle(&file, &offset, &length)) return 0;
    unmap_file(&file);
    return 1;
}

/* `name` in the directory of `path`, allocated from mem; NULL when it does not exist. */
static char *beside(Desktop *d, const char *path, const char *name) {
    size_t dir = strlen(path);
    char *joined;
    FILE *f;
    while (dir && path[dir - 1] != '/' && path[dir - 1] != '\\') --dir;
    joined = (char *)mem_alloc(&d->mem, add_size(dir, strlen(name) + 1), 1);
    memcpy(joined, path, dir);
    strcpy(joined + dir, name);
    f = open_file(joined, "rb");
    if (!f) return NULL;
    fclose(f);
    return joined;
}

static const char *resolve_desktop_path(Desktop *d, const char *arg) {
    size_t len;
    if (!arg) return NULL;
    if (!strcmp(arg, "qwen") || !strcmp(arg, "qwen-pt")) return "models/qwen2.5-0.5b-instruct/model.smoe";
    if (!strcmp(arg, "qwen-q5") || !strcmp(arg, "qwen-pt-q5")) return "models/qwen2.5-0.5b-instruct/model-q5.smoe";
    if (!strcmp(arg, "gemma") || !strcmp(arg, "gemma-3") || !strcmp(arg, "gemma-3-1b"))
        return "models/gemma-3-1b-it/model.smoe";
    if (!strcmp(arg, "gemma-270m") || !strcmp(arg, "gemma-3-270m"))
        return "models/gemma-3-270m-it/model.smoe";
    if (!strcmp(arg, "phi") || !strcmp(arg, "phi-4") || !strcmp(arg, "phi-4-mini"))
        return "models/phi-4-mini-instruct/model.smoe";
    if (!strcmp(arg, "stories") || !strcmp(arg, "tinystories")) return "models/tinystories-15m/model.smoe";
    len = strlen(arg);
    if (len < 5 || strcmp(arg + len - 5, ".smoe") != 0) {
        int sep = len && (arg[len - 1] == '/' || arg[len - 1] == '\\');
        const char *suffix = sep ? "model.smoe" : "/model.smoe";
        char *joined = (char *)mem_alloc(&d->mem, add_size(len, strlen(suffix) + 1), 1);
        FILE *f;
        memcpy(joined, arg, len);
        strcpy(joined + len, suffix);
        f = open_file(joined, "rb");
        if (f) {
            fclose(f);
            return joined;
        }
    }
    return arg;
}

/* Maps the model and parses its directory; a file that does not even start
 * like a container is refused here instead of by die(). */
static int map_model(Desktop *d, const char *path, char *err) {
    char why[DESKTOP_ERROR_BYTES - 64]; /* room for the sentence around it */
    uint64_t offset = 0, length;
    const unsigned char *data;
    if (path) {
        if (!try_map_file(path, &d->file, why, sizeof(why))) return fail(err, "não consegui abrir o arquivo (%s)", why);
        length = d->file.size;
    } else if (!map_bundle(&d->file, &offset, &length)) {
        return fail(err, "%s", "este programa não traz um modelo: escolha um arquivo .smoe");
    }
    data = d->file.data + offset;
    if (length < 16 || memcmp(data, MAGIC, MAGIC_BYTES) || !data[MAGIC_BYTES] || data[MAGIC_BYTES] > MAX_VERSION)
        return fail(err, "%s", "o arquivo não é um modelo do smolmoe (.smoe)");
    if (modelfile_parse(&d->mf, data, length, &d->mem) != SM_OK) return fail_msg(err, sm_last_error());
    return 1;
}

/* Config and tokenizer: embedded, or beside a .smoe that has none. */
static int load_metadata(Desktop *d, const char *path, char *err) {
    static const char missing[] = "este .smoe não traz a configuração e o tokenizador: ponha config.ini e "
                                  "tokenizer.qbp (ou .spb) ao lado dele, ou use um modelo autocontido (smolmoe-bundle)";
    const Tensor *config = modelfile_blob(&d->mf, BLOB_CONFIG), *tokenizer = modelfile_blob(&d->mf, BLOB_TOKENIZER);
    char *config_path = NULL, *tokenizer_path = NULL;
    if ((!config || !tokenizer) && path) {
        config_path = config ? NULL : beside(d, path, "config.ini");
        if (!tokenizer && !(tokenizer_path = beside(d, path, "tokenizer.qbp")))
            tokenizer_path = beside(d, path, "tokenizer.spb");
    }
    if ((!config && !config_path) || (!tokenizer && !tokenizer_path)) return fail(err, "%s", missing);
    if (config) {
        if (config_parse(config->data, (size_t)config->bytes, &d->c) != SM_OK) return fail_msg(err, sm_last_error());
    } else if (config_read(config_path, &d->c) != SM_OK) {
        return fail_msg(err, sm_last_error());
    }
    d->tok = tokenizer ? tokenizer_parse(tokenizer->data, (size_t)tokenizer->bytes, d->c.vocab, &d->mem)
                       : tokenizer_load(tokenizer_path, d->c.vocab, &d->mem);
    if (!d->tok) return fail_msg(err, sm_last_error());
    if (!tokenizer_supports_chat(d->tok))
        return fail(err, "%s", "este modelo não conversa (o tokenizador não tem modelo de chat): use um modelo instruído");
    return 1;
}

/* Buffers, sampler and engine: the terminal chat's defaults. 0 leaves the
 * reason in sm_last_error(). */
static int start_engine(Desktop *d, const DesktopOptions *o) {
    Engine *e = &d->engine;
    uint32_t i, ids, needed, open_marker = 0, close_marker = 0, encode_cap;
    unsigned char *banned;
    const char *tools;
    d->message_cap = d->model.context * DESKTOP_BYTES_PER_TOKEN;
    encode_cap = (uint32_t)strlen(CHAT_DEFAULT_SYSTEM) + TOOLS_PROMPT_BYTES;
    ids = tokenizer_prepare(d->tok, encode_cap > d->message_cap ? encode_cap : d->message_cap, &d->mem);
    d->message = (char *)mem_alloc(&d->mem, (size_t)d->message_cap + 1, 1);
    e->ids = (uint32_t *)mem_alloc(&d->mem, ids, sizeof(uint32_t));
    e->piece = (unsigned char *)mem_alloc(&d->mem, tokenizer_max_piece(d->tok), 1);
    banned = (unsigned char *)mem_alloc(&d->mem, d->c.vocab, 1);
    for (i = 0; i < d->c.vocab; ++i) banned[i] = (unsigned char)tokenizer_is_control(d->tok, i);
    tools = engine_tools_prompt(d->tok, CHAT_DEFAULT_SYSTEM, d->model.context, e->ids, &open_marker, &close_marker,
                                &needed, &d->mem);
    if (sampler_init(&d->sampler, d->c.vocab, banned, 0.0f, 40, 0.95f, 1, &d->mem) != SM_OK) return 0;
    e->model = &d->model;
    e->tok = d->tok;
    e->sampler = &d->sampler;
    e->system = tools ? tools : CHAT_DEFAULT_SYSTEM;
    e->max_reply = DESKTOP_REPLY_TOKENS;
    e->tools = tools != NULL;
    e->tool_open = open_marker;
    e->tool_close = close_marker;
    e->banned = banned;
    e->fixed_clock = NULL;
    e->sink = o->sink;
    return engine_start(e) == SM_OK;
}

typedef struct OpenArgs {
    const DesktopOptions *o;
    char *err;
    Desktop *opened;
} OpenArgs;

static void desktop_open_body(OpenArgs *a) {
    Desktop *d = (Desktop *)calloc(1, sizeof(*d));
    const DesktopOptions *o = a->o;
    char *err = a->err;
    const char *path;
    uint32_t context, threads;
    a->opened = d;
    if (!d) {
        fail_msg(err, "allocation failed");
        return;
    }
    d->mem.limit = mul_size(DESKTOP_MEMORY_MIB, 1048576);
    path = resolve_desktop_path(d, o->path);
    if (!map_model(d, path, err) || !load_metadata(d, path, err)) {
        desktop_close(d);
        a->opened = NULL;
        return;
    }
    context = o->context ? o->context : DESKTOP_DEFAULT_CONTEXT;
    if (context > d->c.max_context) context = d->c.max_context;
    d->mem.limit = add_size(d->mem.limit, kv_cache_bytes(&d->c, context));
    threads = o->threads ? o->threads : cpu_physical_cores();
    if (threads > DESKTOP_MAX_THREADS) threads = DESKTOP_MAX_THREADS;

    if (o->stage) o->stage(o->sink.ctx, DESKTOP_CHECKING);
    d->backend = kernels_select(0);
    d->pool = pool_create(threads, &d->mem);
    if (!d->pool) goto fail;
    if (modelfile_verify_payload(&d->mf, d->pool) != SM_OK) goto fail;
    if (o->stage) o->stage(o->sink.ctx, DESKTOP_PREPARING);
    if (model_init(&d->model, &d->c, &d->mf, context, DESKTOP_BATCH < context ? DESKTOP_BATCH : context, 1, d->pool,
                   &d->mem) != SM_OK)
        goto fail;
    if (!start_engine(d, o)) goto fail;
    return;
fail:
    fail_msg(err, sm_last_error());
    a->opened = NULL;
    desktop_close(d);
}

static void desktop_open_protected(void *ctx) { desktop_open_body((OpenArgs *)ctx); }

Desktop *desktop_open(const DesktopOptions *o, char err[DESKTOP_ERROR_BYTES]) {
    OpenArgs a;
    a.o = o;
    a.err = err;
    a.opened = NULL;
    err[0] = 0;
    if (sm_protect(desktop_open_protected, &a) != SM_OK) {
        fail_msg(err, sm_last_error());
        desktop_close(a.opened);
        return NULL;
    }
    return a.opened;
}

/* The longest prefix of text[0..len) that fits in cap bytes without splitting a UTF-8 character. */
static size_t utf8_prefix(const char *text, size_t len, size_t cap) {
    if (len <= cap) return len;
    while (cap && ((unsigned char)text[cap] & 0xc0) == 0x80) --cap;
    return cap;
}

int desktop_send(Desktop *d, const char *text, size_t len) {
    Engine *e = &d->engine;
    size_t n = utf8_prefix(text, len, d->message_cap);
    if (n < len) {
        e->stopped = 0;
        e->sink.notice(e->sink.ctx, "(mensagem longa demais para o contexto; aumente --context)");
        return 0;
    }
    memcpy(d->message, text, n);
    d->message[n] = 0;
    return engine_send(e, d->message, (uint32_t)n);
}

void desktop_reset(Desktop *d) { engine_reset(&d->engine); }

void desktop_info(const Desktop *d, DesktopInfo *out) {
    const ChatStats *s = &d->engine.stats;
    memset(out, 0, sizeof(*out));
    out->backend = d->backend ? d->backend : "";
    out->threads = d->pool ? pool_threads(d->pool) : 0;
    out->context = d->model.context;
    out->tools = d->engine.tools;
    out->stopped = d->engine.stopped;
    out->turns = s->turns;
    out->prompt_tokens = s->prompt_tokens;
    out->generated = s->generated;
    out->tool_calls = s->tool_calls;
    out->prefill_seconds = s->prefill_seconds;
    out->decode_seconds = s->decode_seconds;
}

void desktop_close(Desktop *d) {
    if (!d) return;
    if (d->pool) pool_destroy(d->pool);
    if (d->file.data) unmap_file(&d->file);
    mem_release(&d->mem);
    free(d);
}

/* ---- notices ------------------------------------------------------------- */

#define TOOL_PREFIX "[ferramenta] "
#define RESULT_SEPARATOR " -> {"

/* s into out, cut at a character boundary when it does not fit. */
static size_t copy_text(char *out, size_t cap, const char *s) {
    size_t n;
    if (!cap) return 0;
    n = utf8_prefix(s, strlen(s), cap - 1);
    memcpy(out, s, n);
    out[n] = 0;
    return n;
}

size_t desktop_notice_text(const char *line, char *out, size_t cap) {
    char value[TOOL_RESULT_MAX], name[TOOL_NAME_MAX], text[TOOL_NAME_MAX + TOOL_RESULT_MAX + 32];
    const char *p, *result = NULL, *hit;
    size_t name_len = 0, len = strlen(line);
    if (strncmp(line, TOOL_PREFIX, strlen(TOOL_PREFIX))) return copy_text(out, cap, line);
    p = line + strlen(TOOL_PREFIX);
    while ((p[name_len] >= 'a' && p[name_len] <= 'z') || p[name_len] == '_') ++name_len;
    if (!name_len || name_len >= sizeof(name) || strncmp(p + name_len, ": ", 2) || line[len - 1] != '}')
        return copy_text(out, cap, line);
    memcpy(name, p, name_len);
    name[name_len] = 0;
    /* The result is after the last separator: the call before it may hold one too. */
    for (hit = strstr(p + name_len + 2, RESULT_SEPARATOR); hit; hit = strstr(hit + 1, RESULT_SEPARATOR)) result = hit;
    if (!result) return copy_text(out, cap, line);
    result += strlen(RESULT_SEPARATOR) - 1;
    if (tool_result_string(result, "resumo", value, sizeof(value)) && value[0])
        snprintf(text, sizeof(text), "ferramenta %s: %s", name, value);
    else if (tool_result_string(result, "erro", value, sizeof(value)) && value[0])
        snprintf(text, sizeof(text), "ferramenta %s: erro: %s", name, value);
    else
        return copy_text(out, cap, line);
    return copy_text(out, cap, text);
}
