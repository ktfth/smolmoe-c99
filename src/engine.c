#include "engine.h"
#include "config.h"
#include "platform.h"
#include "sha256.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CLOSE_RESERVE 4u     /* "<|im_end|>\n" after a reply */
#define MAX_CALLS 4u         /* tool calls run from one reply */
#define CALL_BYTES 512u      /* JSON text of one call */
#define CALL_SHOWN 160u      /* bytes of a call shown in a notice */
#define RESPONSE_BYTES (MAX_CALLS * (TOOL_RESULT_MAX + 40u))
#define NOTICE_BYTES 1024u

/* The longest notice ("[ferramenta] name: call... -> result") always fits. */
typedef char notice_fits[13 + TOOL_NAME_MAX + 2 + CALL_SHOWN + 3 + 4 + TOOL_RESULT_MAX < NOTICE_BYTES ? 1 : -1];

/* Tool calls captured from one reply instead of being shown. */
typedef struct Calls {
    char text[MAX_CALLS][CALL_BYTES];
    uint32_t len[MAX_CALLS], count, dropped;
    int open, truncated[MAX_CALLS];
} Calls;

/* A notice being written; text past the end is cut. */
typedef struct Notice {
    char text[NOTICE_BYTES];
    size_t len;
} Notice;

static void notice_put(Notice *n, const char *s) {
    for (; *s && n->len + 1 < sizeof(n->text); ++s) n->text[n->len++] = *s;
    n->text[n->len] = 0;
}

/* Control bytes from the model (terminal escape sequences) are not echoed. */
static void notice_visible(Notice *n, unsigned char ch) {
    char s[2];
    s[0] = (char)(ch < 0x20 || ch == 0x7f ? '?' : ch);
    s[1] = 0;
    notice_put(n, s);
}

static void notice_name(Notice *n, const char *name) {
    for (; *name; ++name) notice_visible(n, (unsigned char)*name);
}

/* The call as the model wrote it, on one line. */
static void notice_call(Notice *n, const Calls *calls, uint32_t k) {
    uint32_t i, shown = 0;
    for (i = 0; i < calls->len[k] && shown < CALL_SHOWN; ++i) {
        unsigned char ch = (unsigned char)calls->text[k][i];
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        if (ch == ' ' && (!shown || i + 1 == calls->len[k])) continue;
        notice_visible(n, ch);
        ++shown;
    }
    if (calls->truncated[k] || i < calls->len[k]) notice_put(n, "...");
}

static void say(const Engine *e, const char *line) { e->sink.notice(e->sink.ctx, line); }

/* The turn log is the token ids after the system prompt. Replaying it from
 * system_len rebuilds the KV cache with RoPE at the new positions; the system
 * prompt itself stays where engine_start put it. */
static void remember(Engine *e, const uint32_t *ids, uint32_t n) {
    if (!n) return;
    if (e->hist_len + n > e->hist_cap) die("internal error: history overflow");
    memcpy(e->hist + e->hist_len, ids, n * sizeof(uint32_t));
    e->hist_len += n;
}

static uint32_t committed(const Engine *e) {
    uint32_t i, n = 0;
    for (i = 0; i < e->nturns; ++i) n += e->turns[i];
    return n;
}

static void close_turn(Engine *e) {
    uint32_t done = committed(e);
    if (e->hist_len < done || e->nturns >= e->hist_cap) die("internal error: turn log");
    e->turns[e->nturns++] = e->hist_len - done;
}

/* keep is a turn boundary: the prefix of turns that sums to it. */
static void trim_hist(Engine *e, uint32_t keep) {
    uint32_t n = 0, i;
    if (keep > e->hist_len) die("internal error: trim past the history");
    for (i = 0; i < e->nturns; ++i) {
        if (n + e->turns[i] > keep) break;
        n += e->turns[i];
    }
    if (n != keep) die("internal error: trim is not on a turn boundary");
    e->nturns = i;
    e->hist_len = keep;
    e->pos = e->system_len + keep;
}

static void open_call(Calls *calls) {
    calls->open = 1;
    if (calls->count < MAX_CALLS) {
        calls->len[calls->count] = 0;
        calls->truncated[calls->count] = 0;
        ++calls->count;
    } else {
        ++calls->dropped;
    }
}

/* Decoded bytes of one token: into the open call, or to the user. Returns the
 * bytes shown, and the last one in *last. */
static uint32_t emit(const Engine *e, Calls *calls, uint32_t token, unsigned char *last) {
    uint32_t len = tokenizer_decode(e->tok, token, e->piece);
    if (calls->open) {
        uint32_t k = calls->count - 1;
        if (calls->dropped) return 0;
        if (calls->len[k] + len >= CALL_BYTES) {
            calls->truncated[k] = 1;
        } else {
            memcpy(calls->text[k] + calls->len[k], e->piece, len);
            calls->len[k] += len;
        }
        return 0;
    }
    e->sink.text(e->sink.ctx, e->piece, len);
    if (len) *last = e->piece[len - 1];
    return len;
}

static const char *chosen_tool(const char *message) {
    const char *tool = tools_forced_choice(message);
    return tool ? tool : tools_date_choice(message);
}

/* Streams one reply; every generated token is fed back, it is history. With a
 * prefix, the reply starts inside a call the runtime opened (forced choice):
 * the prefix is the call text already in the context. */
static uint32_t reply(Engine *e, double turn_start, Calls *calls, const char *prefix, uint32_t prefix_len) {
    Model *m = e->model;
    uint32_t i, generated = 0, shown = 0;
    unsigned char last = 0;
    int aborted = 0;
    memset(calls, 0, sizeof(*calls));
    if (prefix) {
        open_call(calls);
        memcpy(calls->text[0], prefix, prefix_len);
        calls->len[0] = prefix_len;
    }
    for (i = 0; i < e->max_reply; ++i) {
        uint32_t token;
        /* A stop before the first token leaves an empty reply: still a valid turn. */
        if (e->sink.stop && e->sink.stop(e->sink.ctx)) {
            e->stopped = 1;
            break;
        }
        /* Never an empty reply: with the tool markers banned, the model's next
         * choice after a call it wanted to make was often <|im_end|>. */
        e->banned[EOS_ID] = (unsigned char)(i == 0);
        token = (i == 0 && prefix_len >= 2 && prefix[prefix_len - 1] == '}' && prefix[prefix_len - 2] == '}')
              ? e->tool_close
              : sampler_next(e->sampler, m->logits);
        if (i == 0 && !prefix && e->tools) {
            if (!e->banned[e->tool_open] && token != e->tool_open && chosen_tool(e->message)) {
                aborted = 1;
                break;
            }
            if (e->banned[e->tool_open]) {
                /* When tools are banned, prevent a raw JSON/XML/codeblock leak ('{', '[', '<', '`') on the first token. */
                uint32_t banned_toks[8], nb = 0, k;
                while (nb < 8 && token != EOS_ID && tokenizer_decode(e->tok, token, e->piece) > 0 &&
                       (e->piece[0] == '{' || e->piece[0] == '[' || e->piece[0] == '<' || e->piece[0] == '`')) {
                    banned_toks[nb++] = token;
                    e->banned[token] = 1;
                    token = sampler_next(e->sampler, m->logits);
                }
                for (k = 0; k < nb; ++k) e->banned[banned_toks[k]] = 0;
            }
        }
        if (token == EOS_ID) break;
        if (!e->stats.generated && !generated) e->stats.first_token_seconds = now_seconds() - turn_start;
        if (e->tools && token == e->tool_open) open_call(calls);
        else if (e->tools && token == e->tool_close) calls->open = 0;
        else shown += emit(e, calls, token, &last);
        if (prefix && calls->open && !calls->truncated[0] && calls->len[0] > prefix_len &&
            calls->text[0][calls->len[0] - 1] == '}' &&
            tool_call_grounded(calls->text[0], calls->len[0], e->message)) {
            calls->open = 0;
        }
        model_forward(m, &token, 1, e->pos, LOGITS_LAST);
        remember(e, &token, 1);
        ++e->pos;
        ++generated;
        if (prefix && !calls->open) break;
    }
    e->banned[EOS_ID] = 0;
    if (aborted) return generated;
    /* A reply that called tools ends its line only if it showed an unfinished
     * one; the answer comes in the next reply. */
    if (!calls->count || (shown && last != '\n')) e->sink.text(e->sink.ctx, (const unsigned char *)"\n", 1);
    e->sink.reply_end(e->sink.ctx);
    return generated;
}

/* Prefills `n` tokens already in e->ids, streams the reply and closes it. */
static void assistant_turn(Engine *e, uint32_t n, Calls *calls, const char *prefix, uint32_t prefix_len) {
    double t0 = now_seconds(), t1;
    model_prefill(e->model, e->ids, n, e->pos, 1);
    remember(e, e->ids, n);
    e->pos += n;
    t1 = now_seconds();
    e->stats.generated += reply(e, t0, calls, prefix, prefix_len);
    e->stats.prefill_seconds += t1 - t0;
    e->stats.decode_seconds += now_seconds() - t1;
    e->stats.prompt_tokens += n;
    n = tokenizer_chat_close(e->tok, e->ids);
    model_prefill(e->model, e->ids, n, e->pos, 0);
    remember(e, e->ids, n);
    e->pos += n;
    close_turn(e);
}

/* Runs the captured calls and encodes their results as one user turn, as the
 * Qwen2.5 template does for consecutive tool messages. Returns its tokens. */
static uint32_t tool_results(Engine *e, const Calls *calls) {
    char response[RESPONSE_BYTES], name[TOOL_NAME_MAX], result[TOOL_RESULT_MAX];
    ToolClock now = e->fixed_clock ? *e->fixed_clock : tool_clock_now();
    size_t used = 0;
    uint32_t k;
    if (e->model->c.gemma3) {
        int w = snprintf(response, sizeof(response), "%s", e->message);
        if (w < 0 || (size_t)w >= sizeof(response)) die("internal error: tool response overflow");
        used = (size_t)w;
    }
    for (k = 0; k < calls->count; ++k) {
        Notice n;
        int w;
        tool_run(calls->text[k], calls->truncated[k] ? 0 : calls->len[k], e->message, &now, name, result);
        n.len = 0;
        notice_put(&n, "[ferramenta] ");
        notice_name(&n, name);
        notice_put(&n, ": ");
        notice_call(&n, calls, k);
        notice_put(&n, " -> ");
        notice_put(&n, result);
        say(e, n.text);
        if (e->model->c.gemma3) {
            char summary[TOOL_RESULT_MAX];
            if (!tool_result_string(result, "resumo", summary, sizeof(summary)) &&
                !tool_result_string(result, "erro", summary, sizeof(summary)))
                snprintf(summary, sizeof(summary), "%s", result);
            w = snprintf(response + used, sizeof(response) - used,
                         "\n(Informação exata: %s Responda em português em uma frase curta.)", summary);
        } else {
            w = snprintf(response + used, sizeof(response) - used, "%s<tool_response>\n%s\n</tool_response>",
                         k ? "\n" : "", result);
        }
        if (w < 0 || (size_t)w >= sizeof(response) - used) die("internal error: tool response overflow");
        used += (size_t)w;
        ++e->stats.tool_calls;
    }
    if (calls->dropped) {
        char line[64];
        snprintf(line, sizeof(line), "(%u chamada(s) extra ignorada(s))", calls->dropped);
        say(e, line);
    }
    return tokenizer_chat_turn(e->tok, "user", (const unsigned char *)response, (uint32_t)used, 1, e->ids);
}

/* Tools are declared once, in the system prompt, but offered per message: the
 * sampler may produce a call only when the message is about dates or time. */
static void offer_tools(const Engine *e, int offered) {
    if (!e->tools) return;
    e->banned[e->tool_open] = (unsigned char)!offered;
    e->banned[e->tool_close] = (unsigned char)!offered;
}

/* Whether every captured call fits the user's message (tool_call_grounded);
 * reports those that do not. */
static int all_grounded(const Engine *e, const Calls *calls) {
    uint32_t k;
    int grounded = 1;
    for (k = 0; k < calls->count; ++k) {
        Notice n;
        if (!calls->truncated[k] && tool_call_grounded(calls->text[k], calls->len[k], e->message)) continue;
        n.len = 0;
        notice_put(&n, "(chamada de ferramenta descartada: ");
        notice_call(&n, calls, k);
        notice_put(&n, ")");
        say(e, n.text);
        grounded = 0;
    }
    return grounded;
}

static int fits_at(const Engine *e, uint32_t pos, uint32_t n) {
    uint64_t need = (uint64_t)pos + n + e->max_reply + CLOSE_RESERVE;
    return need <= e->model->context;
}

/* Drop whole turns that end at or before e->anchor until n tokens fit, then
 * replay what remains. The system prompt's KV is left in place. */
static void replay(Engine *e) {
    double t0 = now_seconds();
    e->pos = e->system_len;
    if (e->hist_len) {
        model_prefill(e->model, e->hist, e->hist_len, e->pos, 0);
        e->pos += e->hist_len;
    }
    e->stats.prefill_seconds += now_seconds() - t0;
    e->stats.prompt_tokens += e->hist_len;
}

static int ensure_room(Engine *e, uint32_t n) {
    uint32_t drop = 0, skip = 0;
    if (e->pos != e->system_len + e->hist_len) die("internal error: history and KV diverged");
    if (fits_at(e, e->pos, n)) return 1;
    if (e->anchor > e->hist_len) die("internal error: anchor past the history");
    while (drop < e->nturns && skip + e->turns[drop] <= e->anchor &&
           !fits_at(e, e->system_len + (e->hist_len - skip), n)) {
        skip += e->turns[drop++];
    }
    if (!fits_at(e, e->system_len + (e->hist_len - skip), n)) return 0;
    if (!drop) return 1;
    memmove(e->hist, e->hist + skip, (e->hist_len - skip) * sizeof(uint32_t));
    memmove(e->turns, e->turns + drop, (e->nturns - drop) * sizeof(uint32_t));
    e->hist_len -= skip;
    e->nturns -= drop;
    e->anchor -= skip;
    replay(e);
    say(e, "(contexto cheio: mensagens antigas saíram da conversa)");
    return 1;
}

/* The message clearly needs one tool (tools_forced_choice) but the model
 * picked another: the runtime writes <tool_call> and the call up to the
 * argument, and the model only writes the argument. Returns 0 when there is
 * no forced choice or no room for it. */
static int forced_reply(Engine *e, Calls *calls, uint32_t len) {
    char prefix[CALL_BYTES];
    const char *tool = chosen_tool(e->message);
    Notice note;
    uint32_t n, plen;
    if (!tool || !(plen = (uint32_t)tools_call_prefix(tool, prefix, sizeof(prefix)))) return 0;
    n = tokenizer_chat_turn(e->tok, "user", (const unsigned char *)e->message, len, 1, e->ids);
    e->ids[n++] = e->tool_open;
    n += tokenizer_encode(e->tok, (const unsigned char *)prefix, plen, e->ids + n, 0, NULL);
    if (!ensure_room(e, n)) return 0;
    note.len = 0;
    notice_put(&note, "(chamada refeita com ");
    notice_name(&note, tool);
    notice_put(&note, ")");
    say(e, note.text);
    offer_tools(e, 1);
    assistant_turn(e, n, calls, prefix, plen);
    return 1;
}

/* A reply stopped inside a tool call leaves the history without this message;
 * one stopped in plain text stays as far as it got. */
static void drop_stopped_call(Engine *e, Calls *calls) {
    if (!calls->count) return;
    calls->count = 0;
    trim_hist(e, e->anchor);
    say(e, "(resposta interrompida durante uma ferramenta: a mensagem saiu da conversa)");
}

/* The first reply to a user message. If any call in it does not fit the
 * message, the whole reply is rolled back to the anchor (the history never
 * holds a call without its result). Then, when the message clearly needs one
 * tool, that call is opened for the model; otherwise, or if that call does not
 * fit either, the model answers again with the tools banned. */
static void first_reply(Engine *e, Calls *calls, uint32_t len) {
    const unsigned char *text = (const unsigned char *)e->message;
    uint32_t n = tokenizer_chat_turn(e->tok, "user", text, len, 1, e->ids);
    offer_tools(e, e->tools && tools_relevant(e->message));
    assistant_turn(e, n, calls, NULL, 0);
    if (e->stopped) {
        drop_stopped_call(e, calls);
        return;
    }
    if (calls->count ? all_grounded(e, calls) : (!e->tools || !chosen_tool(e->message))) return;
    trim_hist(e, e->anchor);
    if (forced_reply(e, calls, len)) {
        if (e->stopped) {
            drop_stopped_call(e, calls);
            return;
        }
        if (calls->count && all_grounded(e, calls)) return;
        trim_hist(e, e->anchor);
    }
    offer_tools(e, 0);
    n = tokenizer_chat_turn(e->tok, "user", text, len, 1, e->ids);
    assistant_turn(e, n, calls, NULL, 0);
}

/* One tool round: run the calls, then the model answers with the tools banned
 * (a 0.5B model chains calls badly). Returns 0 when the results do not fit. */
static int tool_round(Engine *e, Calls *calls) {
    uint32_t n;
    if (!calls->count) return 1;
    if (e->model->c.gemma3) trim_hist(e, e->anchor);
    n = tool_results(e, calls);
    if (!ensure_room(e, n)) return 0;
    offer_tools(e, 0);
    assistant_turn(e, n, calls, NULL, 0);
    return 1;
}

/* One user message: the reply and its tool round. A tool result that still
 * does not fit after the older turns are dropped leaves the history without
 * that attempt. */
static void answer(Engine *e, uint32_t len) {
    Calls calls;
    first_reply(e, &calls, len);
    if (e->stopped || tool_round(e, &calls)) return;
    trim_hist(e, e->anchor);
    say(e, "(o resultado da ferramenta não cabe no contexto; aumente --context)");
}

const char *engine_tools_prompt(Tokenizer *tok, const char *system, uint32_t context, uint32_t *ids,
                                uint32_t *tool_open, uint32_t *tool_close, uint32_t *needed, Memory *mem) {
    size_t cap = strlen(system) + TOOLS_PROMPT_BYTES;
    char *text;
    uint32_t n;
    *needed = 0;
    if (!tokenizer_tool_markers(tok, tool_open, tool_close)) return NULL;
    text = (char *)mem_alloc(mem, cap, 1);
    if (!tools_system_prompt(system, text, cap)) die("internal error: tool declarations do not fit");
    n = tokenizer_chat_turn(tok, "system", (const unsigned char *)text, (uint32_t)strlen(text), 0, ids);
    if (n > context / 2) {
        *needed = 2 * n;
        return NULL;
    }
    return text;
}

/* Bump when a stored system-prompt prefix would no longer match a fresh prefill. */
#define SYSTEM_KV_EPOCH "smkv1"
#define CACHE_PATH 4096u

static void hash_u32(Sha256 *s, uint32_t x) {
    unsigned char b[4];
    b[0] = (unsigned char)x;
    b[1] = (unsigned char)(x >> 8);
    b[2] = (unsigned char)(x >> 16);
    b[3] = (unsigned char)(x >> 24);
    sha256_update(s, b, 4);
}

static void hash_mem(Sha256 *s, const void *p, size_t n) { sha256_update(s, p, n); }

static void to_hex(const unsigned char d[32], char out[65]) {
    static const char h[] = "0123456789abcdef";
    uint32_t i;
    for (i = 0; i < 32; ++i) {
        out[2 * i] = h[d[i] >> 4];
        out[2 * i + 1] = h[d[i] & 15];
    }
    out[64] = 0;
}

static int write_u32(FILE *f, uint32_t x) {
    unsigned char b[4] = {(unsigned char)x, (unsigned char)(x >> 8), (unsigned char)(x >> 16), (unsigned char)(x >> 24)};
    return fwrite(b, 1, 4, f) == 4;
}

static int read_u32(FILE *f, uint32_t *x) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    *x = (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24;
    return 1;
}

/* 0 when the cache is off or the directory cannot be created. */
static int kv_directory(char *buf, size_t cap) {
    const char *env = getenv("SMOLMOE_KV_CACHE");
    size_t n;
    if (env && env[0] == '0' && !env[1]) return 0;
    if (env && env[0]) {
        if (strlen(env) + 1 > cap) return 0;
        memcpy(buf, env, strlen(env) + 1);
        return ensure_directory(buf);
    }
    if (!user_cache_dir(buf, cap)) return 0;
    n = strlen(buf);
    if (n + 9 >= cap) return 0;
    memcpy(buf + n, "/smolmoe", 9);
    return ensure_directory(buf);
}

static int system_kv_key(const Engine *e, const uint32_t *ids, uint32_t n, unsigned char out[32]) {
    const Config *c = &e->model->c;
    const Model *m = e->model;
    Sha256 s;
    unsigned char dtype = (unsigned char)m->kv_f16;
    uint32_t i;
    if (!m->checkpoint || m->checkpoint_bytes > SIZE_MAX) return 0;
    sha256_init(&s);
    sha256_update(&s, SYSTEM_KV_EPOCH, sizeof(SYSTEM_KV_EPOCH) - 1);
    sha256_update(&s, SMOLMOE_VERSION, strlen(SMOLMOE_VERSION));
    sha256_update(&s, &dtype, 1);
    hash_u32(&s, c->dim);
    hash_u32(&s, c->hidden);
    hash_u32(&s, c->layers);
    hash_u32(&s, c->heads);
    hash_u32(&s, c->kv_heads);
    hash_u32(&s, c->experts);
    hash_u32(&s, c->top_k);
    hash_u32(&s, c->vocab);
    hash_u32(&s, c->shared);
    hash_u32(&s, c->qkv_bias);
    hash_u32(&s, c->gemma3);
    hash_u32(&s, c->head_dim);
    hash_u32(&s, c->rotary_dim);
    hash_u32(&s, c->sliding_window);
    hash_mem(&s, &c->eps, sizeof(c->eps));
    hash_mem(&s, &c->theta, sizeof(c->theta));
    hash_mem(&s, &c->local_theta, sizeof(c->local_theta));
    hash_mem(&s, &c->rope_factor, sizeof(c->rope_factor));
    hash_u32(&s, n);
    for (i = 0; i < n; ++i) hash_u32(&s, ids[i]);
    sha256_update(&s, m->checkpoint, (size_t)m->checkpoint_bytes);
    sha256_final(&s, out);
    return 1;
}

static void hash_prefix(Model *m, uint32_t ntok, Sha256 *s) {
    uint32_t layer;
    size_t bytes = mul_size(mul_size(ntok, m->kv_dim), m->kv_f16 ? 2u : 4u);
    for (layer = 0; layer < m->c.layers; ++layer) {
        sha256_update(s, model_kv_layer(m, layer, 0), bytes);
        sha256_update(s, model_kv_layer(m, layer, 1), bytes);
    }
}

static const unsigned char KV_MAGIC[8] = {'S', 'M', 'K', 'V', 'C', '1', 0, 1};

/* Loads positions [0, n) when the file matches. A short or corrupt file leaves
 * those positions dirty; the caller prefills them. Returns 1 on a hit. */
static int load_system_kv(Engine *e, const char *path, uint32_t n) {
    Model *m = e->model;
    unsigned char got[32], expect[32], magic[8];
    uint32_t kn, layers, kv_dim, elem, layer;
    size_t bytes;
    Sha256 s;
    FILE *f;
    int ok = 0;
    f = open_file(path, "rb");
    if (!f) return 0;
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, KV_MAGIC, 8) || !read_u32(f, &kn) || !read_u32(f, &layers) ||
        !read_u32(f, &kv_dim) || !read_u32(f, &elem))
        goto done;
    if (kn != n || layers != m->c.layers || kv_dim != m->kv_dim || elem != (m->kv_f16 ? 2u : 4u)) goto done;
    bytes = mul_size(mul_size(n, m->kv_dim), elem);
    for (layer = 0; layer < layers; ++layer) {
        if (fread(model_kv_layer(m, layer, 0), 1, bytes, f) != bytes ||
            fread(model_kv_layer(m, layer, 1), 1, bytes, f) != bytes)
            goto done;
    }
    if (fread(got, 1, 32, f) != 32) goto done;
    sha256_init(&s);
    hash_prefix(m, n, &s);
    sha256_final(&s, expect);
    ok = memcmp(got, expect, 32) == 0;
done:
    fclose(f);
    return ok;
}

static void save_system_kv(Engine *e, const char *dir, const unsigned char key[32], uint32_t n) {
    Model *m = e->model;
    char final_path[CACHE_PATH], tmp[CACHE_PATH], hex[65];
    unsigned char digest[32];
    uint32_t layer;
    size_t bytes;
    Sha256 s;
    FILE *f;
    int wrote, npath;
    to_hex(key, hex);
    npath = snprintf(final_path, sizeof(final_path), "%s/%s", dir, hex);
    if (npath < 0 || npath >= (int)sizeof(final_path)) return;
    npath = snprintf(tmp, sizeof(tmp), "%s.%u.tmp", final_path, process_id());
    if (npath < 0 || npath >= (int)sizeof(tmp)) return;
    f = open_file(tmp, "wb");
    if (!f) return;
    bytes = mul_size(mul_size(n, m->kv_dim), m->kv_f16 ? 2u : 4u);
    wrote = fwrite(KV_MAGIC, 1, 8, f) == 8 && write_u32(f, n) && write_u32(f, m->c.layers) && write_u32(f, m->kv_dim) &&
            write_u32(f, m->kv_f16 ? 2u : 4u);
    for (layer = 0; wrote && layer < m->c.layers; ++layer)
        wrote = fwrite(model_kv_layer(m, layer, 0), 1, bytes, f) == bytes &&
                fwrite(model_kv_layer(m, layer, 1), 1, bytes, f) == bytes;
    sha256_init(&s);
    hash_prefix(m, n, &s);
    sha256_final(&s, digest);
    wrote = wrote && fwrite(digest, 1, 32, f) == 32;
    if (fclose(f) || !wrote) {
        remove(tmp);
        return;
    }
    if (!replace_file(tmp, final_path)) remove(tmp);
}

void engine_start(Engine *e) {
    const char *tools_hdr = e->model->c.gemma3 ? strstr(e->system, "\n\n# Tools") : NULL;
    uint32_t sys_len = tools_hdr ? (uint32_t)(tools_hdr - e->system) : (uint32_t)strlen(e->system);
    uint32_t n = tokenizer_chat_turn(e->tok, "system", (const unsigned char *)e->system, sys_len, 0, e->ids);
    char dir[CACHE_PATH];
    unsigned char key[32];
    int cached = kv_directory(dir, sizeof(dir));
    int keyed = 0;
    double t0;
    if (!e->mem) die("internal error: engine has no heap");
    memset(&e->stats, 0, sizeof(e->stats));
    if (!e->hist) {
        e->hist_cap = e->model->context;
        e->hist = (uint32_t *)mem_alloc(e->mem, e->hist_cap, sizeof(uint32_t));
        e->turns = (uint32_t *)mem_alloc(e->mem, e->hist_cap, sizeof(uint32_t));
    }
    e->hist_len = e->nturns = e->anchor = 0;
    if (n + CLOSE_RESERVE + 16 > e->model->context) die("the system prompt does not fit in --context");
    keyed = cached && system_kv_key(e, e->ids, n, key);
    if (keyed) {
        char path[CACHE_PATH], hex[65];
        int npath;
        to_hex(key, hex);
        npath = snprintf(path, sizeof(path), "%s/%s", dir, hex);
        if (npath > 0 && npath < (int)sizeof(path) && load_system_kv(e, path, n)) e->stats.system_kv = 2;
    }
    if (e->stats.system_kv != 2) {
        t0 = now_seconds();
        model_prefill(e->model, e->ids, n, 0, 0);
        e->stats.prefill_seconds = now_seconds() - t0;
        e->stats.prompt_tokens = n;
        if (keyed) {
            e->stats.system_kv = 1;
            save_system_kv(e, dir, key, n);
        }
    }
    e->system_len = e->pos = n;
    e->message = NULL;
}

int engine_send(Engine *e, const char *text, uint32_t len) {
    uint32_t n = tokenizer_chat_turn(e->tok, "user", (const unsigned char *)text, len, 1, e->ids);
    e->stopped = 0;
    /* Anything already committed may leave; the message about to be added may not. */
    e->anchor = e->hist_len;
    if (!ensure_room(e, n)) {
        say(e, "(mensagem longa demais para o contexto; aumente --context)");
        return 0;
    }
    e->message = text;
    answer(e, len);
    e->message = NULL;
    ++e->stats.turns;
    return 1;
}

void engine_reset(Engine *e) {
    e->pos = e->system_len;
    e->hist_len = 0;
    e->nturns = 0;
    e->anchor = 0;
    e->stopped = 0;
}
