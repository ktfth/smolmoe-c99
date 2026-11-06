#include "chat.h"
#include "config.h"
#include "platform.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

#define CLOSE_RESERVE 4u     /* "<|im_end|>\n" after a reply */
#define MAX_CALLS 4u         /* tool calls run from one reply */
#define CALL_BYTES 512u      /* JSON text of one call */
#define RESPONSE_BYTES (MAX_CALLS * (TOOL_RESULT_MAX + 40u))

/* Tool calls captured from one reply instead of being printed. */
typedef struct Calls {
    char text[MAX_CALLS][CALL_BYTES];
    uint32_t len[MAX_CALLS], count, dropped;
    int open, truncated[MAX_CALLS];
} Calls;

static int is_blank(const char *s) {
    for (; *s; ++s)
        if (*s != ' ' && *s != '\t') return 0;
    return 1;
}

static int is_command(const char *line, const char *a, const char *b) {
    return !strcmp(line, a) || !strcmp(line, b);
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
static uint32_t emit(const ChatContext *c, Calls *calls, uint32_t token, unsigned char *last) {
    uint32_t len = tokenizer_decode(c->tok, token, c->piece);
    if (calls->open) {
        uint32_t k = calls->count - 1;
        if (calls->dropped) return 0;
        if (calls->len[k] + len >= CALL_BYTES) {
            calls->truncated[k] = 1;
        } else {
            memcpy(calls->text[k] + calls->len[k], c->piece, len);
            calls->len[k] += len;
        }
        return 0;
    }
    output_bytes(c->out, c->piece, len);
    if (fflush(c->out->f)) die("output flush failed");
    if (len) *last = c->piece[len - 1];
    return len;
}

/* Streams one reply; every generated token is fed back, it is history. */
static uint32_t reply(const ChatContext *c, uint32_t *pos, ChatStats *st, double turn_start, Calls *calls) {
    Model *m = c->model;
    uint32_t i, generated = 0, shown = 0;
    unsigned char last = 0;
    memset(calls, 0, sizeof(*calls));
    for (i = 0; i < c->max_reply; ++i) {
        uint32_t token;
        /* Never an empty reply: with the tool markers banned, the model's next
         * choice after a call it wanted to make was often <|im_end|>. */
        c->banned[EOS_ID] = (unsigned char)(i == 0);
        token = sampler_next(c->sampler, m->logits);
        if (token == EOS_ID) break;
        if (!st->generated && !generated) st->first_token_seconds = now_seconds() - turn_start;
        if (c->tools && token == c->tool_open) open_call(calls);
        else if (c->tools && token == c->tool_close) calls->open = 0;
        else shown += emit(c, calls, token, &last);
        model_forward(m, &token, 1, (*pos)++, LOGITS_LAST);
        ++generated;
    }
    c->banned[EOS_ID] = 0;
    output_finish(c->out);
    /* A reply that called tools ends its line only if it printed an unfinished
     * one; the answer comes in the next reply. */
    if (!calls->count || (shown && last != '\n')) fputc('\n', c->out->f);
    fflush(c->out->f);
    return generated;
}

/* Prefills `n` tokens already in c->ids, streams the reply and closes it. */
static void assistant_turn(const ChatContext *c, uint32_t n, uint32_t *pos, ChatStats *st, Calls *calls) {
    double t0 = now_seconds(), t1;
    model_prefill(c->model, c->ids, n, *pos, 1);
    *pos += n;
    t1 = now_seconds();
    st->generated += reply(c, pos, st, t0, calls);
    st->prefill_seconds += t1 - t0;
    st->decode_seconds += now_seconds() - t1;
    st->prompt_tokens += n;
    n = tokenizer_chat_close(c->tok, c->ids);
    model_prefill(c->model, c->ids, n, *pos, 0);
    *pos += n;
}

/* The call as the model wrote it, on one line, for the notice. */
/* Control bytes from the model (terminal escape sequences) are not echoed. */
static void put_visible(unsigned char ch) { fputc(ch < 0x20 || ch == 0x7f ? '?' : ch, stderr); }

static void print_call(const Calls *calls, uint32_t k) {
    uint32_t i, shown = 0;
    for (i = 0; i < calls->len[k] && shown < 160; ++i) {
        unsigned char ch = (unsigned char)calls->text[k][i];
        if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        if (ch == ' ' && (!shown || i + 1 == calls->len[k])) continue;
        put_visible(ch);
        ++shown;
    }
    if (calls->truncated[k] || i < calls->len[k]) fputs("...", stderr);
}

static void print_name(const char *name) {
    for (; *name; ++name) put_visible((unsigned char)*name);
}

/* Runs the captured calls and encodes their results as one user turn, as the
 * Qwen2.5 template does for consecutive tool messages. Returns its tokens. */
static uint32_t tool_results(const ChatContext *c, const Calls *calls, ChatStats *st) {
    char response[RESPONSE_BYTES], name[TOOL_NAME_MAX], result[TOOL_RESULT_MAX];
    ToolClock now = c->fixed_clock ? *c->fixed_clock : tool_clock_now();
    size_t used = 0;
    uint32_t k;
    for (k = 0; k < calls->count; ++k) {
        int w;
        tool_run(calls->text[k], calls->truncated[k] ? 0 : calls->len[k], c->line, &now, name, result);
        fputs("[ferramenta] ", stderr);
        print_name(name);
        fputs(": ", stderr);
        print_call(calls, k);
        fprintf(stderr, " -> %s\n", result);
        w = snprintf(response + used, sizeof(response) - used, "%s<tool_response>\n%s\n</tool_response>",
                     k ? "\n" : "", result);
        if (w < 0 || (size_t)w >= sizeof(response) - used) die("internal error: tool response overflow");
        used += (size_t)w;
        ++st->tool_calls;
    }
    if (calls->dropped) fprintf(stderr, "(%u chamada(s) extra ignorada(s))\n", calls->dropped);
    return tokenizer_chat_turn(c->tok, "user", (const unsigned char *)response, (uint32_t)used, 1, c->ids);
}

/* Tools are declared once, in the system prompt, but offered per message: the
 * sampler may produce a call only when the message is about dates or time. */
static void offer_tools(const ChatContext *c, int offered) {
    if (!c->tools) return;
    c->banned[c->tool_open] = (unsigned char)!offered;
    c->banned[c->tool_close] = (unsigned char)!offered;
}

/* Whether every captured call fits the user's message (tool_call_grounded);
 * reports those that do not. */
static int all_grounded(const ChatContext *c, const Calls *calls) {
    uint32_t k;
    int grounded = 1;
    for (k = 0; k < calls->count; ++k) {
        if (!calls->truncated[k] && tool_call_grounded(calls->text[k], calls->len[k], c->line)) continue;
        fputs("(chamada de ferramenta descartada: ", stderr);
        print_call(calls, k);
        fputs(")\n", stderr);
        grounded = 0;
    }
    return grounded;
}

/* The first reply to a user message. If any call in it does not fit the
 * message, the whole reply is rolled back (the KV cache is overwritten from
 * `start`, so the history never holds a call without its result) and the model
 * answers again with the tools banned. */
static void first_reply(const ChatContext *c, uint32_t *pos, ChatStats *st, Calls *calls, int len) {
    const unsigned char *line = (const unsigned char *)c->line;
    uint32_t start = *pos, n = tokenizer_chat_turn(c->tok, "user", line, (uint32_t)len, 1, c->ids);
    offer_tools(c, c->tools && tools_relevant(c->line));
    assistant_turn(c, n, pos, st, calls);
    if (!calls->count || all_grounded(c, calls)) return;
    *pos = start;
    offer_tools(c, 0);
    n = tokenizer_chat_turn(c->tok, "user", line, (uint32_t)len, 1, c->ids);
    assistant_turn(c, n, pos, st, calls);
}

/* One tool round: run the calls, then the model answers with the tools banned
 * (a 0.5B model chains calls badly). Returns 0 when the results do not fit. */
static int tool_round(const ChatContext *c, uint32_t *pos, ChatStats *st, Calls *calls) {
    uint32_t n;
    if (!calls->count) return 1;
    n = tool_results(c, calls, st);
    if (*pos + n + c->max_reply + CLOSE_RESERVE > c->model->context) return 0;
    offer_tools(c, 0);
    assistant_turn(c, n, pos, st, calls);
    return 1;
}

/* One user message: the reply and its tool round. When a tool result does not
 * fit, the conversation restarts from the system prompt and the message is
 * answered again, once. */
static void answer(const ChatContext *c, uint32_t system_len, uint32_t *pos, ChatStats *st, int len) {
    Calls calls;
    int fresh = *pos == system_len;
    first_reply(c, pos, st, &calls, len);
    if (tool_round(c, pos, st, &calls)) return;
    if (!fresh) {
        *pos = system_len;
        fputs("(contexto cheio: conversa reiniciada)\n", stderr);
        first_reply(c, pos, st, &calls, len);
        if (tool_round(c, pos, st, &calls)) return;
    }
    /* Do not leave a call without its result in the history. */
    *pos = system_len;
    fputs("(o resultado da ferramenta não cabe no contexto: conversa reiniciada; aumente --context)\n", stderr);
}

void chat_run(const ChatContext *c, ChatStats *st) {
    Model *m = c->model;
    uint32_t pos, system_len, n;
    memset(st, 0, sizeof(*st));
    n = tokenizer_chat_turn(c->tok, "system", (const unsigned char *)c->system, (uint32_t)strlen(c->system), 0,
                            c->ids);
    if (n + CLOSE_RESERVE + 16 > m->context) die("the system prompt does not fit in --context");
    {
        double t0 = now_seconds();
        model_prefill(m, c->ids, n, 0, 0);
        st->prefill_seconds = now_seconds() - t0;
    }
    st->prompt_tokens = system_len = pos = n;
    fprintf(stderr, "smolmoe " SMOLMOE_VERSION " - chat local. Digite sua mensagem; /limpar reinicia, /sair encerra.\n");
    if (c->tools) fputs("Ferramentas: data_atual, dias_ate.\n", stderr);
    for (;;) {
        int len, truncated;
        fputs("\n> ", stderr);
        fflush(stderr);
        len = read_line_utf8(c->line, c->line_cap, &truncated);
        if (len < 0 || is_command(c->line, "/sair", "/exit")) break;
        if (is_blank(c->line)) continue;
        if (truncated)
            fprintf(stderr, "(mensagem cortada em %d bytes; aumente --context para mensagens maiores)\n", len);
        if (is_command(c->line, "/limpar", "/reset")) {
            pos = system_len;
            fputs("(conversa reiniciada)\n", stderr);
            continue;
        }
        n = tokenizer_chat_turn(c->tok, "user", (const unsigned char *)c->line, (uint32_t)len, 1, c->ids);
        if (pos + n + c->max_reply + CLOSE_RESERVE > m->context) {
            if (system_len + n + c->max_reply + CLOSE_RESERVE > m->context) {
                fputs("(mensagem longa demais para o contexto; aumente --context)\n", stderr);
                continue;
            }
            pos = system_len;
            fputs("(contexto cheio: conversa reiniciada)\n", stderr);
        }
        answer(c, system_len, &pos, st, len);
        ++st->turns;
    }
}
