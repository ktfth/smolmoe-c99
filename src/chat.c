#include "chat.h"
#include "config.h"
#include "platform.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

#define CLOSE_RESERVE 4u /* "<|im_end|>\n" after a reply */

static int is_blank(const char *s) {
    for (; *s; ++s)
        if (*s != ' ' && *s != '\t') return 0;
    return 1;
}

static int is_command(const char *line, const char *a, const char *b) {
    return !strcmp(line, a) || !strcmp(line, b);
}

/* Streams one reply; every generated token is fed back, it is history. */
static uint32_t reply(const ChatContext *c, uint32_t *pos, ChatStats *st, double turn_start) {
    Model *m = c->model;
    uint32_t i, generated = 0;
    for (i = 0; i < c->max_reply; ++i) {
        uint32_t token = sampler_next(c->sampler, m->logits), len;
        if (token == EOS_ID) break;
        len = tokenizer_decode(c->tok, token, c->piece);
        output_bytes(c->out, c->piece, len);
        if (fflush(c->out->f)) die("output flush failed");
        if (!st->generated && !generated) st->first_token_seconds = now_seconds() - turn_start;
        model_forward(m, &token, 1, (*pos)++, LOGITS_LAST);
        ++generated;
    }
    output_finish(c->out);
    fputc('\n', c->out->f);
    fflush(c->out->f);
    return generated;
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
    for (;;) {
        double t0, t1;
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
        t0 = now_seconds();
        model_prefill(m, c->ids, n, pos, 1);
        pos += n;
        t1 = now_seconds();
        st->generated += reply(c, &pos, st, t0);
        st->prefill_seconds += t1 - t0;
        st->decode_seconds += now_seconds() - t1;
        st->prompt_tokens += n;
        ++st->turns;
        n = tokenizer_chat_close(c->tok, c->ids);
        model_prefill(m, c->ids, n, pos, 0);
        pos += n;
    }
}
