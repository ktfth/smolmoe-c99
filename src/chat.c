#include "chat.h"
#include "cmdcheck.h"
#include "config.h"
#include "platform.h"
#include "process.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PASTE_MS 100u        /* lines closer than this were pasted together: nobody types that fast */
#define LINE_ROOM 16u        /* the least room worth reading another pasted line into */
#define PASTE_START "\033[200~" /* bracketed paste markers (POSIX terminals, process.h) */
#define PASTE_END "\033[201~"
#define MARKER_LEN 6

/* Inside a bracketed paste: the terminal sent the start marker but not the end
 * one yet. The end can come after the newline of the last pasted line, so the
 * state carries over to the next read. One chat per process. */
static int in_paste;

static int is_blank(const char *s) {
    for (; *s; ++s)
        if (*s != ' ' && *s != '\t') return 0;
    return 1;
}

static int is_command(const char *line, const char *a, const char *b) {
    return !strcmp(line, a) || !strcmp(line, b);
}

/* Removes the paste markers from line[0..*len) and tells whether any of it was
 * pasted: text read inside the markers, or a line that starts or ends a paste. */
static int strip_paste_markers(char *line, int *len) {
    int r = 0, w = 0, pasted = in_paste;
    while (r < *len) {
        if (!strncmp(line + r, PASTE_START, MARKER_LEN)) {
            in_paste = pasted = 1;
            r += MARKER_LEN;
        } else if (!strncmp(line + r, PASTE_END, MARKER_LEN)) {
            if (!w) pasted = 0; /* the end of a paste that came before this line's own text */
            in_paste = 0;
            r += MARKER_LEN;
        } else {
            pasted |= in_paste;
            line[w++] = line[r++];
        }
    }
    line[w] = 0;
    *len = w;
    return pasted || in_paste;
}

static int read_line(char *buf, size_t cap, int *cut, int *pasted) {
    int len = read_line_utf8(buf, cap, cut);
    if (len >= 0 && strip_paste_markers(buf, &len)) *pasted = 1;
    return len;
}

/* One message from the user: a line plus the lines pasted along with it. A
 * pasted "!" line is text for the model, never a command. A paste shows as:
 *  - the terminal's paste markers, where there are any (bracketed paste);
 *  - lines arriving within PASTE_MS of each other: nobody types that fast;
 *  - text already waiting when the prompt shows, so not typed in answer to it
 *    (the rest of a paste that came in pieces, or typing ahead).
 * Piped input stays one message per line. */
static int read_message(const ChatContext *c, int *truncated, int *pasted) {
    int len;
    *pasted = c->terminal && input_pending(0);
    len = read_line(c->line, c->line_cap, truncated, pasted);
    if (len < 0 || !c->terminal) return len;
    while (input_pending(PASTE_MS)) {
        size_t room = c->line_cap - (size_t)len;
        char spill[256];
        int more, cut;
        *pasted = 1;
        if (room < LINE_ROOM) { /* full: the rest of the paste is read and dropped */
            if (read_line(spill, sizeof(spill), &cut, pasted) < 0) break;
            *truncated = 1;
            continue;
        }
        c->line[len] = '\n';
        more = read_line(c->line + len + 1, room - 1, &cut, pasted);
        if (more < 0) {
            c->line[len] = 0;
            break;
        }
        len += 1 + more;
        *truncated |= cut;
    }
    return len;
}

static int has_command_line(const char *text) {
    return text[0] == '!' || strstr(text, "\n!") != NULL;
}

/* A "!comando" line typed on its own runs in the shell. Returns 1 when the
 * message was such a line (run, or refused because it was cut). */
static int command_line(const ChatContext *c, int truncated, int pasted) {
    if (!c->shell) return 0;
    if (pasted) {
        if (has_command_line(c->line))
            fputs("(texto colado: vai para o modelo como uma mensagem só; um comando com ! só roda quando é digitado "
                  "sozinho)\n", stderr);
        return 0;
    }
    if (c->line[0] != '!') return 0;
    if (truncated) fputs("(comando longo demais: não executado)\n", stderr);
    else shell_execute(c->shell, c->line + 1, !c->terminal);
    return 1;
}

/* ---- engine output: replies to stdout, notices to stderr ---------------- */

#define REPLY_KEPT 32768u /* reply bytes kept for the command check; a 512-token reply is far less */

/* The reply as it is written, kept for the command check at its end. */
typedef struct Reply {
    Output *out;
    CmdChecker *checker; /* NULL: no check */
    char *text;
    size_t len;
    int full; /* a piece did not fit: the rest is dropped too, so the kept text has no gaps */
} Reply;

static void reply_text(void *ctx, const unsigned char *bytes, uint32_t n) {
    Reply *r = (Reply *)ctx;
    output_bytes(r->out, bytes, n);
    if (fflush(r->out->f)) die("output flush failed");
    if (!r->checker || r->full) return;
    if (n > REPLY_KEPT - r->len) {
        r->full = 1;
        return;
    }
    memcpy(r->text + r->len, bytes, n);
    r->len += n;
}

static void reply_end(void *ctx) {
    Reply *r = (Reply *)ctx;
    char notice[CMDCHECK_NOTICE_MAX];
    output_finish(r->out);
    if (r->checker && r->len && cmdcheck_reply(r->checker, r->text, r->len, notice, sizeof(notice)))
        fprintf(stderr, "\n%s", notice);
    r->len = 0;
    r->full = 0;
}

/* Starts the check's PowerShell after the system prompt, so it warms up while
 * the user types instead of competing with the prefill, and before the banner:
 * starting it takes a moment, and a line typed after the banner but before the
 * prompt would be read as typed ahead (a paste). Returns 0 with a reason in err. */
static int start_check(const ChatContext *c, Reply *r, char *err, size_t cap) {
    if (!c->check_shell) return 1;
    r->text = (char *)malloc(REPLY_KEPT);
    if (!r->text) {
        snprintf(err, cap, "sem memória");
        return 0;
    }
    if ((r->checker = cmdcheck_start(c->check_shell, 1, err, cap)) != NULL) return 1;
    free(r->text);
    r->text = NULL;
    return 0;
}

static void notice(void *ctx, const char *line) {
    (void)ctx;
    fprintf(stderr, "%s\n", line);
}

void chat_run(const ChatContext *c) {
    Engine *e = c->engine;
    Reply reply;
    char check_err[256];
    int checking;
    memset(&reply, 0, sizeof(reply));
    reply.out = c->out;
    e->sink.ctx = &reply;
    e->sink.text = reply_text;
    e->sink.reply_end = reply_end;
    e->sink.notice = notice;
    engine_start(e);
    checking = start_check(c, &reply, check_err, sizeof(check_err));
    if (c->terminal) {
        terminal_input_unbuffered();
        bracketed_paste_start();
    }
    fprintf(stderr, "smolmoe " SMOLMOE_VERSION " - chat local. Digite sua mensagem; /limpar reinicia, /sair encerra.\n");
    if (e->tools) fputs("Ferramentas: data_atual, dias_ate, calcular, converter, contar.\n", stderr);
    if (c->shell) fprintf(stderr, "Comandos: !comando roda no %s.\n", c->shell->name);
    if (reply.checker)
        fprintf(stderr, "Conferência: os comandos das respostas são procurados no %s desta máquina (nada roda).\n",
                c->check_shell->name);
    else if (!checking)
        fprintf(stderr, "(conferência dos comandos desligada: %s)\n", check_err);
    for (;;) {
        int len, truncated, pasted;
        fputs("\n> ", stderr);
        fflush(stderr);
        len = read_message(c, &truncated, &pasted);
        if (len < 0 || is_command(c->line, "/sair", "/exit")) break;
        if (is_blank(c->line)) continue;
        if (truncated)
            fprintf(stderr, "(mensagem cortada em %d bytes; aumente --context para mensagens maiores)\n", len);
        if (is_command(c->line, "/limpar", "/reset")) {
            engine_reset(e);
            fputs("(conversa reiniciada)\n", stderr);
            continue;
        }
        if (command_line(c, truncated, pasted)) continue;
        engine_send(e, c->line, (uint32_t)len);
    }
    cmdcheck_stop(reply.checker);
    free(reply.text);
}
