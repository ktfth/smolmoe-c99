#include "chat.h"
#include "config.h"
#include "platform.h"
#include "process.h"
#include "util.h"

#include <stdio.h>
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

static void reply_text(void *ctx, const unsigned char *bytes, uint32_t n) {
    Output *out = (Output *)ctx;
    output_bytes(out, bytes, n);
    if (fflush(out->f)) die("output flush failed");
}

static void reply_end(void *ctx) { output_finish((Output *)ctx); }

static void notice(void *ctx, const char *line) {
    (void)ctx;
    fprintf(stderr, "%s\n", line);
}

void chat_run(const ChatContext *c) {
    Engine *e = c->engine;
    e->sink.ctx = c->out;
    e->sink.text = reply_text;
    e->sink.reply_end = reply_end;
    e->sink.notice = notice;
    engine_start(e);
    if (c->terminal) {
        terminal_input_unbuffered();
        bracketed_paste_start();
    }
    fprintf(stderr, "smolmoe " SMOLMOE_VERSION " - chat local. Digite sua mensagem; /limpar reinicia, /sair encerra.\n");
    if (e->tools) fputs("Ferramentas: data_atual, dias_ate, calcular.\n", stderr);
    if (c->shell) fprintf(stderr, "Comandos: !comando roda no %s.\n", c->shell->name);
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
}
