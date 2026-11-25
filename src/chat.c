#include "chat.h"
#include "cmdcheck.h"
#include "config.h"
#include "mdterm.h"
#include "papel.h"
#include "platform.h"
#include "process.h"
#include "recipes.h"
#include "tools.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PASTE_MS 100u         /* lines closer than this were pasted together: nobody types that fast */
#define PASTE_FOLLOW_MS 1000u /* a later piece of that paste; longer than PASTE_MS because the reply may already be done */
#define LINE_ROOM 16u         /* the least room worth reading another pasted line into */
#define PASTE_START "\033[200~" /* bracketed paste markers (POSIX terminals, process.h) */
#define PASTE_END "\033[201~"
#define MARKER_LEN 6

/* Inside a bracketed paste: the terminal sent the start marker but not the end
 * one yet. The end can come after the newline of the last pasted line, so the
 * state carries over to the next read. One chat per process. */
static int in_paste;
/* This line's own text follows a paste-end marker (the marker started the line). */
static int after_bracket;
/* The message just read started that way; the next line is a new action. */
static int message_after_bracket;
/* The previous message may still be receiving pieces (not a finished bracketed paste). */
static int follow_paste;
static double last_line_s;

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
    after_bracket = 0;
    while (r < *len) {
        if (!strncmp(line + r, PASTE_START, MARKER_LEN)) {
            in_paste = pasted = 1;
            r += MARKER_LEN;
        } else if (!strncmp(line + r, PASTE_END, MARKER_LEN)) {
            if (!w) { /* the end of a paste that came before this line's own text */
                pasted = 0;
                after_bracket = 1;
            }
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
 *    (the rest of a paste that came in pieces, or typing ahead);
 *  - a line that follows the previous message within PASTE_FOLLOW_MS, when that
 *    message was not a finished bracketed paste. A fast reply shows the prompt
 *    again before the next piece arrives; a person does not type a new command
 *    that soon. A line that begins at the paste-end marker is a new action.
 * Piped input stays one message per line. */
static int read_message(const ChatContext *c, int *truncated, int *pasted) {
    int len, resumed;
    *pasted = c->terminal && input_pending(0);
    len = read_line(c->line, c->line_cap, truncated, pasted);
    message_after_bracket = 0;
    if (len < 0 || !c->terminal) return len;
    resumed = after_bracket;
    message_after_bracket = resumed;
    if (!*pasted && follow_paste && !resumed && (now_seconds() - last_line_s) * 1000.0 < PASTE_FOLLOW_MS)
        *pasted = 1;
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
    last_line_s = now_seconds();
    return len;
}

/* The last recipe shown (recipes.h): "/rodar" runs it. */
static const char *last_recipe;

/* Answers a request that a recipe covers, without the model: the command
 * checked for this shell, and how to run it. Returns 1 when it answered. The
 * model does not see this exchange. */
static int recipe_reply(const ChatContext *c, int truncated) {
    Recipe r;
    char text[2048];
    int n;
    if (!c->shell || truncated || !recipe_find(c->line, c->shell->kind, recipe_host_os(), &r)) return 0;
    n = snprintf(text, sizeof(text), "Comando para ver %s no %s:\n\n    %s\n\nDigite /rodar para executá-lo.\n",
                 r.topic, c->shell->name, r.command);
    if (n < 0 || (size_t)n >= sizeof(text)) return 0;
    output_bytes(c->out, (const unsigned char *)text, (uint32_t)n);
    output_finish(c->out);
    if (fflush(c->out->f)) die("output flush failed");
    last_recipe = r.command;
    return 1;
}

/* "/rodar": runs the last recipe, only when typed (a pasted line runs nothing). */
static void run_recipe(const ChatContext *c, int pasted) {
    if (!c->shell) fputs("(comandos desligados: sem shell para o !)\n", stderr);
    else if (pasted) fputs("(texto colado: /rodar só executa quando é digitado sozinho)\n", stderr);
    else if (!last_recipe) fputs("(nenhum comando sugerido ainda)\n", stderr);
    else {
        fprintf(stderr, "(rodando: %s)\n", last_recipe);
        shell_execute(c->shell, last_recipe, !c->terminal);
    }
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
    MdTerm md;
    CmdChecker *checker; /* NULL: no check */
    char *text;
    size_t len;
    int full; /* a piece did not fit: the rest is dropped too, so the kept text has no gaps */
} Reply;

static void reply_text(void *ctx, const unsigned char *bytes, uint32_t n) {
    Reply *r = (Reply *)ctx;
    mdterm_bytes(&r->md, bytes, n);
    if (fflush(r->md.out->f)) die("output flush failed");
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
    mdterm_finish(&r->md);
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

/* ---- the command book ----------------------------------------------------- */

/* Sends the message, after the command book's examples when it matches a task.
 * `message` has room for line_cap + BOOK_PREFIX_MAX bytes; NULL: no book.
 * Returns what engine_send returns (-1: sm_last_error()). */
static int send_message(const ChatContext *c, char *message, int len) {
    int found[BOOK_FOUND_MAX], n, i;
    size_t prefix;
    if (!message || (n = book_find(c->line, c->book_shell, found)) == 0 ||
        (prefix = book_prefix(found, n, c->book_shell, message, BOOK_PREFIX_MAX)) == 0) {
        return engine_send(c->engine, c->line, (uint32_t)len);
    }
    memcpy(message + prefix, c->line, (size_t)len + 1);
    fputs("(livro de comandos:", stderr);
    for (i = 0; i < n; ++i) fprintf(stderr, "%s %s", i ? "," : "", book_task_id(found[i]));
    fputs(")\n", stderr);
    return engine_send_prefixed(c->engine, message, (uint32_t)(prefix + (size_t)len), (uint32_t)prefix);
}

static void notice(void *ctx, const char *line) {
    (void)ctx;
    fprintf(stderr, "%s\n", line);
}

void chat_run(const ChatContext *c) {
    Engine *e = c->engine;
    PapelLido papel;
    Reply reply;
    char check_err[256], *message = NULL;
    int checking;
    papel_do_ambiente(&papel);
    memset(&reply, 0, sizeof(reply));
    mdterm_init(&reply.md, c->out, c->terminal && stdout_color());
    e->sink.ctx = &reply;
    e->sink.text = reply_text;
    e->sink.reply_end = reply_end;
    e->sink.notice = notice;
    sm_check(engine_start(e));
    checking = start_check(c, &reply, check_err, sizeof(check_err));
    if (c->book && (message = (char *)malloc((size_t)c->line_cap + BOOK_PREFIX_MAX)) == NULL)
        fputs("(livro de comandos desligado: sem memória)\n", stderr);
    if (c->terminal) {
        terminal_input_unbuffered();
        bracketed_paste_start();
    }
    fprintf(stderr, "smolmoe " SMOLMOE_VERSION " - chat local. Digite sua mensagem; /limpar reinicia, /sair encerra.\n");
    if (e->tools) {
        char names[128];
        tools_on_offer(names, sizeof(names));
        fprintf(stderr, "Ferramentas: %s.\n", names);
    }
    if (c->shell) fprintf(stderr, "Comandos: !comando roda no %s; /rodar executa o comando sugerido.\n", c->shell->name);
    if (reply.checker)
        fprintf(stderr,
                "Conferência: os comandos das respostas são procurados no %s desta máquina (nada da resposta roda).\n",
                c->check_shell->name);
    else if (!checking)
        fprintf(stderr, "(conferência dos comandos desligada: %s)\n", check_err);
    if (message)
        fprintf(stderr, "Livro de comandos: perguntas de manutenção levam ao modelo exemplos reais de %s.\n",
                c->book_shell == BOOK_POWERSHELL ? "PowerShell" : "bash");
    papel_anunciar(&papel, stderr_color(), stderr);
    for (;;) {
        int len, truncated, pasted;
        fprintf(stderr, "\n%s > ", papel_marca(papel.papel));
        fflush(stderr);
        len = read_message(c, &truncated, &pasted);
        if (len < 0 || is_command(c->line, "/sair", "/exit")) break;
        if (is_blank(c->line)) {
            if (message_after_bracket) follow_paste = 0; /* a paste-end marker on its own line */
            continue;
        }
        if (truncated)
            fprintf(stderr, "(mensagem cortada em %d bytes; aumente --context para mensagens maiores)\n", len);
        if (is_command(c->line, "/limpar", "/reset")) {
            engine_reset(e);
            last_recipe = NULL;
            follow_paste = 0;
            fputs("(conversa reiniciada)\n", stderr);
            continue;
        }
        if (command_line(c, truncated, pasted)) {
            follow_paste = 0;
            continue;
        }
        if (is_command(c->line, "/rodar", "/run")) {
            run_recipe(c, pasted);
            follow_paste = 0;
            continue;
        }
        /* Pasted text (a log) goes to the model, as it forces no tool: a recipe would answer without reading it. */
        if (!pasted && recipe_reply(c, truncated)) {
            follow_paste = !message_after_bracket;
            continue;
        }
        e->pasted = pasted; /* a pasted log line forces no tool */
        if (send_message(c, message, len) < 0) die("%s", sm_last_error());
        follow_paste = !message_after_bracket;
    }
    cmdcheck_stop(reply.checker);
    free(reply.text);
    free(message);
}
