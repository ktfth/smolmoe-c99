/* Tests for the desktop window's model layer (src/desktop.c): how notices
 * read, how tool results are read, which files are refused with a message,
 * and, given a chat model, stopping a reply.
 *
 *   test-desktop                      no model needed
 *   test-desktop MODEL.smoe [DIR]     also the model tests; DIR holds a copy of
 *                                     the bare model and a TinyStories folder */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "desktop.h"
#include "tools.h"

static int failures;

#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fputc('\n', stderr);                                 \
            ++failures;                                          \
        }                                                        \
    } while (0)

#define REPLY_BYTES 8192
#define NOTICES 16
#define NOTICE_BYTES 1024

/* ---- notices and tool results ------------------------------------------- */

static void expect_notice(const char *line, const char *expected) {
    char out[NOTICE_BYTES];
    size_t n = desktop_notice_text(line, out, sizeof(out));
    CHECK(!strcmp(out, expected), "notice %s:\n  got      %s\n  expected %s", line, out, expected);
    CHECK(n == strlen(out), "notice length %zu for %s", n, out);
}

static void test_notices(void) {
    char small[8];
    expect_notice("[ferramenta] calcular: {\"name\": \"calcular\", \"arguments\": {\"expressao\": \"15% de 230\"}} -> "
                  "{\"resumo\": \"O resultado é 34,5.\", \"expressao\": \"15% de 230\", \"resultado\": \"34,5\"}",
                  "ferramenta calcular: O resultado é 34,5.");
    expect_notice("[ferramenta] dias_ate: {...} -> {\"erro\": \"data inválida\"}", "ferramenta dias_ate: erro: data inválida");
    /* Without a summary, or with a result that is not JSON, the line stays. */
    expect_notice("[ferramenta] data_atual: {} -> {\"data\": \"6 de novembro\"}",
                  "[ferramenta] data_atual: {} -> {\"data\": \"6 de novembro\"}");
    expect_notice("[ferramenta] x: y -> {not json}", "[ferramenta] x: y -> {not json}");
    expect_notice("(contexto cheio: conversa reiniciada)", "(contexto cheio: conversa reiniciada)");
    expect_notice("[ferramenta] Calcular: a -> {\"resumo\": \"x\"}", "[ferramenta] Calcular: a -> {\"resumo\": \"x\"}");
    /* The result is the last " -> {": a call may contain the separator. */
    expect_notice("[ferramenta] calcular: {\"a\": \"1 -> {2}\"} -> {\"resumo\": \"três\"}", "ferramenta calcular: três");
    /* Escapes in the summary are decoded. */
    expect_notice("[ferramenta] calcular: c -> {\"resumo\": \"a \\\"b\\\" \\u00e9\"}", "ferramenta calcular: a \"b\" é");
    /* A small buffer is cut, never overrun. */
    CHECK(desktop_notice_text("[ferramenta] calcular: c -> {\"resumo\": \"resultado longo\"}", small, sizeof(small)) == 7,
          "cut notice length");
    CHECK(!strcmp(small, "ferrame"), "cut notice: %s", small);
    /* A cut never splits a character: "ferramenta calcular: é" is 23 bytes, and 22 fit. */
    {
        char cut[23];
        CHECK(desktop_notice_text("[ferramenta] calcular: c -> {\"resumo\": \"é\"}", cut, sizeof(cut)) == 21 &&
                  !strcmp(cut, "ferramenta calcular: "), "cut inside a character: \"%s\"", cut);
        /* "à" takes bytes 21 and 22; 22 bytes fit. */
        CHECK(desktop_notice_text("(conversa reiniciada à força)", cut, sizeof(cut)) == 21 &&
                  !strcmp(cut, "(conversa reiniciada "), "a plain line cut: \"%s\"", cut);
        CHECK(desktop_notice_text("(ok)", cut, 0) == 0, "cap 0");
    }
}

static void test_result_string(void) {
    char out[64];
    CHECK(tool_result_string("{\"resumo\": \"ok\", \"n\": 3}", "resumo", out, sizeof(out)) && !strcmp(out, "ok"), "resumo");
    CHECK(tool_result_string("{\"n\": -1.5e3, \"t\": true, \"resumo\": \"x\"}", "resumo", out, sizeof(out)), "after scalars");
    CHECK(!tool_result_string("{\"n\": 3}", "n", out, sizeof(out)), "a number is not a string");
    CHECK(!tool_result_string("{\"a\": \"b\"}", "resumo", out, sizeof(out)), "absent key");
    CHECK(!tool_result_string("{\"resumo\": \"x\"} trailing", "resumo", out, sizeof(out)), "trailing text");
    CHECK(!tool_result_string("{\"resumo\": {\"x\": 1}}", "resumo", out, sizeof(out)), "nested object");
    CHECK(!tool_result_string("{\"resumo\": \"0123456789\"}", "resumo", out, 4), "too long");
    CHECK(tool_result_string("{\"resumo\": \"a\", \"resumo\": \"b\"}", "resumo", out, sizeof(out)) && !strcmp(out, "a"),
          "the first of repeated keys");
    CHECK(tool_result_string(" {\n\t\"resumo\" :  \"x\" } \n", "resumo", out, sizeof(out)) && !strcmp(out, "x"), "whitespace");
    CHECK(!tool_result_string("{}", "resumo", out, sizeof(out)), "empty object");
    CHECK(!tool_result_string("{\"uma_chave_com_mais_de_trinta_e_um_bytes\": \"x\", \"resumo\": \"y\"}", "resumo", out,
                              sizeof(out)), "a key longer than the parser's room");
    /* A failure never leaves part of a value behind. */
    CHECK(!tool_result_string("{\"resumo\": \"0123456789\"}", "resumo", out, 6) && !out[0], "failure leaves \"%s\"", out);
    CHECK(!tool_result_string("{\"resumo\": \"abc", "resumo", out, sizeof(out)) && !out[0], "unterminated: \"%s\"", out);
}

/* ---- opening ------------------------------------------------------------ */

typedef struct Capture {
    char reply[REPLY_BYTES];
    size_t len;
    char notices[NOTICES][NOTICE_BYTES];
    int notice_count, replies, polls, stop_at_poll, texts, stop_at_text;
} Capture;

static void capture_text(void *ctx, const unsigned char *bytes, uint32_t n) {
    Capture *c = (Capture *)ctx;
    if (c->len + n < sizeof(c->reply)) {
        memcpy(c->reply + c->len, bytes, n);
        c->len += n;
        c->reply[c->len] = 0;
    }
    if (n) ++c->texts;
}

static void capture_reply_end(void *ctx) { ++((Capture *)ctx)->replies; }

static void capture_notice(void *ctx, const char *line) {
    Capture *c = (Capture *)ctx;
    if (c->notice_count < NOTICES) snprintf(c->notices[c->notice_count++], NOTICE_BYTES, "%s", line);
}

/* Stops at the given poll (each generated token), or once that many text pieces arrived. */
static int capture_stop(void *ctx) {
    Capture *c = (Capture *)ctx;
    ++c->polls;
    return (c->stop_at_poll && c->polls >= c->stop_at_poll) || (c->stop_at_text && c->texts >= c->stop_at_text);
}

static DesktopOptions options(const char *path, Capture *c) {
    DesktopOptions o;
    memset(&o, 0, sizeof(o));
    o.path = path;
    o.context = 2048;
    o.sink.ctx = c;
    o.sink.text = capture_text;
    o.sink.reply_end = capture_reply_end;
    o.sink.notice = capture_notice;
    o.sink.stop = capture_stop;
    return o;
}

static void expect_refused(const char *path, const char *fragment) {
    Capture c;
    char err[DESKTOP_ERROR_BYTES];
    DesktopOptions o = options(path, &c);
    Desktop *d = desktop_open(&o, err);
    CHECK(!d, "%s opened", path ? path : "(bundle)");
    CHECK(strstr(err, fragment) != NULL, "%s: error \"%s\" lacks \"%s\"", path ? path : "(bundle)", err, fragment);
    desktop_close(d);
}

/* dir: where a file that is not a model can be written; NULL skips that case. */
static void test_refused(const char *dir) {
    char path[1024];
    FILE *f;
    expect_refused(NULL, "não traz um modelo");
    expect_refused("arquivo-que-nao-existe.smoe", "não consegui abrir o arquivo");
    if (!dir) return;
    snprintf(path, sizeof(path), "%s/nao-modelo.smoe", dir);
    f = fopen(path, "wb");
    CHECK(f != NULL, "cannot write %s", path);
    if (!f) return;
    fputs("isto não é um modelo, só texto com mais de dezesseis bytes", f);
    fclose(f);
    expect_refused(path, "não é um modelo do smolmoe");
    /* The right magic with a version this runtime does not know. */
    for (unsigned version = 0; version <= 4; version += 4) {
        f = fopen(path, "wb");
        if (!f) break;
        fwrite("SMOEQ8", 1, 7, f); /* with its NUL */
        fputc((int)version, f);
        fputs("e mais alguns bytes de enchimento", f);
        fclose(f);
        expect_refused(path, "não é um modelo do smolmoe");
    }
    remove(path);
}

/* ---- a chat model --------------------------------------------------------- */

static Desktop *open_model(const char *path, Capture *c) {
    char err[DESKTOP_ERROR_BYTES];
    DesktopOptions o = options(path, c);
    Desktop *d = desktop_open(&o, err);
    CHECK(d != NULL, "cannot open %s: %s", path, err);
    return d;
}

static void reset_capture(Capture *c, int stop_at_poll, int stop_at_text) {
    memset(c, 0, sizeof(*c));
    c->stop_at_poll = stop_at_poll;
    c->stop_at_text = stop_at_text;
}

static int has_notice(const Capture *c, const char *prefix) {
    int i;
    for (i = 0; i < c->notice_count; ++i)
        if (!strncmp(c->notices[i], prefix, strlen(prefix))) return 1;
    return 0;
}

static void send_text(Desktop *d, const char *text) { desktop_send(d, text, strlen(text)); }

static void test_stop(const char *model) {
    static const char sea[] = "Escreva duas frases sobre o mar.";
    static const char calc[] = "Quanto é 15% de 230?";
    Capture c;
    DesktopInfo info;
    char full[REPLY_BYTES], fresh[REPLY_BYTES];
    Desktop *d = open_model(model, &c);
    if (!d) return;

    reset_capture(&c, 0, 0);
    send_text(d, sea);
    desktop_info(d, &info);
    CHECK(!info.stopped && c.len > 20, "full reply (%zu bytes): %s", c.len, c.reply);
    snprintf(full, sizeof(full), "%s", c.reply);

    /* Stopped in plain text: the reply so far, a prefix of the full one. */
    desktop_reset(d);
    reset_capture(&c, 0, 3);
    send_text(d, sea);
    desktop_info(d, &info);
    CHECK(info.stopped, "the reply was not stopped");
    CHECK(c.len > 0 && c.len < strlen(full) && !strncmp(full, c.reply, c.len - 1),
          "stopped reply is not a prefix:\n  full    %s\n  stopped %s", full, c.reply);

    /* Stopped before the first token: an empty reply, and the conversation goes on. */
    desktop_reset(d);
    reset_capture(&c, 1, 0);
    send_text(d, sea);
    desktop_info(d, &info);
    CHECK(info.stopped && !strcmp(c.reply, "\n"), "stop at the first token: stopped=%d, reply \"%s\"", info.stopped,
          c.reply); /* only the line end every reply gets */
    desktop_reset(d);
    desktop_info(d, &info);
    CHECK(!info.stopped, "a new conversation still reports the old stop");
    reset_capture(&c, 0, 0);
    send_text(d, sea);
    CHECK(!strcmp(c.reply, full), "after a reset the reply changed:\n  expected %s\n  got      %s", full, c.reply);

    /* A message longer than the context's room is refused with a notice, without a reply. */
    {
        static char longer[DESKTOP_DEFAULT_CONTEXT * 8 + 1];
        memset(longer, 'a', sizeof(longer) - 1);
        reset_capture(&c, 0, 0);
        CHECK(!desktop_send(d, longer, sizeof(longer) - 1), "an oversized message was accepted");
        CHECK(has_notice(&c, "(mensagem longa demais") && c.len == 0, "oversized: %d notices, %zu bytes", c.notice_count,
              c.len);
    }

    /* Stopped inside a tool call: the message leaves the history, so asking
     * again answers exactly as a fresh conversation does. (After another
     * exchange the model answers this one without the tool.) */
    desktop_reset(d);
    reset_capture(&c, 0, 0);
    send_text(d, calc);
    CHECK(has_notice(&c, "[ferramenta] calcular"), "the calculation did not call the tool: %s", c.reply);
    snprintf(fresh, sizeof(fresh), "%s", c.reply);

    desktop_reset(d);
    reset_capture(&c, 2, 0); /* the first token opens the call; stop before the second */
    send_text(d, calc);
    desktop_info(d, &info);
    CHECK(info.stopped, "the call was not stopped");
    CHECK(has_notice(&c, "(resposta interrompida durante uma ferramenta"), "no notice for the dropped call (%d notices%s%s)",
          c.notice_count, c.notice_count ? ": " : "", c.notice_count ? c.notices[0] : "");
    CHECK(!has_notice(&c, "[ferramenta]"), "a stopped call ran");
    reset_capture(&c, 0, 0);
    send_text(d, calc);
    CHECK(!strcmp(c.reply, fresh), "history changed by the stopped call:\n  expected %s\n  got      %s", fresh, c.reply);
    desktop_close(d);
}

/* A bare .smoe needs config.ini and a tokenizer beside it; a model without chat is refused. */
static void test_layouts(const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/bare/model.smoe", dir);
    expect_refused(path, "não traz a configuração e o tokenizador");
    snprintf(path, sizeof(path), "%s/stories/model.smoe", dir);
    expect_refused(path, "não conversa");
}

int main(int argc, char **argv) {
    const char *dir = argc > 2 ? argv[2] : NULL;
    test_notices();
    test_result_string();
    test_refused(dir);
    if (argc > 1) test_stop(argv[1]);
    if (argc > 2) test_layouts(dir);
    if (failures) {
        fprintf(stderr, "%d desktop test failure(s)\n", failures);
        return 1;
    }
    puts(argc > 1 ? "desktop tests passed (with a model)" : "desktop tests passed");
    return 0;
}
