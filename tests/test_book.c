/* Unit tests for the command book (book.h): the C search finds exactly what
 * scripts/cmdcheck/livro.ts finds (tests/goldens/livro-busca.tsv, every eval
 * question and every phrase of the book under both shells), the stem, and the
 * prompt prefix. Run from the repository root, or pass the golden's path. */
#include <stdio.h>
#include <string.h>

#include "book.h"

static int failures;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                      \
            fputc('\n', stderr);                               \
            ++failures;                                        \
        }                                                      \
    } while (0)

static int starts_with(const char *s, const char *head) { return !strncmp(s, head, strlen(head)); }

static void expect_stem(const char *word, const char *want) {
    char got[8];
    book_stem(word, strlen(word), got);
    CHECK(!strcmp(got, want), "stem(%s): got [%s] want [%s]", word, got, want);
}

static void test_stem(void) {
    expect_stem("reiniciar", "reinic");
    expect_stem("reinicie", "reinic");
    expect_stem("servicos", "servic");
    expect_stem("servidor", "servid");
    expect_stem("ocupam", "ocup");
    expect_stem("ocupa", "ocup");
    expect_stem("rodando", "rod");
    expect_stem("ip", "ip");
    expect_stem("dns", "dns");
}

/* "task,task" for what book_find returns. */
static void found_ids(const char *message, BookShell shell, char *out, size_t cap) {
    int found[BOOK_FOUND_MAX], n = book_find(message, shell, found), i;
    out[0] = 0;
    for (i = 0; i < n; ++i) {
        if (i) strncat(out, ",", cap - strlen(out) - 1);
        strncat(out, book_task_id(found[i]), cap - strlen(out) - 1);
    }
}

static void test_golden(const char *path) {
    char line[2048], got[256];
    int cases = 0;
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "cannot open %s", path);
    if (!f) return;
    while (fgets(line, sizeof(line), f)) {
        char *question, *want, *end;
        BookShell shell;
        if (line[0] == '#') continue;
        if ((end = strpbrk(line, "\r\n")) != NULL) *end = 0;
        question = strchr(line, '\t');
        want = question ? strchr(question + 1, '\t') : NULL;
        CHECK(want != NULL, "malformed golden line: %s", line);
        if (!want) continue;
        *question++ = 0;
        *want++ = 0;
        shell = !strcmp(line, "powershell") ? BOOK_POWERSHELL : BOOK_POSIX;
        found_ids(question, shell, got, sizeof(got));
        CHECK(!strcmp(got, want), "%s [%s]: got [%s] want [%s]", line, question, got, want);
        ++cases;
    }
    fclose(f);
    CHECK(cases >= 1000, "only %d golden cases in %s", cases, path);
}

static void test_prefix(void) {
    char out[4096];
    int found[BOOK_FOUND_MAX], n = book_find("Como vejo o espaço livre do disco?", BOOK_POWERSHELL, found);
    size_t len;
    CHECK(n >= 1 && !strcmp(book_task_id(found[0]), "espaco-disco"), "espaco-disco not found first");
    len = book_prefix(found, 1, BOOK_POWERSHELL, out, sizeof(out));
    CHECK(len == strlen(out), "prefix length %u, string %u", (unsigned)len, (unsigned)strlen(out));
    CHECK(starts_with(out, "Comandos reais que costumam resolver isso (PowerShell):\n```powershell\nGet-Volume  # "),
          "prefix start: [%s]", out);
    CHECK(len > 5 && !strcmp(out + len - 5, "```\n\n"), "prefix end: [%s]", out);
    len = book_prefix(found, 1, BOOK_POSIX, out, sizeof(out));
    CHECK(starts_with(out, "Comandos reais que costumam resolver isso (bash):\n```bash\ndf -h  # "), "posix prefix: [%s]",
          out);
    /* No tasks, or no room: nothing. */
    CHECK(book_prefix(found, 0, BOOK_POWERSHELL, out, sizeof(out)) == 0, "prefix without tasks");
    CHECK(book_prefix(found, 1, BOOK_POWERSHELL, out, 40) == 0, "prefix that does not fit");
    /* Off topic and empty messages find nothing. */
    CHECK(book_find("Escreva um poema sobre o mar", BOOK_POWERSHELL, found) == 0, "poem found a task");
    CHECK(book_find("", BOOK_POSIX, found) == 0, "empty message found a task");
    CHECK(!strcmp(book_task_id(-1), "") && !strcmp(book_task_id(100000), ""), "task id out of range");
}

/* Any two tasks fit in BOOK_PREFIX_MAX, under both shells. */
static void test_prefix_room(void) {
    static char out[BOOK_PREFIX_MAX];
    int pair[2], i, j, s;
    for (s = 0; s < 2; ++s)
        for (i = 0; i < book_task_count(); ++i)
            for (j = 0; j < book_task_count(); ++j) {
                pair[0] = i;
                pair[1] = j;
                CHECK(book_prefix(pair, 2, s ? BOOK_POSIX : BOOK_POWERSHELL, out, sizeof(out)) > 0,
                      "%s + %s do not fit", book_task_id(i), book_task_id(j));
            }
}

/* A long message is searched by its end: the question after a pasted log. */
static void test_long_message(void) {
    static char text[8192];
    int found[BOOK_FOUND_MAX], n;
    size_t i;
    for (i = 0; i + 1 < sizeof(text) - 64; ++i) text[i] = "abc xyz "[i % 8];
    strcpy(text + i, " Como vejo o espaço livre do disco?");
    n = book_find(text, BOOK_POSIX, found);
    CHECK(n >= 1 && !strcmp(book_task_id(found[0]), "espaco-disco"), "long message: %d", n);
}

int main(int argc, char **argv) {
    test_stem();
    test_golden(argc > 1 ? argv[1] : "tests/goldens/livro-busca.tsv");
    test_prefix();
    test_prefix_room();
    test_long_message();
    if (failures) {
        fprintf(stderr, "%d book test(s) failed\n", failures);
        return 1;
    }
    puts("book ok");
    return 0;
}
