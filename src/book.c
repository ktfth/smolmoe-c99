#include "book.h"
#include "textfold.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct BookTask {
    const char *id;
    const short *phrases;
    const char *powershell; /* example lines, "\n" between them; "" when none */
    const char *posix;
} BookTask;

#include "book_data.inc"

#define STEM_MAX 7 /* six letters and the NUL */
#define PHRASE_END (-1)
#define TASK_END (-2)

/* book_find keeps two tasks: the book must not ask for more. */
typedef char book_tasks_max_check[BOOK_TASKS_MAX == BOOK_FOUND_MAX ? 1 : -1];

static int ends_with(const char *w, size_t n, const char *tail) {
    size_t t = strlen(tail);
    return n >= t && !memcmp(w + n - t, tail, t);
}

void book_stem(const char *word, size_t n, char *out) {
    const char *w = word;
    if (n > 3 && w[n - 1] == 's') --n;
    if (n > 5 && (ends_with(w, n, "ando") || ends_with(w, n, "endo") || ends_with(w, n, "indo"))) n -= 4;
    else if (n > 4 && (ends_with(w, n, "ar") || ends_with(w, n, "er") || ends_with(w, n, "ir") || ends_with(w, n, "am") ||
                       ends_with(w, n, "em")))
        n -= 2;
    if (n > 3 && (w[n - 1] == 'a' || w[n - 1] == 'e' || w[n - 1] == 'o')) --n;
    if (n > STEM_MAX - 1) n = STEM_MAX - 1;
    memcpy(out, w, n);
    out[n] = 0;
}

static int compare_name(const void *key, const void *item) {
    return strcmp((const char *)key, *(const char *const *)item);
}

static int lookup(const char *word, const char *const *list, size_t count) {
    const char *const *p = (const char *const *)bsearch(word, list, count, sizeof(*list), compare_name);
    return p ? (int)(p - list) : -1;
}

/* The message's stems that the book knows, each once, in the order they first
 * appear: the order livro.ts adds their weights in. Returns how many. */
static int message_stems(const char *message, int *stems, int cap) {
    char folded[BOOK_TEXT_MAX], word[BOOK_TEXT_MAX], stem[STEM_MAX];
    const char *p = folded;
    int n = 0;
    text_fold_tail(message, folded, sizeof(folded));
    while (*p) {
        const char *start;
        size_t len;
        int s, i;
        while (*p && !((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9'))) ++p;
        start = p;
        while ((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9')) ++p;
        len = (size_t)(p - start);
        if (len < 2) continue;
        memcpy(word, start, len);
        word[len] = 0;
        if (lookup(word, BOOK_STOP, sizeof(BOOK_STOP) / sizeof(BOOK_STOP[0])) >= 0) continue;
        book_stem(word, len, stem);
        if ((s = lookup(stem, BOOK_STEMS, BOOK_STEM_COUNT)) < 0) continue;
        for (i = 0; i < n && stems[i] != s; ++i) {}
        if (i == n && n < cap) stems[n++] = s;
    }
    return n;
}

static int task_has(const BookTask *t, int stem) {
    const short *p;
    for (p = t->phrases; *p != TASK_END; ++p)
        if (*p == stem) return 1;
    return 0;
}

static int in_message(const int *stems, int n, int stem) {
    int i;
    for (i = 0; i < n; ++i)
        if (stems[i] == stem) return 1;
    return 0;
}

/* The share of the best-covered phrase, by weight. */
static double coverage(const BookTask *t, const int *stems, int n) {
    double best = 0, total = 0, covered = 0;
    const short *p;
    for (p = t->phrases; *p != TASK_END; ++p) {
        if (*p == PHRASE_END) {
            if (total > 0 && covered / total > best) best = covered / total;
            total = covered = 0;
            continue;
        }
        total += BOOK_WEIGHT[*p];
        if (in_message(stems, n, *p)) covered += BOOK_WEIGHT[*p];
    }
    return best;
}

static double score(const BookTask *t, const int *stems, int n) {
    double points = 0, cover;
    int i, shared = 0;
    for (i = 0; i < n; ++i)
        if (task_has(t, stems[i])) {
            points += BOOK_WEIGHT[stems[i]];
            ++shared;
        }
    cover = coverage(t, stems, n);
    /* One shared word is a match only when it is a whole phrase ("docker", "ping"). */
    return shared >= 2 || cover == 1 ? points + BOOK_PHRASE_BONUS * cover : 0;
}

static const char *commands(const BookTask *t, BookShell shell) {
    return shell == BOOK_POWERSHELL ? t->powershell : t->posix;
}

int book_find(const char *message, BookShell shell, int found[BOOK_FOUND_MAX]) {
    int stems[BOOK_STEM_COUNT], n = message_stems(message, stems, BOOK_STEM_COUNT), i, best = -1, second = -1;
    double best_points = 0, second_points = 0;
    for (i = 0; n && i < BOOK_TASK_COUNT; ++i) {
        double points;
        if (!*commands(&BOOK_TASKS[i], shell)) continue;
        points = score(&BOOK_TASKS[i], stems, n);
        if (points < BOOK_MIN_POINTS) continue;
        /* Strictly greater: of equal scores the earlier task comes first, as in a stable sort. */
        if (best < 0 || points > best_points) {
            second = best;
            second_points = best_points;
            best = i;
            best_points = points;
        } else if (second < 0 || points > second_points) {
            second = i;
            second_points = points;
        }
    }
    if (best < 0) return 0;
    found[0] = best;
    if (second < 0 || second_points < BOOK_SECOND * best_points) return 1;
    found[1] = second;
    return 2;
}

int book_task_count(void) { return BOOK_TASK_COUNT; }

const char *book_task_id(int task) { return task >= 0 && task < BOOK_TASK_COUNT ? BOOK_TASKS[task].id : ""; }

size_t book_prefix(const int *found, int n, BookShell shell, char *out, size_t cap) {
    size_t len;
    int i, w;
    if (n <= 0 || !cap) return 0;
    w = snprintf(out, cap, "Comandos reais que costumam resolver isso (%s):\n```%s\n",
                 shell == BOOK_POWERSHELL ? "PowerShell" : "bash", shell == BOOK_POWERSHELL ? "powershell" : "bash");
    if (w < 0 || (size_t)w >= cap) return 0;
    len = (size_t)w;
    for (i = 0; i < n; ++i) {
        w = snprintf(out + len, cap - len, "%s\n", commands(&BOOK_TASKS[found[i]], shell));
        if (w < 0 || (size_t)w >= cap - len) return 0;
        len += (size_t)w;
    }
    w = snprintf(out + len, cap - len, "```\n\n");
    if (w < 0 || (size_t)w >= cap - len) return 0;
    return len + (size_t)w;
}
