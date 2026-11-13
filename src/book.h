/* The command book: maintenance tasks in Portuguese, each with real commands
 * for PowerShell and for bash (scripts/cmdcheck/livro.json, carried in the
 * binary as src/book_data.inc). A chat message that shares enough words with
 * a task's phrases gets that task's example commands before it, so the model
 * starts from a real command instead of inventing one. Measured in
 * docs/DECISIONS.md (iteration 12): on 120 requests written without seeing the
 * book, 58 right answers against 42.
 *
 * The search is scripts/cmdcheck/livro.ts, the same to the bit (the weights
 * come computed from there, and tests/goldens/livro-busca.tsv holds both to
 * the same answers):
 *  - the text is folded (lowercase, no accents), split into words of two or
 *    more letters and digits, stopwords dropped, and each word stemmed
 *    (plural, verb ending and final vowel off, then six letters);
 *  - a task scores the weights of the stems it shares with the message, plus
 *    a bonus for the share of its best-covered phrase; one shared stem counts
 *    only when it is a whole phrase;
 *  - the best task needs BOOK_MIN_POINTS; a second joins when it scores within
 *    BOOK_SECOND of the first. Tasks without commands for the shell are skipped.
 * Long messages are searched by their last BOOK_TEXT_MAX bytes: a question
 * usually ends with what it asks about. */
#ifndef SMOLMOE_BOOK_H
#define SMOLMOE_BOOK_H

#include <stddef.h>

#define BOOK_FOUND_MAX 2
#define BOOK_TEXT_MAX 1024
#define BOOK_PREFIX_MAX 4096 /* room for any prefix: the longest two tasks take well under this (tests/test_book.c) */

typedef enum { BOOK_POWERSHELL, BOOK_POSIX } BookShell;

/* The tasks for a message, best first; returns how many (0 to BOOK_FOUND_MAX). */
int book_find(const char *message, BookShell shell, int found[BOOK_FOUND_MAX]);

/* A task's name ("espaco-disco"), for notices; "" out of range. */
const char *book_task_id(int task);
int book_task_count(void);

/* The text that goes before the message:
 *   Comandos reais que costumam resolver isso (PowerShell):
 *   ```powershell
 *   <each task's example lines>
 *   ```
 *   <blank line>
 * ("bash" for BOOK_POSIX). Returns its length, NUL-terminated; 0 when there
 * are no tasks or it does not fit. */
size_t book_prefix(const int *found, int n, BookShell shell, char *out, size_t cap);

/* The word stem the search uses, of a folded word (a-z, 0-9), into out
 * (at least 7 bytes). Exposed for the tests. */
void book_stem(const char *word, size_t n, char *out);

#endif
