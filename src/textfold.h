/* Portuguese text matching: chat messages folded to lowercase ASCII without
 * accents, then searched for whole words. Shared by the calendar and the
 * calculator, which decide from the user's own words. */
#ifndef SMOLMOE_TEXTFOLD_H
#define SMOLMOE_TEXTFOLD_H

#include <stddef.h>

#define FOLD_MAX 512 /* longest text searched: a chat message */

/* Lowercase ASCII, Portuguese accents removed, ordinal marks dropped, spaces
 * collapsed. Other non-ASCII bytes become '?'. Returns 0 when too long. */
int text_fold(const char *s, char *out, size_t cap);

/* Folds at most the last cap-1 bytes of a long text, starting on a character
 * boundary: a chat question usually ends with what it asks about. */
void text_fold_tail(const char *text, char *out, size_t cap);

int text_is_letter(char c); /* a..z, on folded text */

/* The next whole-word occurrence of `word` in folded s, at or after `from`. */
const char *text_find_word(const char *s, const char *from, const char *word);

/* Whether folded s contains any of `words` as a whole word. */
int text_has_word(const char *s, const char *const *words, size_t count);

#endif
