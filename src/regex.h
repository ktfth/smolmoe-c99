/* A small regular expression matcher for the change list of proposal 4a
 * (tests/evals/manutencao/lista-mudanca.tsv), with the semantics of a
 * JavaScript RegExp with the flags "iu" on the subset that list uses:
 * literals, escaped syntax characters, ".", \b \B \s \S \w \W \d \D \n \r \t,
 * classes with ranges, negation and those escapes, groups (capturing or
 * (?:...)), alternation, the greedy quantifiers * + ? and the anchors ^ $.
 * Anything else ({m,n}, lazy quantifiers, lookaround, back references, \p,
 * other escapes) is refused when compiling, never guessed.
 *
 * The input is UTF-8 read as code points; a byte that is not well-formed UTF-8
 * is a code point of its own that matches no letter, digit or blank. As in
 * JavaScript with "iu", \s includes the Unicode spaces, and case-insensitive
 * matching folds U+017F (long s) to "s" and U+212A (Kelvin sign) to "k", so
 * both count as word characters. Matching is a Thompson NFA simulation: time is
 * linear in the input for a given pattern, with no backtracking. */
#ifndef SMOLMOE_REGEX_H
#define SMOLMOE_REGEX_H

#include <stddef.h>

typedef struct Regex Regex;

/* Compiles `pattern` (UTF-8, as written in the list). Returns NULL and an
 * explanation in err for a syntax error or a construction outside the subset. */
Regex *regex_compile(const char *pattern, char *err, size_t errlen);
void regex_free(Regex *re);

/* Whether the pattern matches anywhere in s[0..n). */
int regex_search(const Regex *re, const char *s, size_t n);

#endif
