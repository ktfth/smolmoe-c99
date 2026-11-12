/* Pre-tokenizers of the byte-level BPE (SMQBP1), written as matchers instead of
 * a regex engine. Each returns the end of the pre-token that starts at code
 * point i, given the code points c[0..n) and their UC_* class bits k[0..n).
 *
 * qwen2: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|
 *        \s*[\r\n]+|\s+(?!\S)|\s+
 * llama3: qwen2 with \p{N}{1,3} in place of \p{N} (Llama 3.x, tiktoken cl100k-style digits).
 * o200k: [^\r\n\p{L}\p{N}]?[UPPER]*[LOWER]+C?|[^\r\n\p{L}\p{N}]?[UPPER]+[LOWER]*C?|\p{N}{1,3}|
 *        ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *        with C = (?i:'s|'t|'re|'ve|'m|'ll|'d), UPPER = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and
 *        LOWER = [\p{Ll}\p{Lm}\p{Lo}\p{M}] (GPT-4o, Phi-4-mini).
 * \s is Oniguruma's (unicode_space), the engine of HF tokenizers; alternatives are
 * tried in order with backtracking, as a regex find-all would. */
#ifndef SMOLMOE_TOKENIZER_PRETOK_H
#define SMOLMOE_TOKENIZER_PRETOK_H

#include <stdint.h>

#include "unicode.h"
#include "util.h"

#define PRETOK_NO_MATCH UINT32_MAX

/* Unicode simple case folding of the contraction letters: (?i) also matches U+017F as s. */
static inline uint32_t pretok_fold(uint32_t c) {
    return c == 0x17f ? 's' : c >= 'A' && c <= 'Z' ? c + 32 : c;
}

/* End of (?i:'s|'t|'re|'ve|'m|'ll|'d) at j, or j when there is none. */
static inline uint32_t pretok_contraction(const uint32_t *c, uint32_t n, uint32_t j) {
    if (j + 1 < n && c[j] == '\'') {
        uint32_t a = pretok_fold(c[j + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return j + 2;
        if (j + 2 < n) {
            uint32_t b = pretok_fold(c[j + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return j + 3;
        }
    }
    return j;
}

/* [^\s\p{L}\p{N}]: punctuation, symbols, marks and everything unassigned. */
static inline int pretok_punct(const uint32_t *c, const uint8_t *k, uint32_t j) {
    return !unicode_space(c[j]) && !(k[j] & (UC_L | UC_N));
}

/* [^\r\n\p{L}\p{N}]: the optional character before a word. */
static inline int pretok_prefix(const uint32_t *c, const uint8_t *k, uint32_t j) {
    return !(k[j] & (UC_L | UC_N)) && c[j] != '\r' && c[j] != '\n';
}

/* ' ?[^\s\p{L}\p{N}]+' then the tail [\r\n]* (or [\r\n/]* with slash); PRETOK_NO_MATCH if none. */
static inline uint32_t pretok_punct_run(const uint32_t *c, const uint8_t *k, uint32_t n, uint32_t i, int slash) {
    uint32_t j = i;
    if (c[j] == ' ' && j + 1 < n && pretok_punct(c, k, j + 1)) ++j;
    if (!pretok_punct(c, k, j)) return PRETOK_NO_MATCH;
    do { ++j; } while (j < n && pretok_punct(c, k, j));
    while (j < n && (c[j] == '\r' || c[j] == '\n' || (slash && c[j] == '/'))) ++j;
    return j;
}

/* \s*[\r\n]+ (up to the last CR/LF of the run), \s+(?!\S) (the run minus one), \s+. */
static inline uint32_t pretok_whitespace(const uint32_t *c, uint32_t n, uint32_t i) {
    uint32_t j = i, q, last = UINT32_MAX;
    while (j < n && unicode_space(c[j])) ++j;
    if (j == i) die("pre-tokenizer made no progress");
    for (q = i; q < j; ++q)
        if (c[q] == '\r' || c[q] == '\n') last = q;
    if (last != UINT32_MAX) return last + 1;
    if (j == n || j - i == 1) return j;
    return j - 1;
}

/* qwen2 (max_digits 1) and llama3 (max_digits 3) differ only in the digit run. */
static inline uint32_t qwen2_like_boundary(const uint32_t *c, const uint8_t *k, uint32_t n, uint32_t i,
                                           uint32_t max_digits) {
    uint32_t j = pretok_contraction(c, n, i);
    if (j > i) return j;
    if (pretok_prefix(c, k, j) && j + 1 < n && (k[j + 1] & UC_L)) ++j;
    if (k[j] & UC_L) {
        do { ++j; } while (j < n && (k[j] & UC_L));
        return j;
    }
    if (k[i] & UC_N) {
        for (j = i + 1; j < n && j < i + max_digits && (k[j] & UC_N); ++j) {}
        return j;
    }
    j = pretok_punct_run(c, k, n, i, 0);
    return j != PRETOK_NO_MATCH ? j : pretok_whitespace(c, n, i);
}

static inline uint32_t qwen2_boundary(const uint32_t *c, const uint8_t *k, uint32_t n, uint32_t i) {
    return qwen2_like_boundary(c, k, n, i, 1);
}

static inline uint32_t llama3_boundary(const uint32_t *c, const uint8_t *k, uint32_t n, uint32_t i) {
    return qwen2_like_boundary(c, k, n, i, 3);
}

static inline uint32_t pretok_run(const uint8_t *k, uint32_t n, uint32_t j, unsigned bit) {
    while (j < n && (k[j] & bit)) ++j;
    return j;
}

/* [UPPER]*[LOWER]+ from s. Classes overlap (Lm, Lo, marks), so when the maximal
 * UPPER run is not followed by LOWER, the match ends after its last LOWER member. */
static inline uint32_t o200k_word_lower(const uint8_t *k, uint32_t n, uint32_t s) {
    uint32_t e = pretok_run(k, n, s, UC_UPPER), q;
    if (e < n && (k[e] & UC_LOWER)) return pretok_run(k, n, e, UC_LOWER);
    for (q = e; q > s; --q)
        if (k[q - 1] & UC_LOWER) return q;
    return PRETOK_NO_MATCH;
}

/* [UPPER]+[LOWER]* from s. */
static inline uint32_t o200k_word_upper(const uint8_t *k, uint32_t n, uint32_t s) {
    uint32_t e = pretok_run(k, n, s, UC_UPPER);
    return e == s ? PRETOK_NO_MATCH : pretok_run(k, n, e, UC_LOWER);
}

/* [^\r\n\p{L}\p{N}]? WORD C?, trying the optional prefix first (greedy). */
static inline uint32_t o200k_word(const uint32_t *c, const uint8_t *k, uint32_t n, uint32_t i, int upper) {
    uint32_t e = PRETOK_NO_MATCH;
    if (i + 1 < n && pretok_prefix(c, k, i))
        e = upper ? o200k_word_upper(k, n, i + 1) : o200k_word_lower(k, n, i + 1);
    if (e == PRETOK_NO_MATCH) e = upper ? o200k_word_upper(k, n, i) : o200k_word_lower(k, n, i);
    return e == PRETOK_NO_MATCH ? e : pretok_contraction(c, n, e);
}

static inline uint32_t o200k_boundary(const uint32_t *c, const uint8_t *k, uint32_t n, uint32_t i) {
    uint32_t j;
    if ((j = o200k_word(c, k, n, i, 0)) != PRETOK_NO_MATCH) return j;
    if ((j = o200k_word(c, k, n, i, 1)) != PRETOK_NO_MATCH) return j;
    if (k[i] & UC_N) {
        for (j = i + 1; j < n && j < i + 3 && (k[j] & UC_N); ++j) {}
        return j;
    }
    j = pretok_punct_run(c, k, n, i, 1);
    return j != PRETOK_NO_MATCH ? j : pretok_whitespace(c, n, i);
}

/* Gemma / SentencePiece Split(' ', MergedWithNext): digits are single tokens,
 * newlines stand alone, and a leading space attaches to the word/punct run that
 * follows it. */
static inline uint32_t gemma_boundary(const uint32_t *c, const uint8_t *k, uint32_t n, uint32_t i) {
    uint32_t j = i;
    if (c[j] == '\r' || c[j] == '\n') {
        while (j < n && (c[j] == '\r' || c[j] == '\n')) ++j;
        return j;
    }
    if (k[j] & UC_N) return j + 1;
    if (c[j] == ' ') {
        ++j;
        if (j < n && (c[j] == ' ' || c[j] == '\r' || c[j] == '\n' || (k[j] & UC_N))) return j;
    }
    while (j < n && c[j] != ' ' && c[j] != '\r' && c[j] != '\n' && !(k[j] & UC_N)) ++j;
    return j > i ? j : i + 1;
}

#endif
