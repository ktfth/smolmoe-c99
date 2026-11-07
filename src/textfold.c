#include "textfold.h"

#include <string.h>

int text_fold(const char *s, char *out, size_t cap) {
    static const char latin[65] = "aaaaaaaceeeeiiii?nooooo?ouuuuy??aaaaaaaceeeeiiii?nooooo?ouuuuy?y";
    const unsigned char *p = (const unsigned char *)s;
    size_t n = 0;
    for (; *p; ++p) {
        char c;
        if (*p == 0xc3 && p[1] >= 0x80 && p[1] <= 0xbf) c = latin[*++p - 0x80];
        else if (*p == 0xc2 && (p[1] == 0xaa || p[1] == 0xba)) { ++p; continue; } /* ª º */
        else if (*p >= 0x80) c = '?';
        else c = (char)(*p >= 'A' && *p <= 'Z' ? *p + 32 : *p);
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
        if (c == ' ' && (n == 0 || out[n - 1] == ' ')) continue;
        if (n + 1 >= cap) return 0;
        out[n++] = c;
    }
    if (n && out[n - 1] == ' ') --n;
    out[n] = 0;
    return 1;
}

void text_fold_tail(const char *text, char *out, size_t cap) {
    size_t len = strlen(text);
    if (len >= cap) {
        text += len - (cap - 1);
        while (((unsigned char)*text & 0xc0) == 0x80) ++text;
    }
    if (!text_fold(text, out, cap)) out[0] = 0; /* cannot happen: folding never lengthens */
}

int text_is_letter(char c) { return c >= 'a' && c <= 'z'; }

const char *text_find_word(const char *s, const char *from, const char *word) {
    size_t n = strlen(word);
    const char *p = from;
    while ((p = strstr(p, word)) != NULL) {
        if ((p == s || !text_is_letter(p[-1])) && !text_is_letter(p[n])) return p;
        ++p;
    }
    return NULL;
}

int text_has_word(const char *s, const char *const *words, size_t count) {
    while (*s) {
        const char *start;
        size_t n, i;
        while (*s && !text_is_letter(*s)) ++s;
        start = s;
        while (text_is_letter(*s)) ++s;
        n = (size_t)(s - start);
        for (i = 0; n && i < count; ++i)
            if (strlen(words[i]) == n && !strncmp(start, words[i], n)) return 1;
    }
    return 0;
}
