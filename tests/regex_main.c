/* test-regex: the C matcher of src/regex.c, driven by tests/regex.test.ts.
 * Reads records from stdin, each "<pattern bytes> <input bytes>\n" followed by
 * those bytes, and answers one line per record: 1 (match), 0 (no match) or
 * "E <error>" (pattern refused). */
#include "regex.h"

#include <stdio.h>
#include <stdlib.h>
#if defined(_WIN32)
#  include <fcntl.h>
#  include <io.h>
#endif

static char *read_bytes(size_t n) {
    char *b = (char *)malloc(n + 1);
    if (!b || fread(b, 1, n, stdin) != n) exit(2);
    b[n] = 0;
    return b;
}

int main(void) {
    unsigned long plen, ilen;
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY); /* the inputs carry "\r\n" and must arrive byte for byte */
#endif
    while (scanf("%lu %lu", &plen, &ilen) == 2) {
        char *pattern, *input, err[160];
        Regex *re;
        if (getchar() != '\n') return 2;
        pattern = read_bytes(plen);
        input = read_bytes(ilen);
        re = regex_compile(pattern, err, sizeof(err));
        if (!re) printf("E %s\n", err);
        else printf("%d\n", regex_search(re, input, ilen));
        regex_free(re);
        free(pattern);
        free(input);
    }
    return 0;
}
