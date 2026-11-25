/* Streaming Markdown-to-ANSI pretty-printer for interactive terminals.
 * When disabled (pipes, files, NO_COLOR), passes bytes straight through. */
#ifndef SMOLMOE_MDTERM_H
#define SMOLMOE_MDTERM_H

#include "output.h"

#include <stddef.h>
#include <stdint.h>

typedef struct MdTerm {
    Output *out;
    int enabled;
    int in_code;
    char line[4096];
    size_t len;
} MdTerm;

void mdterm_init(MdTerm *m, Output *out, int enabled);
void mdterm_bytes(MdTerm *m, const unsigned char *p, uint32_t n);
void mdterm_finish(MdTerm *m);

#endif
