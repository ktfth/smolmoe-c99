/* Streaming text output that never splits a UTF-8 character across writes. */
#ifndef SMOLMOE_OUTPUT_H
#define SMOLMOE_OUTPUT_H

#include <stdint.h>
#include <stdio.h>

typedef struct Output {
    FILE *f;
    unsigned char pending[4];
    uint32_t n;
} Output;

Output output_to(FILE *f);
/* Writes bytes, holding back an incomplete trailing UTF-8 sequence. The byte
 * content is unchanged, only delayed. Signature matches ByteSink. */
void output_bytes(void *ctx, const unsigned char *p, uint32_t n);
/* Emits anything held back and flushes. */
void output_finish(Output *o);

#endif
