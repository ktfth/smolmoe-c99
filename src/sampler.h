/* Token selection: greedy, or temperature with optional top-k / top-p.
 * Deterministic for a given seed; ties prefer the lower token ID. */
#ifndef SMOLMOE_SAMPLER_H
#define SMOLMOE_SAMPLER_H

#include <stdint.h>

#include "util.h"

typedef struct Sampler {
    uint32_t vocab, top_k, rng;
    float temperature, top_p;
    const unsigned char *banned;
    uint32_t *ids;
    float *logits;
    double *probs;
} Sampler;

/* banned[id] != 0 excludes a token (BOS, control tokens). */
void sampler_init(Sampler *s, uint32_t vocab, const unsigned char *banned, float temperature,
                  uint32_t top_k, float top_p, uint32_t seed, Memory *mem);
uint32_t sampler_next(Sampler *s, const float *logits);

#endif
