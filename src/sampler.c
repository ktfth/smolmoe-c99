#include "sampler.h"

#include <math.h>
#include <stdlib.h>

void sampler_init(Sampler *s, uint32_t vocab, const unsigned char *banned, float temperature,
                  uint32_t top_k, float top_p, uint32_t seed, Memory *mem) {
    uint32_t cap;
    if (!seed) die("seed must be nonzero");
    s->vocab = vocab;
    s->banned = banned;
    s->temperature = temperature;
    s->top_k = top_k && top_k < vocab ? top_k : 0;
    s->top_p = top_p;
    s->rng = seed;
    cap = s->top_k ? s->top_k : vocab;
    s->ranked = (SamplerRank *)mem_alloc(mem, cap, sizeof(SamplerRank));
    s->probs = (double *)mem_alloc(mem, cap, sizeof(double));
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static uint32_t argmax(const Sampler *s, const float *logits) {
    uint32_t i, best = UINT32_MAX;
    for (i = 0; i < s->vocab; ++i)
        if (!s->banned[i] && (best == UINT32_MAX || logits[i] > logits[best])) best = i;
    if (best == UINT32_MAX) die("every token is banned");
    return best;
}

/* Keeps the k best (logit desc, id asc) in a sorted array. */
static uint32_t collect_top_k(Sampler *s, const float *logits) {
    uint32_t i, n = 0, k = s->top_k;
    for (i = 0; i < s->vocab; ++i) {
        float l = logits[i];
        uint32_t pos;
        if (s->banned[i] || (n == k && !(l > s->ranked[n - 1].logit))) continue;
        pos = n < k ? n++ : k - 1;
        while (pos > 0 && l > s->ranked[pos - 1].logit) {
            s->ranked[pos] = s->ranked[pos - 1];
            --pos;
        }
        s->ranked[pos].logit = l;
        s->ranked[pos].id = i;
    }
    return n;
}

/* The key is inside the record, so the comparator is reentrant. */
static int by_rank(const void *a, const void *b) {
    const SamplerRank *x = (const SamplerRank *)a, *y = (const SamplerRank *)b;
    if (x->logit != y->logit) return x->logit > y->logit ? -1 : 1;
    if (x->id < y->id) return -1;
    return x->id > y->id ? 1 : 0;
}

static uint32_t collect_all(Sampler *s, const float *logits) {
    uint32_t i, n = 0;
    for (i = 0; i < s->vocab; ++i) {
        if (s->banned[i]) continue;
        s->ranked[n].id = i;
        s->ranked[n].logit = logits[i];
        ++n;
    }
    qsort(s->ranked, n, sizeof(*s->ranked), by_rank);
    return n;
}

uint32_t sampler_next(Sampler *s, const float *logits) {
    uint32_t n, i, kept;
    double total = 0, cumulative = 0, target;
    if (s->temperature == 0) return argmax(s, logits);
    n = s->top_k ? collect_top_k(s, logits) : collect_all(s, logits);
    if (!n) die("every token is banned");
    for (i = 0; i < n; ++i) {
        s->probs[i] = exp(((double)s->ranked[i].logit - (double)s->ranked[0].logit) / (double)s->temperature);
        total += s->probs[i];
    }
    kept = n;
    if (s->top_p < 1.0f) {
        for (i = 0; i < n; ++i) {
            cumulative += s->probs[i];
            if (cumulative >= (double)s->top_p * total) { kept = i + 1; break; }
        }
        total = cumulative;
    }
    target = (double)(xorshift32(&s->rng) >> 8) / 16777216.0 * total;
    for (i = 0; i < kept; ++i) {
        target -= s->probs[i];
        if (target < 0) return s->ranked[i].id;
    }
    return s->ranked[kept - 1].id;
}
