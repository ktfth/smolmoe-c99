#include "sampler.h"

#include <math.h>
#include <stdlib.h>

static void sampler_init_body(Sampler *s, uint32_t vocab, const unsigned char *banned, float temperature,
                             uint32_t top_k, float top_p, uint32_t seed, Memory *mem) {
    uint32_t cap;
    if (!seed) sm_fail(SM_ERR_RANGE, "seed must be nonzero");
    s->vocab = vocab;
    s->banned = banned;
    s->temperature = temperature;
    s->top_k = top_k && top_k < vocab ? top_k : 0;
    s->top_p = top_p;
    s->rng = seed;
    cap = s->top_k ? s->top_k : vocab;
    s->ids = (uint32_t *)mem_alloc(mem, cap, sizeof(uint32_t));
    s->logits = (float *)mem_alloc(mem, cap, sizeof(float));
    s->probs = (double *)mem_alloc(mem, cap, sizeof(double));
}

typedef struct SamplerInitArgs {
    Sampler *s;
    uint32_t vocab, top_k, seed;
    const unsigned char *banned;
    float temperature, top_p;
    Memory *mem;
} SamplerInitArgs;

static void sampler_init_protected(void *ctx) {
    SamplerInitArgs *a = (SamplerInitArgs *)ctx;
    sampler_init_body(a->s, a->vocab, a->banned, a->temperature, a->top_k, a->top_p, a->seed, a->mem);
}

sm_status sampler_init(Sampler *s, uint32_t vocab, const unsigned char *banned, float temperature,
                       uint32_t top_k, float top_p, uint32_t seed, Memory *mem) {
    SamplerInitArgs a;
    a.s = s;
    a.vocab = vocab;
    a.banned = banned;
    a.temperature = temperature;
    a.top_k = top_k;
    a.top_p = top_p;
    a.seed = seed;
    a.mem = mem;
    return sm_protect(sampler_init_protected, &a);
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
    if (best == UINT32_MAX) sm_fail(SM_ERR_STATE, "every token is banned");
    return best;
}

/* Keeps the k best (logit desc, id asc) in a sorted array. */
static uint32_t collect_top_k(Sampler *s, const float *logits) {
    uint32_t i, n = 0, k = s->top_k;
    for (i = 0; i < s->vocab; ++i) {
        float l = logits[i];
        uint32_t pos;
        if (s->banned[i] || (n == k && !(l > s->logits[n - 1]))) continue;
        pos = n < k ? n++ : k - 1;
        while (pos > 0 && l > s->logits[pos - 1]) {
            s->logits[pos] = s->logits[pos - 1];
            s->ids[pos] = s->ids[pos - 1];
            --pos;
        }
        s->logits[pos] = l;
        s->ids[pos] = i;
    }
    return n;
}

static const float *sort_logits;
static int by_logit(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    if (sort_logits[x] != sort_logits[y]) return sort_logits[x] > sort_logits[y] ? -1 : 1;
    return x < y ? -1 : x > y;
}

static uint32_t collect_all(Sampler *s, const float *logits) {
    uint32_t i, n = 0;
    for (i = 0; i < s->vocab; ++i)
        if (!s->banned[i]) s->ids[n++] = i;
    sort_logits = logits;
    qsort(s->ids, n, sizeof(*s->ids), by_logit);
    for (i = 0; i < n; ++i) s->logits[i] = logits[s->ids[i]];
    return n;
}

static uint32_t sampler_next_body(Sampler *s, const float *logits) {
    uint32_t n, i, kept;
    double total = 0, cumulative = 0, target;
    if (s->temperature == 0) return argmax(s, logits);
    n = s->top_k ? collect_top_k(s, logits) : collect_all(s, logits);
    if (!n) sm_fail(SM_ERR_STATE, "every token is banned");
    for (i = 0; i < n; ++i) {
        s->probs[i] = exp(((double)s->logits[i] - (double)s->logits[0]) / (double)s->temperature);
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
        if (target < 0) return s->ids[i];
    }
    return s->ids[kept - 1];
}

typedef struct SamplerNextArgs {
    Sampler *s;
    const float *logits;
    uint32_t token;
} SamplerNextArgs;

static void sampler_next_protected(void *ctx) {
    SamplerNextArgs *a = (SamplerNextArgs *)ctx;
    a->token = sampler_next_body(a->s, a->logits);
}

sm_status sampler_next(Sampler *s, const float *logits, uint32_t *token) {
    SamplerNextArgs a;
    sm_status st;
    a.s = s;
    a.logits = logits;
    a.token = 0;
    st = sm_protect(sampler_next_protected, &a);
    if (st == SM_OK) *token = a.token;
    return st;
}
