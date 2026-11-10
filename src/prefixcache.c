#include "prefixcache.h"
#include "platform.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

/* File: a 64-byte header, then per layer the keys and the values of the n
 * prefix positions, as the native floats of the KV cache. The file is a cache
 * of this machine, never shipped, so it is not portable.
 *   0  "SMKVPRE1"
 *   8  u32 version, n, layers, kv_dim, vocab, 0
 *  32  u64 hash of the prefix token ids
 *  40  u64 hash of the model: its config and a sample of its weights
 *  48  u64 hash of the keys and values that follow
 *  56  u64 hash of the logits of the last prefix position */
#define MAGIC "SMKVPRE1"
#define HEADER_BYTES 64u
#define MATCHED_BYTES 48u /* checked before anything else is read */
#define FNV_BASIS UINT64_C(14695981039346656037)

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    size_t i;
    for (i = 0; i < n; ++i) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

/* Over the keys and values, 8 bytes a step: each step is a bijection of the
 * running hash, so any damage to the data changes the result. This catches a
 * flipped low bit that rounding hides from the logits check. */
static uint64_t data_hash(uint64_t h, const unsigned char *p, size_t n) {
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        uint64_t w;
        memcpy(&w, p + i, 8);
        h = mix64(h ^ w);
    }
    for (; i < n; ++i) h = mix64(h ^ p[i]);
    return h;
}

/* Another model fails here, before its cache is read: the config, the final
 * norm and the first and last embedding rows (another quantization or other
 * weights change them). The logits check covers the rest of the weights. */
static uint64_t model_hash(const Model *m) {
    const Config *c = &m->c;
    const QMat *e = &m->embed;
    uint32_t ints[17];
    float reals[3];
    uint64_t h;
    ints[0] = c->dim; ints[1] = c->hidden; ints[2] = c->layers; ints[3] = c->heads; ints[4] = c->kv_heads;
    ints[5] = c->experts; ints[6] = c->top_k; ints[7] = c->vocab; ints[8] = c->max_context; ints[9] = c->shared;
    ints[10] = c->bos; ints[11] = c->eos; ints[12] = c->qkv_bias; ints[13] = c->gemma3; ints[14] = c->head_dim;
    ints[15] = c->rotary_dim; ints[16] = c->sliding_window;
    reals[0] = c->eps; reals[1] = c->theta; reals[2] = c->local_theta;
    h = fnv(FNV_BASIS, ints, sizeof(ints));
    h = fnv(h, reals, sizeof(reals));
    h = fnv(h, &c->rope_factor, sizeof(c->rope_factor));
    h = fnv(h, &e->dtype, sizeof(e->dtype));
    h = fnv(h, m->final_norm, (size_t)c->dim * sizeof(float));
    h = fnv(h, e->data, e->row_bytes);
    return fnv(h, e->data + (size_t)(e->rows - 1) * e->row_bytes, e->row_bytes);
}

/* The logits of the last processed position (LOGITS_LAST leaves them in row 0). */
static uint64_t last_logits_hash(const Model *m) { return fnv(FNV_BASIS, m->logits, (size_t)m->c.vocab * sizeof(float)); }

static size_t block_bytes(const Model *m, uint32_t n) { return mul_size(mul_size(n, m->kv_dim), sizeof(float)); }

static size_t layer_stride(const Model *m) { return mul_size(m->context, m->kv_dim); }

/* The fields that must match before anything is read: magic, shape, prompt and model. */
static void header_head(unsigned char out[MATCHED_BYTES], const Model *m, const uint32_t *ids, uint32_t n) {
    uint32_t fields[6];
    uint64_t ids_hash = fnv(FNV_BASIS, ids, (size_t)n * sizeof(*ids)), model = model_hash(m);
    fields[0] = 1; /* version */
    fields[1] = n;
    fields[2] = m->c.layers;
    fields[3] = m->kv_dim;
    fields[4] = m->c.vocab;
    fields[5] = 0;
    memcpy(out, MAGIC, 8);
    memcpy(out + 8, fields, sizeof(fields));
    memcpy(out + 32, &ids_hash, 8);
    memcpy(out + 40, &model, 8);
}

/* Loads a cache written for this model and this prefix whose data is intact,
 * then computes the last position again over it: 1 when its logits hash to the
 * stored value (the same weights wrote it). */
static int load(Model *m, const uint32_t *ids, uint32_t n, const char *path) {
    unsigned char want[MATCHED_BYTES];
    char err[256];
    MappedFile f;
    size_t block = block_bytes(m, n), stride = layer_stride(m), data, l;
    uint64_t stored_data, stored_logits;
    if (!try_map_file(path, &f, err, sizeof(err))) return 0;
    header_head(want, m, ids, n);
    data = mul_size(mul_size(block, 2), m->c.layers);
    if (f.size != add_size(HEADER_BYTES, data) || memcmp(f.data, want, sizeof(want)) != 0) {
        unmap_file(&f);
        return 0;
    }
    memcpy(&stored_data, f.data + 48, 8);
    memcpy(&stored_logits, f.data + 56, 8);
    if (data_hash(FNV_BASIS, f.data + HEADER_BYTES, data) != stored_data) {
        unmap_file(&f);
        return 0;
    }
    for (l = 0; l < m->c.layers; ++l) {
        const unsigned char *src = f.data + HEADER_BYTES + l * 2 * block;
        memcpy(m->keys + l * stride, src, block);
        memcpy(m->values + l * stride, src + block, block);
    }
    unmap_file(&f);
    model_forward(m, ids + n - 1, 1, n - 1, LOGITS_LAST);
    return last_logits_hash(m) == stored_logits;
}

/* Writes the cache next to its final name, then renames it over that name. */
static int save(const Model *m, const uint32_t *ids, uint32_t n, const char *path) {
    unsigned char head[HEADER_BYTES];
    char tmp[4096];
    size_t block = block_bytes(m, n), stride = layer_stride(m), l;
    uint64_t h = FNV_BASIS, logits = last_logits_hash(m);
    int ok;
    FILE *f;
    for (l = 0; l < m->c.layers; ++l) {
        h = data_hash(h, (const unsigned char *)(m->keys + l * stride), block);
        h = data_hash(h, (const unsigned char *)(m->values + l * stride), block);
    }
    header_head(head, m, ids, n);
    memcpy(head + 48, &h, 8);
    memcpy(head + 56, &logits, 8);
    /* One name per process: two chats starting together never write the same file. */
    if (snprintf(tmp, sizeof(tmp), "%s.%lu.partial", path, current_process_id()) >= (int)sizeof(tmp)) return 0;
    if (!(f = open_file(tmp, "wb"))) return 0;
    ok = fwrite(head, 1, HEADER_BYTES, f) == HEADER_BYTES;
    for (l = 0; ok && l < m->c.layers; ++l)
        ok = fwrite(m->keys + l * stride, 1, block, f) == block && fwrite(m->values + l * stride, 1, block, f) == block;
    ok = fclose(f) == 0 && ok;
    if (ok && replace_file(tmp, path)) return 1;
    remove(tmp);
    return 0;
}

PrefixCacheState prefix_cache_prefill(Model *m, const uint32_t *ids, uint32_t n, const char *path) {
    MappedFile probe;
    char err[256];
    int existed;
    if (!path) {
        model_prefill(m, ids, n, 0, 0);
        return PREFIX_CACHE_OFF;
    }
    if (load(m, ids, n, path)) return PREFIX_CACHE_LOADED;
    existed = try_map_file(path, &probe, err, sizeof(err));
    if (existed) unmap_file(&probe);
    model_prefill(m, ids, n, 0, 1); /* the last logits are the check written with the cache */
    if (!save(m, ids, n, path)) return PREFIX_CACHE_UNWRITABLE;
    return existed ? PREFIX_CACHE_REBUILT : PREFIX_CACHE_WRITTEN;
}

const char *prefix_cache_state_name(PrefixCacheState s) {
    switch (s) {
    case PREFIX_CACHE_LOADED: return "carregado";
    case PREFIX_CACHE_WRITTEN: return "gravado";
    case PREFIX_CACHE_REBUILT: return "refeito";
    case PREFIX_CACHE_UNWRITABLE: return "nao_gravado";
    default: return NULL;
    }
}
