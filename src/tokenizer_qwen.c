#include "tokenizer_qwen.h"
#include "config.h"
#include "platform.h"

#include <stdio.h>
#include <string.h>

#include "unicode.h"

#define MAX_PIECE 65536u
#define MAX_PAYLOAD 67108864u

typedef struct { uint32_t a, b, out; } QwenMerge;

struct QwenTokenizer {
    uint32_t vocab, nmerge, im_start, hash_size, capacity, max_piece;
    unsigned char **piece, *special, *normalized;
    uint32_t *length, *hash, *cp, *offset, *symbols;
    QwenMerge *merge;
};

static void read_exact(FILE *f, void *buf, size_t n, const char *what) {
    if (fread(buf, 1, n, f) != n) die("truncated Qwen tokenizer: %s", what);
}

static void insert_merge(QwenTokenizer *q, uint32_t i) {
    const QwenMerge *m = &q->merge[i];
    uint32_t slot = (uint32_t)mix64(((uint64_t)m->a << 32) | m->b) & (q->hash_size - 1);
    while (q->hash[slot]) {
        const QwenMerge *old = &q->merge[q->hash[slot] - 1];
        if (old->a == m->a && old->b == m->b) die("duplicate Qwen merge");
        slot = (slot + 1) & (q->hash_size - 1);
    }
    q->hash[slot] = i + 1;
}

static void validate_merge(const QwenTokenizer *q, const QwenMerge *m) {
    uint32_t vocab = q->vocab;
    if (m->a >= vocab || m->b >= vocab || m->out >= vocab ||
        q->special[m->a] || q->special[m->b] || q->special[m->out] ||
        q->length[m->a] + q->length[m->b] != q->length[m->out] ||
        memcmp(q->piece[m->out], q->piece[m->a], q->length[m->a]) ||
        memcmp(q->piece[m->out] + q->length[m->a], q->piece[m->b], q->length[m->b]))
        die("invalid Qwen merge");
}

QwenTokenizer *qwen_load(const char *path, uint32_t vocab, Memory *mem) {
    FILE *f = open_file(path, "rb");
    unsigned char header[24], rec[12], *data;
    uint32_t i, total, used = 0;
    QwenTokenizer *q = (QwenTokenizer *)mem_alloc(mem, 1, sizeof(*q));
    if (!f || fread(header, 1, 24, f) != 24 || memcmp(header, "SMQBP1\0\1", 8) || rd_u32(header + 8) != vocab)
        die("invalid Qwen tokenizer header");
    q->vocab = vocab;
    q->nmerge = rd_u32(header + 12);
    total = rd_u32(header + 16);
    q->im_start = rd_u32(header + 20);
    if (q->nmerge > MAX_MERGES || total > MAX_PAYLOAD || q->im_start >= vocab)
        die("invalid Qwen tokenizer dimensions");
    q->piece = (unsigned char **)mem_alloc(mem, vocab, sizeof(*q->piece));
    q->length = (uint32_t *)mem_alloc(mem, vocab, sizeof(uint32_t));
    q->special = (unsigned char *)mem_alloc(mem, vocab, 1);
    data = (unsigned char *)mem_alloc(mem, total, 1);
    q->merge = (QwenMerge *)mem_alloc(mem, q->nmerge, sizeof(*q->merge));
    q->hash_size = 8;
    while (q->hash_size < q->nmerge * 2) q->hash_size *= 2;
    q->hash = (uint32_t *)mem_alloc(mem, q->hash_size, sizeof(uint32_t));
    for (i = 0; i < vocab; ++i) {
        uint32_t n, flag;
        read_exact(f, rec, 8, "token record");
        n = rd_u32(rec);
        flag = rd_u32(rec + 4);
        if (!n || n > MAX_PIECE || n > total - used || flag > 1) die("invalid Qwen piece");
        q->piece[i] = data + used;
        q->length[i] = n;
        q->special[i] = (unsigned char)flag;
        used += n;
        if (n > q->max_piece) q->max_piece = n;
        read_exact(f, q->piece[i], n, "piece bytes");
        if (i < 256 && (n != 1 || q->piece[i][0] != i || flag)) die("invalid Qwen byte-token permutation");
    }
    if (used != total) die("Qwen piece byte count mismatch");
    for (i = 0; i < q->nmerge; ++i) {
        QwenMerge *m = &q->merge[i];
        read_exact(f, rec, 12, "merges");
        m->a = rd_u32(rec);
        m->b = rd_u32(rec + 4);
        m->out = rd_u32(rec + 8);
        validate_merge(q, m);
        insert_merge(q, i);
    }
    if (fgetc(f) != EOF || ferror(f)) die("extra Qwen tokenizer payload");
    fclose(f);
    return q;
}

uint32_t qwen_prepare(QwenTokenizer *q, uint32_t max_bytes, Memory *mem) {
    /* NFC can expand code points; the chat template adds a few hundred. */
    q->capacity = max_bytes * 4 + 512;
    q->cp = (uint32_t *)mem_alloc(mem, q->capacity, sizeof(uint32_t));
    q->offset = (uint32_t *)mem_alloc(mem, (size_t)q->capacity + 1, sizeof(uint32_t));
    q->symbols = (uint32_t *)mem_alloc(mem, q->capacity, sizeof(uint32_t));
    q->normalized = (unsigned char *)mem_alloc(mem, q->capacity, 4);
    return q->capacity;
}

static uint32_t merge_rank(const QwenTokenizer *q, uint32_t a, uint32_t b) {
    uint32_t slot = (uint32_t)mix64(((uint64_t)a << 32) | b) & (q->hash_size - 1);
    while (q->hash[slot]) {
        uint32_t i = q->hash[slot] - 1;
        if (q->merge[i].a == a && q->merge[i].b == b) return i;
        slot = (slot + 1) & (q->hash_size - 1);
    }
    return UINT32_MAX;
}

static int unicode_space(uint32_t c) {
    return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f ||
           c == 0x205f || c == 0x3000;
}
static uint32_t ascii_lower(uint32_t c) { return c >= 'A' && c <= 'Z' ? c + 32 : c; }
static int punctuation(uint32_t c) { return !unicode_space(c) && !unicode_kind(c); }

/* End of the pre-token starting at i, following Qwen2's regex alternatives:
 * contractions, letters (with one leading non-letter), single digits,
 * punctuation runs, newline-terminated and trailing whitespace. */
static uint32_t qwen_boundary(const uint32_t *c, uint32_t n, uint32_t i) {
    uint32_t j = i, k;
    if (c[i] == '\'' && i + 1 < n) {
        uint32_t a = ascii_lower(c[i + 1]);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
        if (i + 2 < n) {
            uint32_t b = ascii_lower(c[i + 2]);
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
        }
    }
    if (unicode_kind(c[j]) != 1 && c[j] != '\r' && c[j] != '\n' && !unicode_kind(c[j]) &&
        j + 1 < n && unicode_kind(c[j + 1]) == 1)
        ++j;
    if (unicode_kind(c[j]) == 1) {
        do { ++j; } while (j < n && unicode_kind(c[j]) == 1);
        return j;
    }
    if (unicode_kind(c[i]) == 2) return i + 1;
    j = i;
    if (c[j] == 32 && j + 1 < n && punctuation(c[j + 1])) ++j;
    if (punctuation(c[j])) {
        do { ++j; } while (j < n && punctuation(c[j]));
        while (j < n && (c[j] == '\r' || c[j] == '\n')) ++j;
        return j;
    }
    j = i;
    while (j < n && unicode_space(c[j])) ++j;
    if (j > i) {
        uint32_t last = UINT32_MAX;
        for (k = i; k < j; ++k)
            if (c[k] == '\r' || c[k] == '\n') last = k;
        if (last != UINT32_MAX) return last + 1;
        if (j == n) return j;
        if (j - i > 1) return j - 1;
        return j;
    }
    die("Qwen pre-tokenizer made no progress");
}

static void push_id(uint32_t *ids, uint32_t *n, uint32_t cap, uint32_t id) {
    if (*n >= cap) die("Qwen prompt workspace exceeded");
    ids[(*n)++] = id;
}

static void encode_text(QwenTokenizer *q, const unsigned char *s, uint32_t bytes, uint32_t *ids, uint32_t *n) {
    uint32_t nc = normalize_nfc(s, bytes, q->cp, q->capacity), i, j, total = 0;
    for (i = 0; i < nc; ++i) {
        q->offset[i] = total;
        if (total + 4 > q->capacity * 4) die("Qwen byte workspace exceeded");
        total += write_utf8(q->cp[i], q->normalized + total);
    }
    q->offset[nc] = total;
    for (i = 0; i < nc; i = j) {
        uint32_t start, count, k;
        j = qwen_boundary(q->cp, nc, i);
        start = q->offset[i];
        count = q->offset[j] - start;
        if (count > q->capacity) die("Qwen byte workspace exceeded");
        for (k = 0; k < count; ++k) q->symbols[k] = q->normalized[start + k];
        /* Lowest-rank merge first, leftmost on ties. */
        for (;;) {
            uint32_t best = UINT32_MAX, pos = 0;
            for (k = 0; k + 1 < count; ++k) {
                uint32_t r = merge_rank(q, q->symbols[k], q->symbols[k + 1]);
                if (r < best) { best = r; pos = k; }
            }
            if (best == UINT32_MAX) break;
            q->symbols[pos] = q->merge[best].out;
            memmove(q->symbols + pos + 1, q->symbols + pos + 2, (count - pos - 2) * sizeof(uint32_t));
            --count;
        }
        for (k = 0; k < count; ++k) push_id(ids, n, q->capacity, q->symbols[k]);
    }
}

static void encode_literal(QwenTokenizer *q, const char *s, uint32_t *ids, uint32_t *n) {
    encode_text(q, (const unsigned char *)s, (uint32_t)strlen(s), ids, n);
}

/* ChatML: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}
 * <|im_end|>\n<|im_start|>assistant\n — <|im_end|> is EOS (257). */
uint32_t qwen_encode(QwenTokenizer *q, const unsigned char *s, uint32_t bytes, uint32_t *ids,
                     int chat, const char *system) {
    uint32_t n = 0;
    if (chat) {
        push_id(ids, &n, q->capacity, q->im_start);
        encode_literal(q, "system\n", ids, &n);
        encode_literal(q, system, ids, &n);
        push_id(ids, &n, q->capacity, EOS_ID);
        encode_literal(q, "\n", ids, &n);
        push_id(ids, &n, q->capacity, q->im_start);
        encode_literal(q, "user\n", ids, &n);
    }
    encode_text(q, s, bytes, ids, &n);
    if (chat) {
        push_id(ids, &n, q->capacity, EOS_ID);
        encode_literal(q, "\n", ids, &n);
        push_id(ids, &n, q->capacity, q->im_start);
        encode_literal(q, "assistant\n", ids, &n);
    }
    if (!n) push_id(ids, &n, q->capacity, BOS_ID);
    return n;
}

uint32_t qwen_decode(const QwenTokenizer *q, uint32_t id, unsigned char *buf) {
    if (id >= q->vocab) die("invalid Qwen token ID %u", id);
    if (q->special[id]) return 0;
    memcpy(buf, q->piece[id], q->length[id]);
    return q->length[id];
}

uint32_t qwen_max_piece(const QwenTokenizer *q) { return q->max_piece; }

int qwen_is_special(const QwenTokenizer *q, uint32_t id) { return id < q->vocab && q->special[id]; }
