#include "tokenizer.h"
#include "config.h"
#include "platform.h"
#include "tokenizer_qwen.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define MAX_PIECE 65536u
#define MAX_PAYLOAD 67108864u

enum { TOK_BPE, TOK_SCORE, TOK_QWEN };

typedef struct { uint32_t a, b, bytes; } Merge;

struct Tokenizer {
    int kind;
    uint32_t vocab, count, max_piece, hash_size;
    uint32_t *hash;
    /* SMBPE1 */
    Merge *merges;
    uint32_t *stack;
    /* SMSPB1 */
    unsigned char **pieces, *scratch;
    uint32_t *lengths;
    float *scores;
    /* SMQBP1 */
    QwenTokenizer *qwen;
};

static uint64_t fnv1a(const unsigned char *p, uint32_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    uint32_t i;
    for (i = 0; i < n; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

/* ---- SMSPB1 score-BPE -------------------------------------------------- */

static uint32_t piece_lookup(const Tokenizer *t, const unsigned char *p, uint32_t n) {
    uint32_t slot = (uint32_t)fnv1a(p, n) & (t->hash_size - 1);
    while (t->hash[slot]) {
        uint32_t id = t->hash[slot] - 1;
        if (t->lengths[id] == n && !memcmp(t->pieces[id], p, n)) return id;
        slot = (slot + 1) & (t->hash_size - 1);
    }
    return UINT32_MAX;
}

static void score_load(Tokenizer *t, FILE *f, Memory *mem) {
    unsigned char header[20], rec[8], *data;
    uint32_t i, total, used = 0, vocab = t->vocab;
    if (fread(header, 1, 20, f) != 20 || memcmp(header, "SMSPB1\0\1", 8)) die("invalid score-BPE header");
    t->kind = TOK_SCORE;
    t->count = vocab - 258;
    t->max_piece = rd_u32(header + 12);
    total = rd_u32(header + 16);
    if (rd_u32(header + 8) != vocab || !t->max_piece || t->max_piece > MAX_PIECE || total > MAX_PAYLOAD)
        die("invalid score-BPE dimensions");
    t->pieces = (unsigned char **)mem_alloc(mem, vocab, sizeof(*t->pieces));
    t->lengths = (uint32_t *)mem_alloc(mem, vocab, sizeof(uint32_t));
    t->scores = (float *)mem_alloc(mem, vocab, sizeof(float));
    data = (unsigned char *)mem_alloc(mem, total, 1);
    t->scratch = (unsigned char *)mem_alloc(mem, (size_t)t->max_piece * 2 + 8, 1);
    t->hash_size = 8;
    while (t->hash_size < vocab * 2) t->hash_size *= 2;
    t->hash = (uint32_t *)mem_alloc(mem, t->hash_size, sizeof(uint32_t));
    for (i = 0; i < vocab; ++i) {
        uint32_t n, slot;
        if (fread(rec, 1, 8, f) != 8) die("truncated score-BPE record");
        t->scores[i] = rd_f32(rec);
        n = rd_u32(rec + 4);
        if (!isfinite(t->scores[i]) || !n || n > t->max_piece || n > total - used) die("invalid score-BPE record");
        t->pieces[i] = data + used;
        t->lengths[i] = n;
        used += n;
        if (fread(t->pieces[i], 1, n, f) != n) die("truncated score-BPE piece");
        if (piece_lookup(t, t->pieces[i], n) != UINT32_MAX) die("duplicate score-BPE piece");
        slot = (uint32_t)fnv1a(t->pieces[i], n) & (t->hash_size - 1);
        while (t->hash[slot]) slot = (slot + 1) & (t->hash_size - 1);
        t->hash[slot] = i + 1;
    }
    if (used != total || fgetc(f) != EOF || ferror(f)) die("invalid score-BPE payload size");
    if (piece_lookup(t, (const unsigned char *)" ", 1) == UINT32_MAX) die("score-BPE missing dummy prefix");
}

/* BOS, dummy-space prefix, UTF-8 code points (byte fallback), then repeatedly
 * merge the adjacent pair whose concatenation has the highest score. */
static uint32_t score_encode(const Tokenizer *t, const unsigned char *s, uint32_t bytes, uint32_t *ids) {
    uint32_t n = 1, i = 0;
    ids[0] = BOS_ID;
    if (bytes) ids[n++] = piece_lookup(t, (const unsigned char *)" ", 1);
    while (i < bytes) {
        uint32_t j = i + 1, id, k;
        while (j < bytes && j - i < 4 && (s[j] & 0xc0) == 0x80) ++j;
        id = piece_lookup(t, s + i, j - i);
        if (id != UINT32_MAX) ids[n++] = id;
        else for (k = i; k < j; ++k) ids[n++] = s[k];
        i = j;
    }
    for (;;) {
        float best = -FLT_MAX;
        uint32_t best_id = UINT32_MAX, pos = 0;
        for (i = 1; i + 1 < n; ++i) {
            uint32_t a = ids[i], b = ids[i + 1], na = t->lengths[a], nb = t->lengths[b], id;
            if (na + nb > t->max_piece) continue;
            memcpy(t->scratch, t->pieces[a], na);
            memcpy(t->scratch + na, t->pieces[b], nb);
            id = piece_lookup(t, t->scratch, na + nb);
            if (id != UINT32_MAX && t->scores[id] > best) { best = t->scores[id]; best_id = id; pos = i; }
        }
        if (best_id == UINT32_MAX) break;
        ids[pos] = best_id;
        memmove(ids + pos + 1, ids + pos + 2, (n - pos - 2) * sizeof(*ids));
        --n;
    }
    return n;
}

/* ---- SMBPE1 ranked merges ---------------------------------------------- */

static uint32_t merge_rank(const Tokenizer *t, uint32_t a, uint32_t b) {
    uint32_t p = (uint32_t)mix64(((uint64_t)a << 32) | b) & (t->hash_size - 1);
    while (t->hash[p]) {
        uint32_t r = t->hash[p] - 1;
        if (t->merges[r].a == a && t->merges[r].b == b) return r;
        p = (p + 1) & (t->hash_size - 1);
    }
    return UINT32_MAX;
}

/* Reads one line; 0 at end of file. Overlong lines are an error, never split. */
static int read_line(char *line, int size, FILE *f) {
    if (!fgets(line, size, f)) return 0;
    if (!strchr(line, '\n') && !feof(f)) die("tokenizer line too long");
    return 1;
}

static void bpe_load(Tokenizer *t, FILE *f, Memory *mem) {
    char line[128];
    uint32_t i;
    if (!read_line(line, (int)sizeof(line), f) || strcmp(trim(line), "SMBPE1")) die("invalid tokenizer magic");
    t->kind = TOK_BPE;
    t->count = t->vocab - 258;
    t->max_piece = MAX_PIECE;
    t->hash_size = 8;
    while (t->hash_size < t->count * 2) t->hash_size *= 2;
    t->merges = (Merge *)mem_alloc(mem, t->count, sizeof(Merge));
    t->hash = (uint32_t *)mem_alloc(mem, t->hash_size, sizeof(uint32_t));
    t->stack = (uint32_t *)mem_alloc(mem, (size_t)t->count + 2, sizeof(uint32_t));
    for (i = 0; i < t->count; ++i) {
        char a[32], b[32], extra;
        uint32_t p;
        Merge *m = &t->merges[i];
        if (!read_line(line, (int)sizeof(line), f) || sscanf(line, "%31s %31s %c", a, b, &extra) != 2)
            die("invalid tokenizer merge %u", i);
        m->a = parse_u32(a, 257 + i, "merge token");
        m->b = parse_u32(b, 257 + i, "merge token");
        if (m->a == BOS_ID || m->a == EOS_ID || m->b == BOS_ID || m->b == EOS_ID) die("special token in BPE merge");
        m->bytes = (m->a < 256 ? 1 : t->merges[m->a - 258].bytes) + (m->b < 256 ? 1 : t->merges[m->b - 258].bytes);
        if (m->bytes > MAX_PIECE) die("BPE token expands beyond %u bytes", MAX_PIECE);
        for (p = (uint32_t)mix64(((uint64_t)m->a << 32) | m->b) & (t->hash_size - 1); t->hash[p];
             p = (p + 1) & (t->hash_size - 1)) {
            const Merge *old = &t->merges[t->hash[p] - 1];
            if (old->a == m->a && old->b == m->b) die("duplicate BPE pair");
        }
        t->hash[p] = i + 1;
    }
    while (read_line(line, (int)sizeof(line), f))
        if (*trim(line)) die("extra tokenizer merges");
    if (ferror(f)) die("tokenizer read error");
}

/* Leftmost lowest-rank merge first, deliberately simple O(n^2). */
static uint32_t bpe_encode(const Tokenizer *t, const unsigned char *s, uint32_t bytes, uint32_t *ids) {
    uint32_t n = bytes + 1, i;
    ids[0] = BOS_ID;
    for (i = 0; i < bytes; ++i) ids[i + 1] = s[i];
    for (;;) {
        uint32_t best = UINT32_MAX, pos = 0;
        for (i = 1; i + 1 < n; ++i) {
            uint32_t r = merge_rank(t, ids[i], ids[i + 1]);
            if (r < best) { best = r; pos = i; }
        }
        if (best == UINT32_MAX) break;
        ids[pos] = 258 + best;
        memmove(ids + pos + 1, ids + pos + 2, (n - pos - 2) * sizeof(*ids));
        --n;
    }
    return n;
}

/* Iterative expansion: no recursion depth for adversarial merge trees. */
static uint32_t bpe_decode(Tokenizer *t, uint32_t id, unsigned char *buf) {
    uint32_t n = 0, out = 0;
    t->stack[n++] = id;
    while (n) {
        uint32_t x = t->stack[--n];
        if (x < 256) {
            buf[out++] = (unsigned char)x;
        } else if (x >= 258 && x - 258 < t->count) {
            const Merge *m = &t->merges[x - 258];
            if (n + 2 > t->count + 2) die("BPE decode stack overflow");
            t->stack[n++] = m->b;
            t->stack[n++] = m->a;
        } else if (x != BOS_ID && x != EOS_ID) {
            die("invalid output token %u", x);
        }
    }
    return out;
}

/* ---- dispatch ---------------------------------------------------------- */

Tokenizer *tokenizer_load(const char *path, uint32_t vocab, Memory *mem) {
    Tokenizer *t = (Tokenizer *)mem_alloc(mem, 1, sizeof(*t));
    unsigned char magic[8];
    FILE *f = open_file(path, "rb");
    if (!f) die("cannot open tokenizer %s: %s", path, strerror(errno));
    t->vocab = vocab;
    if (fread(magic, 1, 8, f) == 8 && !memcmp(magic, "SMQBP1\0\1", 8)) {
        fclose(f);
        t->kind = TOK_QWEN;
        t->qwen = qwen_load(path, vocab, mem);
        t->max_piece = qwen_max_piece(t->qwen);
        return t;
    }
    rewind(f);
    if (fread(magic, 1, 8, f) == 8 && !memcmp(magic, "SMSPB1\0\1", 8)) {
        rewind(f);
        score_load(t, f, mem);
    } else {
        rewind(f);
        bpe_load(t, f, mem);
    }
    fclose(f);
    return t;
}

uint32_t tokenizer_prepare(Tokenizer *t, uint32_t max_bytes, Memory *mem) {
    if (t->kind == TOK_QWEN) return qwen_prepare(t->qwen, max_bytes, mem);
    return max_bytes + 2;
}

uint32_t tokenizer_encode(Tokenizer *t, const unsigned char *s, uint32_t n, uint32_t *ids,
                          int chat, const char *system) {
    if (t->kind == TOK_QWEN) return qwen_encode(t->qwen, s, n, ids, chat, system);
    if (chat) die("--chat requires a Qwen tokenizer");
    return t->kind == TOK_SCORE ? score_encode(t, s, n, ids) : bpe_encode(t, s, n, ids);
}

uint32_t tokenizer_decode(Tokenizer *t, uint32_t id, unsigned char *buf) {
    switch (t->kind) {
    case TOK_QWEN:
        return qwen_decode(t->qwen, id, buf);
    case TOK_SCORE:
        if (id >= t->vocab) die("invalid score-BPE token ID %u", id);
        if (id < 256) { buf[0] = (unsigned char)id; return 1; }
        if (id == BOS_ID || id == EOS_ID) return 0;
        memcpy(buf, t->pieces[id], t->lengths[id]);
        return t->lengths[id];
    default:
        return bpe_decode(t, id, buf);
    }
}

uint32_t tokenizer_max_piece(const Tokenizer *t) { return t->max_piece; }

void tokenizer_render_prompt(Tokenizer *t, const uint32_t *ids, uint32_t n, unsigned char *buf,
                             ByteSink sink, void *ctx) {
    uint32_t i = t->kind == TOK_QWEN ? 0 : 1; /* BOS is implicit for the other formats */
    for (; i < n; ++i) {
        uint32_t len = tokenizer_decode(t, ids[i], buf);
        /* The score-BPE dummy prefix is not part of the user's text. */
        if (i == 1 && t->kind == TOK_SCORE && ids[i] >= 258 && len && buf[0] == ' ') sink(ctx, buf + 1, len - 1);
        else sink(ctx, buf, len);
    }
}

int tokenizer_supports_chat(const Tokenizer *t) { return t->kind == TOK_QWEN; }

int tokenizer_is_control(const Tokenizer *t, uint32_t id) {
    if (id == BOS_ID) return 1;
    if (id == EOS_ID) return 0;
    return t->kind == TOK_QWEN && qwen_is_special(t->qwen, id);
}
