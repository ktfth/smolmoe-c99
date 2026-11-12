#include "tokenizer_qwen.h"
#include "config.h"
#include "platform.h"

#include <stdio.h>
#include <string.h>

#include "tokenizer_pretok.h"

#define MAX_PIECE 65536u
#define MAX_PAYLOAD 67108864u
#define MAX_QBP_MERGES 1048576u
#define MAX_TRUSTED 6u
#define PHI_ROLES 3u

typedef struct { uint32_t a, b, out; } QwenMerge;
/* One live symbol of the pre-token being merged. Indexes increase along the list. */
typedef struct { uint32_t prev, next, id; } QwenSym;
/* A candidate pair. rank is the merge index (lower merges first). */
typedef struct { uint32_t left, right, rank; } QwenBigram;

/* A control token the operator's system text may spell out. rstrip: the
 * whitespace after it belongs to the marker (HF added-token rstrip). */
typedef struct {
    const char *text;
    int rstrip;
    uint32_t id, len; /* id is UINT32_MAX when the vocabulary lacks it */
} Marker;

static const char *const PHI_ROLE_NAMES[PHI_ROLES] = {"system", "user", "assistant"};
static const char *const PHI_ROLE_TOKENS[PHI_ROLES] = {"<|system|>", "<|user|>", "<|assistant|>"};

struct QwenTokenizer {
    uint32_t vocab, nmerge, hash_size, capacity, max_piece;
    uint32_t normalizer, pretok, chat;
    uint32_t im_start;                /* ChatML */
    uint32_t role[PHI_ROLES];         /* Phi: <|system|>, <|user|>, <|assistant|> */
    uint32_t tool_open, tool_close;   /* ChatML tool calls, or UINT32_MAX */
    uint32_t heap_cap;
    uint32_t whole_size;              /* ignore_merges: open-addressing table of piece id + 1 */
    uint32_t *whole;
    Marker trusted[MAX_TRUSTED];
    unsigned char **piece, *special, *normalized;
    uint8_t *cls;
    uint32_t *length, *hash, *cp, *offset;
    QwenMerge *merge;
    QwenSym *sym;
    QwenBigram *heap;
};

static void read_exact(ByteReader *r, void *buf, size_t n, const char *what) {
    if (!br_read(r, buf, n)) die("truncated SMQBP1 tokenizer: %s", what);
}

static void insert_merge(QwenTokenizer *q, uint32_t i) {
    const QwenMerge *m = &q->merge[i];
    uint32_t slot = (uint32_t)mix64(((uint64_t)m->a << 32) | m->b) & (q->hash_size - 1);
    while (q->hash[slot]) {
        const QwenMerge *old = &q->merge[q->hash[slot] - 1];
        if (old->a == m->a && old->b == m->b) die("duplicate SMQBP1 merge");
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
        die("invalid SMQBP1 merge");
}

static int piece_is(const QwenTokenizer *q, uint32_t id, const char *text) {
    uint32_t n = (uint32_t)strlen(text);
    return q->length[id] == n && !memcmp(q->piece[id], text, n);
}

static uint32_t find_special(const QwenTokenizer *q, const char *text) {
    uint32_t i;
    for (i = 256; i < q->vocab; ++i)
        if (q->special[i] && piece_is(q, i, text)) return i;
    return UINT32_MAX;
}

static uint32_t require_special(const QwenTokenizer *q, const char *text) {
    uint32_t id = find_special(q, text);
    if (id == UINT32_MAX) die("SMQBP1 chat template needs a %s token", text);
    return id;
}

/* v1 implies NFC, the Qwen2 pre-tokenizer and ChatML with im_start in the header. */
static uint32_t read_header(QwenTokenizer *q, ByteReader *f, uint32_t vocab) {
    unsigned char h[32];
    uint32_t version, total;
    if (!br_read(f, h, 8) || memcmp(h, "SMQBP1\0", 7) || (h[7] != 1 && h[7] != 2)) die("invalid SMQBP1 tokenizer header");
    version = h[7];
    if (!br_read(f, h + 8, version == 1 ? 16 : 24) || rd_u32(h + 8) != vocab || vocab < 258)
        die("invalid SMQBP1 tokenizer header");
    q->nmerge = rd_u32(h + 12);
    total = rd_u32(h + 16);
    if (version == 1) {
        q->normalizer = QBP_NORM_NFC;
        q->pretok = QBP_PRETOK_QWEN2;
        q->chat = QBP_CHAT_CHATML;
        q->im_start = rd_u32(h + 20);
    } else {
        q->normalizer = rd_u32(h + 20);
        q->pretok = rd_u32(h + 24);
        q->chat = rd_u32(h + 28);
        q->im_start = UINT32_MAX;
        if (q->normalizer > QBP_NORM_NFC) die("unsupported SMQBP1 normalizer %u", q->normalizer);
        if (q->pretok < QBP_PRETOK_QWEN2 || q->pretok > QBP_PRETOK_LLAMA3)
            die("unsupported SMQBP1 pre-tokenizer %u", q->pretok);
        if (q->chat > QBP_CHAT_CHATML_BOS) die("unsupported SMQBP1 chat template %u", q->chat);
    }
    if (q->nmerge > MAX_QBP_MERGES || total > MAX_PAYLOAD || (version == 1 && q->im_start >= vocab))
        die("invalid SMQBP1 tokenizer dimensions");
    return total;
}

static void add_marker(QwenTokenizer *q, uint32_t idx, const char *text, const char *token_name, int rstrip) {
    q->trusted[idx].text = text;
    q->trusted[idx].rstrip = rstrip;
    q->trusted[idx].len = (uint32_t)strlen(text);
    q->trusted[idx].id = find_special(q, token_name);
}

/* BOS and EOS are control tokens; each template's markers are found by text. */
static void resolve_template(QwenTokenizer *q) {
    uint32_t i;
    if (!q->special[BOS_ID] || !q->special[EOS_ID]) die("SMQBP1 tokens 256 and 257 must be flagged special");
    q->tool_open = q->tool_close = UINT32_MAX;
    for (i = 0; i < MAX_TRUSTED; ++i) {
        q->trusted[i].text = "";
        q->trusted[i].len = 0;
        q->trusted[i].rstrip = 0;
        q->trusted[i].id = UINT32_MAX;
    }
    if (q->chat == QBP_CHAT_CHATML || q->chat == QBP_CHAT_CHATML_BOS) {
        if (!piece_is(q, EOS_ID, "<|im_end|>")) die("SMQBP1 token 257 must be <|im_end|> for ChatML");
        if (q->im_start == UINT32_MAX) q->im_start = require_special(q, "<|im_start|>");
        else if (!q->special[q->im_start] || !piece_is(q, q->im_start, "<|im_start|>"))
            die("SMQBP1 im_start must be the <|im_start|> token");
        add_marker(q, 0, "<tool_call>", "<tool_call>", 0);
        add_marker(q, 1, "</tool_call>", "</tool_call>", 0);
        q->tool_open = q->trusted[0].id;
        q->tool_close = q->trusted[1].id;
    } else if (q->chat == QBP_CHAT_PHI) {
        if (!piece_is(q, BOS_ID, "<|endoftext|>")) die("SMQBP1 token 256 must be <|endoftext|> for the Phi template");
        if (!piece_is(q, EOS_ID, "<|end|>")) die("SMQBP1 token 257 must be <|end|> for the Phi template");
        for (i = 0; i < PHI_ROLES; ++i) q->role[i] = require_special(q, PHI_ROLE_TOKENS[i]);
        add_marker(q, 0, "<|tool|>", "<|tool|>", 1);
        add_marker(q, 1, "<|/tool|>", "<|/tool|>", 1);
        add_marker(q, 2, "<|tool_call|>", "<|tool_call|>", 1);
        add_marker(q, 3, "<|/tool_call|>", "<|/tool_call|>", 1);
        add_marker(q, 4, "<tool_call>", "<|tool_call|>", 1);
        add_marker(q, 5, "</tool_call>", "<|/tool_call|>", 1);
        q->tool_open = q->trusted[2].id;
        q->tool_close = q->trusted[3].id;
    } else if (q->chat == QBP_CHAT_GEMMA) {
        if (!piece_is(q, BOS_ID, "<bos>")) die("SMQBP1 token 256 must be <bos> for the Gemma template");
        if (!piece_is(q, EOS_ID, "<end_of_turn>")) die("SMQBP1 token 257 must be <end_of_turn> for the Gemma template");
        q->im_start = require_special(q, "<start_of_turn>");
        add_marker(q, 0, "<tool_call>", "<tool_call>", 0);
        add_marker(q, 1, "</tool_call>", "</tool_call>", 0);
        q->tool_open = q->trusted[0].id;
        q->tool_close = q->trusted[1].id;
    }
}

static uint32_t bytes_hash(const unsigned char *s, uint32_t n) {
    uint64_t h = 1469598103934665603ull;
    uint32_t i;
    for (i = 0; i < n; ++i) h = (h ^ s[i]) * 1099511628211ull;
    return (uint32_t)mix64(h);
}

/* The ordinary token whose piece is exactly s[0..n), or UINT32_MAX. */
static uint32_t whole_piece(const QwenTokenizer *q, const unsigned char *s, uint32_t n) {
    uint32_t slot = bytes_hash(s, n) & (q->whole_size - 1);
    while (q->whole[slot]) {
        uint32_t id = q->whole[slot] - 1;
        if (q->length[id] == n && !memcmp(q->piece[id], s, n)) return id;
        slot = (slot + 1) & (q->whole_size - 1);
    }
    return UINT32_MAX;
}

/* ignore_merges: every ordinary multi-byte piece (single bytes are the byte tokens). */
static void build_whole(QwenTokenizer *q, Memory *mem) {
    uint32_t i, count = 0;
    for (i = 256; i < q->vocab; ++i) count += !q->special[i] && q->length[i] > 1;
    q->whole_size = 8;
    while (q->whole_size < count * 2) q->whole_size *= 2;
    q->whole = (uint32_t *)mem_alloc(mem, q->whole_size, sizeof(uint32_t));
    for (i = 256; i < q->vocab; ++i) {
        uint32_t slot;
        if (q->special[i] || q->length[i] < 2) continue;
        if (whole_piece(q, q->piece[i], q->length[i]) != UINT32_MAX) die("duplicate SMQBP1 piece");
        slot = bytes_hash(q->piece[i], q->length[i]) & (q->whole_size - 1);
        while (q->whole[slot]) slot = (slot + 1) & (q->whole_size - 1);
        q->whole[slot] = i + 1;
    }
}

QwenTokenizer *qwen_load(ByteReader *f, uint32_t vocab, Memory *mem) {
    unsigned char rec[12], *data;
    uint32_t i, total, used = 0;
    QwenTokenizer *q = (QwenTokenizer *)mem_alloc(mem, 1, sizeof(*q));
    total = read_header(q, f, vocab);
    q->vocab = vocab;
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
        if (!n || n > MAX_PIECE || n > total - used || flag > 1) die("invalid SMQBP1 piece");
        q->piece[i] = data + used;
        q->length[i] = n;
        q->special[i] = (unsigned char)flag;
        used += n;
        if (n > q->max_piece) q->max_piece = n;
        read_exact(f, q->piece[i], n, "piece bytes");
        if (i < 256 && (n != 1 || q->piece[i][0] != i || flag)) die("invalid SMQBP1 byte-token permutation");
    }
    if (used != total) die("SMQBP1 piece byte count mismatch");
    resolve_template(q);
    if (q->pretok == QBP_PRETOK_LLAMA3) build_whole(q, mem);
    for (i = 0; i < q->nmerge; ++i) {
        QwenMerge *m = &q->merge[i];
        read_exact(f, rec, 12, "merges");
        m->a = rd_u32(rec);
        m->b = rd_u32(rec + 4);
        m->out = rd_u32(rec + 8);
        validate_merge(q, m);
        insert_merge(q, i);
    }
    if (!br_at_end(f)) die("extra SMQBP1 tokenizer payload");
    return q;
}

uint32_t qwen_prepare(QwenTokenizer *q, uint32_t max_bytes, Memory *mem) {
    /* NFC can expand code points; the chat template adds a few hundred. */
    q->capacity = max_bytes * 4 + 512;
    q->cp = (uint32_t *)mem_alloc(mem, q->capacity, sizeof(uint32_t));
    q->offset = (uint32_t *)mem_alloc(mem, (size_t)q->capacity + 1, sizeof(uint32_t));
    q->sym = (QwenSym *)mem_alloc(mem, q->capacity, sizeof(*q->sym));
    /* Initial pairs plus at most two new pairs per merge: under 3n entries. */
    q->heap_cap = q->capacity * 3;
    q->heap = (QwenBigram *)mem_alloc(mem, q->heap_cap, sizeof(*q->heap));
    q->normalized = (unsigned char *)mem_alloc(mem, q->capacity, 4);
    q->cls = (uint8_t *)mem_alloc(mem, q->capacity, 1);
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

static void push_id(uint32_t *ids, uint32_t *n, uint32_t cap, uint32_t id) {
    if (*n >= cap) die("prompt workspace exceeded");
    ids[(*n)++] = id;
}

/* Lower rank first; the same rank keeps the leftmost symbol (smaller original index). */
static int bigram_less(const QwenBigram *a, const QwenBigram *b) {
    return a->rank < b->rank || (a->rank == b->rank && a->left < b->left);
}

static void heap_sift_up(QwenBigram *h, uint32_t i) {
    while (i) {
        uint32_t p = (i - 1) / 2;
        QwenBigram t;
        if (!bigram_less(&h[i], &h[p])) break;
        t = h[i];
        h[i] = h[p];
        h[p] = t;
        i = p;
    }
}

static void heap_sift_down(QwenBigram *h, uint32_t n, uint32_t i) {
    for (;;) {
        uint32_t l = i * 2 + 1, m = i;
        QwenBigram t;
        if (l >= n) break;
        if (bigram_less(&h[l], &h[m])) m = l;
        if (l + 1 < n && bigram_less(&h[l + 1], &h[m])) m = l + 1;
        if (m == i) break;
        t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

static void heap_push(QwenBigram *h, uint32_t *n, uint32_t cap, QwenBigram b) {
    if (*n >= cap) die("BPE merge heap exceeded");
    h[*n] = b;
    heap_sift_up(h, *n);
    *n += 1;
}

static QwenBigram heap_pop(QwenBigram *h, uint32_t *n) {
    QwenBigram top = h[0];
    *n -= 1;
    if (*n) {
        h[0] = h[*n];
        heap_sift_down(h, *n, 0);
    }
    return top;
}

/* Queue the pair that starts at left, when those two symbols have a merge. */
static void consider_pair(QwenTokenizer *q, uint32_t *heap_n, uint32_t left) {
    uint32_t right, rank;
    QwenBigram b;
    if (left == UINT32_MAX) return;
    right = q->sym[left].next;
    if (right == UINT32_MAX) return;
    rank = merge_rank(q, q->sym[left].id, q->sym[right].id);
    if (rank == UINT32_MAX) return;
    b.left = left;
    b.right = right;
    b.rank = rank;
    heap_push(q->heap, heap_n, q->heap_cap, b);
}

/* Byte-level BPE of one pre-token: lowest-rank merge first, leftmost on ties.
 * A binary heap holds candidate pairs and a linked list holds the symbols, so a
 * merge only reconsiders the two pairs that touch it. Stale heap entries (the
 * symbols were already merged, or a token changed) fail the link or rank check. */
static void merge_piece(QwenTokenizer *q, uint32_t start, uint32_t count, uint32_t *ids, uint32_t *n) {
    uint32_t k, heap_n = 0;
    if (count > q->capacity) die("byte workspace exceeded");
    if (!count) return;
    for (k = 0; k < count; ++k) {
        q->sym[k].id = q->normalized[start + k];
        q->sym[k].prev = k ? k - 1 : UINT32_MAX;
        q->sym[k].next = k + 1 < count ? k + 1 : UINT32_MAX;
    }
    for (k = 0; k + 1 < count; ++k) consider_pair(q, &heap_n, k);
    while (heap_n) {
        uint32_t right_next, prev;
        QwenBigram b = heap_pop(q->heap, &heap_n);
        if (q->sym[b.left].next != b.right || q->sym[b.right].prev != b.left) continue;
        if (merge_rank(q, q->sym[b.left].id, q->sym[b.right].id) != b.rank) continue;
        q->sym[b.left].id = q->merge[b.rank].out;
        right_next = q->sym[b.right].next;
        q->sym[b.left].next = right_next;
        if (right_next != UINT32_MAX) q->sym[right_next].prev = b.left;
        q->sym[b.right].prev = UINT32_MAX;
        q->sym[b.right].next = UINT32_MAX;
        prev = q->sym[b.left].prev;
        if (prev != UINT32_MAX) consider_pair(q, &heap_n, prev);
        consider_pair(q, &heap_n, b.left);
    }
    for (k = 0; k != UINT32_MAX; k = q->sym[k].next) push_id(ids, n, q->capacity, q->sym[k].id);
}

/* Normalize (NFC or not at all), classify, pre-tokenize, then merge each pre-token. */
static void encode_text(QwenTokenizer *q, const unsigned char *s, uint32_t bytes, uint32_t *ids, uint32_t *n) {
    uint32_t nc = q->normalizer == QBP_NORM_NFC ? normalize_nfc(s, bytes, q->cp, q->capacity)
                                                : decode_utf8(s, bytes, q->cp, q->capacity);
    uint32_t i, j, total = 0;
    for (i = 0; i < nc; ++i) {
        q->offset[i] = total;
        if (total + 4 > q->capacity * 4) die("byte workspace exceeded");
        total += write_utf8(q->cp[i], q->normalized + total);
        q->cls[i] = unicode_class_bits(q->cp[i]);
    }
    q->offset[nc] = total;
    for (i = 0; i < nc; i = j) {
        uint32_t start, count, id;
        j = q->pretok == QBP_PRETOK_O200K ? o200k_boundary(q->cp, q->cls, nc, i)
          : q->pretok == QBP_PRETOK_GEMMA ? gemma_boundary(q->cp, q->cls, nc, i)
          : q->pretok == QBP_PRETOK_LLAMA3 ? llama3_boundary(q->cp, q->cls, nc, i)
                                           : qwen2_boundary(q->cp, q->cls, nc, i);
        start = q->offset[i];
        count = q->offset[j] - start;
        if (q->whole && count > 1 && (id = whole_piece(q, q->normalized + start, count)) != UINT32_MAX)
            push_id(ids, n, q->capacity, id);
        else
            merge_piece(q, start, count, ids, n);
    }
}

static void encode_literal(QwenTokenizer *q, const char *s, uint32_t *ids, uint32_t *n) {
    encode_text(q, (const unsigned char *)s, (uint32_t)strlen(s), ids, n);
}

/* Bytes of leading \s (rstrip after a marker). */
static uint32_t leading_space(const unsigned char *s, uint32_t bytes) {
    uint32_t pos = 0;
    while (pos < bytes) {
        uint32_t next = pos;
        if (!unicode_space(read_utf8(s, bytes, &next))) break;
        pos = next;
    }
    return pos;
}

/* First trusted marker at or after s, or end (and *marker = NULL). */
static const unsigned char *next_marker(const QwenTokenizer *q, const unsigned char *s, const unsigned char *end,
                                        const Marker **marker) {
    for (; s < end; ++s) {
        uint32_t k;
        for (k = 0; k < MAX_TRUSTED; ++k) {
            const Marker *m = &q->trusted[k];
            if (m->id != UINT32_MAX && (size_t)(end - s) >= m->len && !memcmp(s, m->text, m->len)) {
                *marker = m;
                return s;
            }
        }
    }
    *marker = NULL;
    return end;
}

/* System text is the operator's: the template's markers in it become their
 * tokens, as in the templates the models were trained on. User text never does. */
static void encode_trusted(QwenTokenizer *q, const unsigned char *s, uint32_t bytes, uint32_t *ids, uint32_t *n) {
    const unsigned char *end = s + bytes;
    while (s < end) {
        const Marker *m;
        const unsigned char *at = next_marker(q, s, end, &m);
        if (at > s) encode_text(q, s, (uint32_t)(at - s), ids, n);
        if (!m) break;
        push_id(ids, n, q->capacity, m->id);
        s = at + m->len;
        if (m->rstrip) s += leading_space(s, (uint32_t)(end - s));
    }
}

static void encode_content(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                           uint32_t *ids, uint32_t *n) {
    if (!strcmp(role, "system")) encode_trusted(q, text, bytes, ids, n);
    else encode_text(q, text, bytes, ids, n);
}

/* ChatML: <|im_start|>{role}\n{text}<|im_end|>\n, then <|im_start|>assistant\n
 * when open_assistant. <|im_end|> is EOS (257). ChatML-BOS opens the system
 * message that starts a sequence with BOS (256). */
static void chatml_message(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                           int open_assistant, uint32_t *ids, uint32_t *n) {
    if (q->chat == QBP_CHAT_CHATML_BOS && !strcmp(role, "system") && *n == 0) push_id(ids, n, q->capacity, BOS_ID);
    push_id(ids, n, q->capacity, q->im_start);
    encode_literal(q, role, ids, n);
    encode_literal(q, "\n", ids, n);
    encode_content(q, role, text, bytes, ids, n);
    push_id(ids, n, q->capacity, EOS_ID);
    encode_literal(q, "\n", ids, n);
    if (open_assistant) {
        push_id(ids, n, q->capacity, q->im_start);
        encode_literal(q, "assistant\n", ids, n);
    }
}

/* Phi: <|role|>{text}<|end|>, then <|assistant|> when open_assistant; no newlines.
 * Role markers have rstrip, so the text's leading whitespace is dropped. <|end|> is EOS (257). */
static void phi_message(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                        int open_assistant, uint32_t *ids, uint32_t *n) {
    uint32_t r, skip;
    for (r = 0; r < PHI_ROLES && strcmp(role, PHI_ROLE_NAMES[r]); ++r) {}
    if (r == PHI_ROLES) die("the Phi chat template has no role %s", role);
    push_id(ids, n, q->capacity, q->role[r]);
    skip = leading_space(text, bytes);
    encode_content(q, role, text + skip, bytes - skip, ids, n);
    push_id(ids, n, q->capacity, EOS_ID);
    if (open_assistant) push_id(ids, n, q->capacity, q->role[2]);
}

/* Gemma: <start_of_turn>{role}\n{text}<end_of_turn>\n, with assistant mapped to
 * "model" and system opened with <bos>. <end_of_turn> is EOS (257). */
static void gemma_message(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                          int open_assistant, uint32_t *ids, uint32_t *n) {
    const char *gemma_role = !strcmp(role, "assistant") ? "model" : !strcmp(role, "system") ? "user" : role;
    if (strcmp(role, "system") && strcmp(role, "user") && strcmp(role, "assistant") && strcmp(role, "model"))
        die("the Gemma chat template has no role %s", role);
    if (!strcmp(role, "system") && *n == 0) push_id(ids, n, q->capacity, BOS_ID);
    push_id(ids, n, q->capacity, q->im_start);
    encode_literal(q, gemma_role, ids, n);
    encode_literal(q, "\n", ids, n);
    encode_content(q, role, text, bytes, ids, n);
    push_id(ids, n, q->capacity, EOS_ID);
    encode_literal(q, "\n", ids, n);
    if (open_assistant) {
        push_id(ids, n, q->capacity, q->im_start);
        encode_literal(q, "model\n", ids, n);
    }
}

static void chat_message(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                         int open_assistant, uint32_t *ids, uint32_t *n) {
    if (q->chat == QBP_CHAT_PHI) phi_message(q, role, text, bytes, open_assistant, ids, n);
    else if (q->chat == QBP_CHAT_GEMMA) gemma_message(q, role, text, bytes, open_assistant, ids, n);
    else if (q->chat == QBP_CHAT_CHATML || q->chat == QBP_CHAT_CHATML_BOS) chatml_message(q, role, text, bytes, open_assistant, ids, n);
    else die("this tokenizer has no chat template");
}

uint32_t qwen_encode(QwenTokenizer *q, const unsigned char *s, uint32_t bytes, uint32_t *ids,
                     int chat, const char *system) {
    uint32_t n = 0;
    if (chat) {
        chat_message(q, "system", (const unsigned char *)system, (uint32_t)strlen(system), 0, ids, &n);
        chat_message(q, "user", s, bytes, 1, ids, &n);
    } else {
        if (q->chat == QBP_CHAT_GEMMA || q->chat == QBP_CHAT_CHATML_BOS) push_id(ids, &n, q->capacity, BOS_ID);
        encode_text(q, s, bytes, ids, &n);
    }
    if (!n) push_id(ids, &n, q->capacity, BOS_ID);
    return n;
}

uint32_t qwen_chat_turn(QwenTokenizer *q, const char *role, const unsigned char *text, uint32_t bytes,
                        int open_assistant, uint32_t *ids) {
    uint32_t n = 0;
    chat_message(q, role, text, bytes, open_assistant, ids, &n);
    return n;
}

uint32_t qwen_chat_close(QwenTokenizer *q, uint32_t *ids) {
    uint32_t n = 0;
    if (q->chat == QBP_CHAT_NONE) die("this tokenizer has no chat template");
    push_id(ids, &n, q->capacity, EOS_ID);
    if (q->chat == QBP_CHAT_CHATML || q->chat == QBP_CHAT_CHATML_BOS || q->chat == QBP_CHAT_GEMMA) encode_literal(q, "\n", ids, &n);
    return n;
}

uint32_t qwen_decode(const QwenTokenizer *q, uint32_t id, unsigned char *buf) {
    if (id >= q->vocab) die("invalid SMQBP1 token ID %u", id);
    if (q->special[id]) return 0;
    memcpy(buf, q->piece[id], q->length[id]);
    return q->length[id];
}

uint32_t qwen_max_piece(const QwenTokenizer *q) { return q->max_piece; }

int qwen_is_special(const QwenTokenizer *q, uint32_t id) { return id < q->vocab && q->special[id]; }

int qwen_supports_chat(const QwenTokenizer *q) { return q->chat != QBP_CHAT_NONE; }

int qwen_tool_markers(const QwenTokenizer *q, uint32_t *open, uint32_t *close) {
    if (q->tool_open == UINT32_MAX || q->tool_close == UINT32_MAX) return 0;
    *open = q->tool_open;
    *close = q->tool_close;
    return 1;
}
