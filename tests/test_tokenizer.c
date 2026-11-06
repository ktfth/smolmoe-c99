/* Unit tests for the byte-level BPE tokenizer (SMQBP1): Unicode class bits,
 * the Qwen2 and o200k pre-tokenizers, the v1/v2 loader and the ChatML and Phi
 * chat templates. Rejections (die) are covered by tests/tokenizer.test.ts. */
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "tokenizer.h"
#include "tokenizer_pretok.h"
#include "tokenizer_qwen.h"

static int failures;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                      \
            fputc('\n', stderr);                               \
            ++failures;                                        \
        }                                                      \
    } while (0)

/* ---- Unicode classes ---------------------------------------------------- */

static void expect_bits(uint32_t cp, unsigned bits) {
    unsigned got = unicode_class_bits(cp);
    CHECK(got == bits, "U+%04X: bits %u, expected %u", (unsigned)cp, got, bits);
}

static void test_classes(void) {
    expect_bits('a', UC_L | UC_LOWER);
    expect_bits('A', UC_L | UC_UPPER);
    expect_bits(0x01c5, UC_L | UC_UPPER);              /* Lt */
    expect_bits(0x02b0, UC_L | UC_UPPER | UC_LOWER);   /* Lm */
    expect_bits(0x4e2d, UC_L | UC_UPPER | UC_LOWER);   /* Lo */
    expect_bits(0x0301, UC_UPPER | UC_LOWER);          /* Mn: not a letter */
    expect_bits(0x0903, UC_UPPER | UC_LOWER);          /* Mc */
    expect_bits(0x20dd, UC_UPPER | UC_LOWER);          /* Me */
    expect_bits('7', UC_N);
    expect_bits(0x00b2, UC_N);                         /* No */
    expect_bits(0x216b, UC_N);                         /* Nl */
    expect_bits(' ', 0);
    expect_bits('\'', 0);
    expect_bits(0x0378, 0);                            /* unassigned */
    expect_bits(0x10ffff, 0);
    expect_bits(0x2ebf0, UC_L | UC_UPPER | UC_LOWER);  /* CJK Ext. I, Unicode 15.1 */
    expect_bits(0x1c89, UC_L | UC_UPPER);              /* Cyrillic Tje, Unicode 16.0 */
    expect_bits(0x1c8a, UC_L | UC_LOWER);              /* Unicode 16.0 */
    CHECK(unicode_space(0x85) && unicode_space(0x3000) && unicode_space(0xa0), "Oniguruma \\s members");
    CHECK(!unicode_space(0xfeff) && !unicode_space(0x200b), "U+FEFF and U+200B are not \\s");
}

/* ---- pre-tokenizers ----------------------------------------------------- */

/* Splits UTF-8 text and joins the pieces with '|' (absent from the inputs). */
static const char *split(int o200k, const char *text) {
    static char out[2048];
    uint32_t cp[256], n = 0, pos = 0, i = 0, w = 0, len = (uint32_t)strlen(text);
    uint8_t k[256];
    while (pos < len) cp[n++] = read_utf8((const unsigned char *)text, len, &pos);
    for (i = 0; i < n; ++i) k[i] = unicode_class_bits(cp[i]);
    for (i = 0; i < n;) {
        uint32_t j = o200k ? o200k_boundary(cp, k, n, i) : qwen2_boundary(cp, k, n, i);
        if (j <= i || j > n) return "(no progress)";
        if (i) out[w++] = '|';
        for (; i < j; ++i) w += write_utf8(cp[i], (unsigned char *)out + w);
    }
    out[w] = 0;
    return out;
}

static void expect_split(int o200k, const char *text, const char *pieces) {
    const char *got = split(o200k, text);
    CHECK(!strcmp(got, pieces), "%s split of \"%s\":\n  got      %s\n  expected %s", o200k ? "o200k" : "qwen2",
          text, got, pieces);
}

static void test_qwen2_split(void) {
    expect_split(0, "Hello world", "Hello| world");
    expect_split(0, "HelloWorld iPhone", "HelloWorld| iPhone");
    expect_split(0, "don't DON'T", "don|'t| DON|'T");
    expect_split(0, "1234567", "1|2|3|4|5|6|7");
    expect_split(0, "a/b//\n/c", "a|/b|//\n|/c");
    expect_split(0, "नमस्ते", "नमस|्त|े");      /* marks are not letters in Qwen2 */
    expect_split(0, "x'ſa", "x|'ſ|a");                /* (?i) folds U+017F to s */
    expect_split(0, "  hi\n\n  there", " | hi|\n\n| | there");
}

static void test_o200k_split(void) {
    expect_split(1, "HelloWorld iPhone HELLOworld", "Hello|World| i|Phone| HELLOworld");
    expect_split(1, "DON'T don't I'm WE'LL she's", "DON'T| don't| I'm| WE'LL| she's");
    expect_split(1, "'s alone", "'s| alone");
    expect_split(1, "x'ſ", "x'ſ");
    expect_split(1, "1234567 + 89 = ?", "123|456|7| +| |89| =| ?"); /* digits take no space prefix */
    expect_split(1, " 123", " |123");
    expect_split(1, "a/b//\n/c\n", "a|/b|//\n/|c|\n");
    expect_split(1, "नमस्ते दुनिया", "नमस्ते| दुनिया");
    expect_split(1, "中A ABC", "中|A| ABC");
    expect_split(1, "\tA\r\n B C D", "\tA|\r\n| B| C| D");
    expect_split(1, "Ⅻ²٣4", "Ⅻ²٣|4");
    expect_split(1, "  Uma frase com espaços.\n\nNova linha.", " | Uma| frase| com| espaços|.\n\n|Nova| linha|.");
    /* Oniguruma's \s has U+0085 and lacks U+FEFF (JavaScript's is the opposite). */
    expect_split(1, "a\xc2\x85\xc2\x85" "b", "a|\xc2\x85|\xc2\x85" "b");
    expect_split(1, "a\xef\xbb\xbf\xef\xbb\xbf" "b", "a|\xef\xbb\xbf\xef\xbb\xbf|b");
    expect_split(1, "e\xcc\x81" "clair", "e\xcc\x81" "clair");
}

/* ---- in-memory .qbp files ----------------------------------------------- */

typedef struct Blob {
    unsigned char data[8192];
    uint32_t n;
} Blob;

typedef struct Piece {
    const char *text;
    uint32_t flag;
} Piece;

static void put_u32(Blob *b, uint32_t v) {
    b->data[b->n++] = (unsigned char)v;
    b->data[b->n++] = (unsigned char)(v >> 8);
    b->data[b->n++] = (unsigned char)(v >> 16);
    b->data[b->n++] = (unsigned char)(v >> 24);
}

/* Bytes 0..255, then `extra` from ID 256, then one merge "a"+"b" -> the last extra
 * piece, which must be "ab". h[0..2] = normalizer, pre-tokenizer, template (v2)
 * or h[0] = im_start (v1). */
static uint32_t build(Blob *b, int version, const uint32_t *h, const Piece *extra, uint32_t nextra) {
    uint32_t i, total = 256, vocab = 256 + nextra;
    for (i = 0; i < nextra; ++i) total += (uint32_t)strlen(extra[i].text);
    b->n = 0;
    memcpy(b->data, version == 1 ? "SMQBP1\0\1" : "SMQBP1\0\2", 8);
    b->n = 8;
    put_u32(b, vocab);
    put_u32(b, 1);
    put_u32(b, total);
    for (i = 0; i < (version == 1 ? 1u : 3u); ++i) put_u32(b, h[i]);
    for (i = 0; i < 256; ++i) {
        put_u32(b, 1);
        put_u32(b, 0);
        b->data[b->n++] = (unsigned char)i;
    }
    for (i = 0; i < nextra; ++i) {
        uint32_t len = (uint32_t)strlen(extra[i].text);
        put_u32(b, len);
        put_u32(b, extra[i].flag);
        memcpy(b->data + b->n, extra[i].text, len);
        b->n += len;
    }
    put_u32(b, 'a');
    put_u32(b, 'b');
    put_u32(b, vocab - 1);
    return vocab;
}

/* Appends the bytes of s as IDs. */
static uint32_t text_ids(uint32_t *ids, uint32_t n, const char *s) {
    for (; *s; ++s) ids[n++] = (unsigned char)*s;
    return n;
}

static void expect_ids(const char *what, const uint32_t *got, uint32_t n, const uint32_t *want, uint32_t m) {
    uint32_t i;
    int same = n == m;
    for (i = 0; same && i < n; ++i) same = got[i] == want[i];
    if (same) return;
    fprintf(stderr, "FAIL %s\n  got     ", what);
    for (i = 0; i < n; ++i) fprintf(stderr, " %u", got[i]);
    fputs("\n  expected", stderr);
    for (i = 0; i < m; ++i) fprintf(stderr, " %u", want[i]);
    fputc('\n', stderr);
    ++failures;
}

static uint32_t chat(Tokenizer *t, const char *role, const char *text, int open, uint32_t *ids) {
    return tokenizer_chat_turn(t, role, (const unsigned char *)text, (uint32_t)strlen(text), open, ids);
}

/* ---- ChatML (Qwen) ------------------------------------------------------- */

static const Piece CHATML[] = {{"<|endoftext|>", 1}, {"<|im_end|>", 1}, {"<|im_start|>", 1},
                               {"<tool_call>", 1}, {"</tool_call>", 1}, {"ab", 0}};

static void check_chatml(Tokenizer *t, const char *what) {
    uint32_t ids[512], want[512], n, m = 0, open = 0, close = 0;
    CHECK(tokenizer_supports_chat(t), "%s: supports chat", what);
    CHECK(tokenizer_tool_markers(t, &open, &close) && open == 259 && close == 260, "%s: tool markers", what);
    n = tokenizer_encode(t, (const unsigned char *)"ab", 2, ids, 1, "Use <tool_call>.");
    want[m++] = 258;
    m = text_ids(want, m, "system\nUse ");
    want[m++] = 259;
    m = text_ids(want, m, ".");
    want[m++] = 257;
    m = text_ids(want, m, "\n");
    want[m++] = 258;
    m = text_ids(want, m, "user\n");
    want[m++] = 261; /* "ab" */
    want[m++] = 257;
    m = text_ids(want, m, "\n");
    want[m++] = 258;
    m = text_ids(want, m, "assistant\n");
    expect_ids(what, ids, n, want, m);
    n = tokenizer_chat_close(t, ids);
    want[0] = 257;
    want[1] = '\n';
    expect_ids("ChatML close", ids, n, want, 2);
    /* NFC: e + U+0301 becomes U+00E9; the user's markers stay text. */
    n = chat(t, "user", "e\xcc\x81<tool_call>", 0, ids);
    m = 0;
    want[m++] = 258;
    m = text_ids(want, m, "user\n\xc3\xa9<tool_call>");
    want[m++] = 257;
    m = text_ids(want, m, "\n");
    expect_ids("ChatML user turn", ids, n, want, m);
}

static void test_chatml(void) {
    static Blob b;
    Memory mem;
    Tokenizer *t;
    const uint32_t v1[1] = {258}, v2[3] = {1, 1, 1};
    uint32_t vocab;
    memset(&mem, 0, sizeof(mem));
    mem.limit = (size_t)64 << 20;
    vocab = build(&b, 1, v1, CHATML, 6);
    t = tokenizer_parse(b.data, b.n, vocab, &mem);
    tokenizer_prepare(t, 256, &mem);
    check_chatml(t, "ChatML v1");
    vocab = build(&b, 2, v2, CHATML, 6);
    t = tokenizer_parse(b.data, b.n, vocab, &mem);
    tokenizer_prepare(t, 256, &mem);
    check_chatml(t, "ChatML v2");
    mem_release(&mem);
}

/* ---- Phi-4-mini ---------------------------------------------------------- */

static const Piece PHI[] = {{"<|endoftext|>", 1}, {"<|end|>", 1}, {"<|system|>", 1}, {"<|user|>", 1},
                            {"<|assistant|>", 1}, {"<|tool|>", 1}, {"<|/tool|>", 1}, {"ab", 0}};
enum { P_SYSTEM = 258, P_USER, P_ASSISTANT, P_TOOL, P_TOOL_END, P_AB };

static void test_phi(void) {
    static Blob b;
    Memory mem;
    Tokenizer *t;
    const uint32_t h[3] = {0, 2, 2};
    uint32_t ids[512], want[512], n, m, open = 0, close = 0;
    memset(&mem, 0, sizeof(mem));
    mem.limit = (size_t)64 << 20;
    build(&b, 2, h, PHI, 8);
    t = tokenizer_parse(b.data, b.n, 264, &mem);
    tokenizer_prepare(t, 256, &mem);
    CHECK(tokenizer_supports_chat(t), "Phi supports chat");
    CHECK(!tokenizer_tool_markers(t, &open, &close), "Phi has no tool-call protocol yet");
    CHECK(tokenizer_is_control(t, P_USER) && tokenizer_is_control(t, 256) && !tokenizer_is_control(t, 257),
          "Phi markers are control tokens, <|end|> (EOS) is not");

    n = tokenizer_encode(t, (const unsigned char *)"U", 1, ids, 1, "S");
    {
        const uint32_t expected[] = {P_SYSTEM, 'S', 257, P_USER, 'U', 257, P_ASSISTANT};
        expect_ids("Phi single turn", ids, n, expected, 7);
    }
    /* rstrip: whitespace after a role marker is dropped; trailing whitespace is kept. */
    n = chat(t, "user", " \n\t\xc2\x85" "ab \n", 1, ids);
    {
        const uint32_t expected[] = {P_USER, P_AB, ' ', '\n', 257, P_ASSISTANT};
        expect_ids("Phi rstrip", ids, n, expected, 6);
    }
    n = chat(t, "user", "   ", 0, ids);
    {
        const uint32_t expected[] = {P_USER, 257};
        expect_ids("Phi blank message", ids, n, expected, 2);
    }
    /* A user's literal marker stays text; no NFC. */
    n = chat(t, "user", "<|end|>e\xcc\x81", 0, ids);
    m = 0;
    want[m++] = P_USER;
    m = text_ids(want, m, "<|end|>e\xcc\x81");
    want[m++] = 257;
    expect_ids("Phi user markers stay text", ids, n, want, m);
    /* System text is trusted: tool declaration markers become tokens, rstrip after them. */
    n = chat(t, "system", "S<|tool|> [x]<|/tool|>  ", 0, ids);
    {
        const uint32_t expected[] = {P_SYSTEM, 'S', P_TOOL, '[', 'x', ']', P_TOOL_END, 257};
        expect_ids("Phi system with tool declarations", ids, n, expected, 8);
    }
    n = chat(t, "assistant", "ok", 0, ids);
    {
        const uint32_t expected[] = {P_ASSISTANT, 'o', 'k', 257};
        expect_ids("Phi assistant turn", ids, n, expected, 4);
    }
    n = tokenizer_chat_close(t, ids);
    want[0] = 257;
    expect_ids("Phi close", ids, n, want, 1);
    n = tokenizer_encode(t, (const unsigned char *)"", 0, ids, 0, "");
    want[0] = BOS_ID;
    expect_ids("empty plain prompt", ids, n, want, 1);
    mem_release(&mem);
}

static void test_no_template(void) {
    static Blob b;
    static const Piece plain[] = {{"<|endoftext|>", 1}, {"<|end|>", 1}, {"ab", 0}};
    Memory mem;
    Tokenizer *t;
    const uint32_t h[3] = {0, 2, 0};
    uint32_t ids[64], n, open, close;
    memset(&mem, 0, sizeof(mem));
    mem.limit = (size_t)64 << 20;
    build(&b, 2, h, plain, 3);
    t = tokenizer_parse(b.data, b.n, 259, &mem);
    tokenizer_prepare(t, 64, &mem);
    CHECK(!tokenizer_supports_chat(t), "template 0 has no chat");
    CHECK(!tokenizer_tool_markers(t, &open, &close), "template 0 has no tool markers");
    n = tokenizer_encode(t, (const unsigned char *)"HelloWorld ab", 13, ids, 0, "");
    CHECK(n == 12 && ids[11] == 258, "plain o200k encoding (%u IDs)", n);
    mem_release(&mem);
}

int main(void) {
    test_classes();
    test_qwen2_split();
    test_o200k_split();
    test_chatml();
    test_phi();
    test_no_template();
    if (failures) {
        fprintf(stderr, "%d tokenizer test(s) failed\n", failures);
        return 1;
    }
    puts("tokenizer tests passed");
    return 0;
}
