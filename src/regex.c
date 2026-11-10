#include "regex.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- limits (a pattern beyond them is refused) ------------------------------ */

#define MAX_PATTERN 512
#define MAX_NODES 1024
#define MAX_INSTS 2048
#define MAX_CLASSES 64
#define MAX_RANGES 32
#define BAD_BYTE 0x110000u /* + the byte: a byte that is not well-formed UTF-8 */

/* ---- character tests, as JavaScript with the flags "iu" ---------------------- */

/* Simple case folding restricted to what folds into ASCII: A-Z, U+017F (long s) and U+212A (Kelvin sign). */
static unsigned fold(unsigned c) {
    if (c >= 'A' && c <= 'Z') return c - 'A' + 'a';
    if (c == 0x17F) return 's';
    if (c == 0x212A) return 'k';
    return c;
}

/* \w with "iu": ASCII word characters, plus the two code points whose folding is one. */
static int is_word(unsigned c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == 0x17F ||
           c == 0x212A;
}

static int is_digit(unsigned c) { return c >= '0' && c <= '9'; }

/* \s: WhiteSpace and LineTerminator of ECMAScript (Unicode Zs included). */
static int is_space(unsigned c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000 || c == 0xFEFF;
}

static int is_line_terminator(unsigned c) { return c == '\n' || c == '\r' || c == 0x2028 || c == 0x2029; }

/* ---- classes ------------------------------------------------------------------ */

enum { SET_SPACE = 1, SET_NSPACE = 2, SET_WORD = 4, SET_NWORD = 8, SET_DIGIT = 16, SET_NDIGIT = 32 };

typedef struct Class {
    int negated, ranges;
    unsigned sets;
    unsigned lo[MAX_RANGES], hi[MAX_RANGES];
} Class;

static int in_ranges(const Class *k, unsigned c) {
    int i;
    for (i = 0; i < k->ranges; ++i)
        if (c >= k->lo[i] && c <= k->hi[i]) return 1;
    return 0;
}

/* A member matches when the input, case-folded, or its upper case, is in a range (the ranges are ASCII). */
static int class_has(const Class *k, unsigned c) {
    unsigned f = fold(c);
    if (in_ranges(k, c) || in_ranges(k, f) || (f >= 'a' && f <= 'z' && in_ranges(k, f - 'a' + 'A'))) return 1;
    if ((k->sets & SET_SPACE) && is_space(c)) return 1;
    if ((k->sets & SET_NSPACE) && !is_space(c)) return 1;
    if ((k->sets & SET_WORD) && is_word(c)) return 1;
    if ((k->sets & SET_NWORD) && !is_word(c)) return 1;
    if ((k->sets & SET_DIGIT) && is_digit(c)) return 1;
    if ((k->sets & SET_NDIGIT) && !is_digit(c)) return 1;
    return 0;
}

static int class_match(const Class *k, unsigned c) { return class_has(k, c) != k->negated; }

/* ---- parse tree ---------------------------------------------------------------- */

enum { N_EMPTY, N_CHAR, N_CLASS, N_ANY, N_WORDB, N_NWORDB, N_BOL, N_EOL, N_CAT, N_ALT, N_STAR, N_PLUS, N_QUEST };

typedef struct Node {
    int type, a, b; /* children (node indices) */
    unsigned value; /* N_CHAR: the code point; N_CLASS: the class index */
} Node;

typedef struct Parser {
    const char *s;
    size_t pos, len;
    Node *nodes;
    int nodes_used;
    Class *classes;
    int classes_used;
    char *err;
    size_t errlen;
    int failed;
} Parser;

static int fail(Parser *p, const char *what) {
    if (!p->failed) snprintf(p->err, p->errlen, "regex: %s at offset %u", what, (unsigned)p->pos);
    p->failed = 1;
    return -1;
}

static int node(Parser *p, int type, int a, int b, unsigned value) {
    Node *n;
    if (p->failed) return -1;
    if (p->nodes_used >= MAX_NODES) return fail(p, "pattern too large");
    n = &p->nodes[p->nodes_used];
    n->type = type;
    n->a = a;
    n->b = b;
    n->value = value;
    return p->nodes_used++;
}

static int peek(const Parser *p) { return p->pos < p->len ? (unsigned char)p->s[p->pos] : -1; }

static int is_syntax(int c) { return c >= 0 && strchr("^$\\.*+?()[]{}|/", c) != NULL; }

/* An escape inside a class: a set (\s \S \w \W \d \D, returned through *sets) or one code point. */
static int class_escape(Parser *p, unsigned *cp, unsigned *sets) {
    int c = peek(p);
    *sets = 0;
    ++p->pos;
    switch (c) {
    case 's': *sets = SET_SPACE; return 0;
    case 'S': *sets = SET_NSPACE; return 0;
    case 'w': *sets = SET_WORD; return 0;
    case 'W': *sets = SET_NWORD; return 0;
    case 'd': *sets = SET_DIGIT; return 0;
    case 'D': *sets = SET_NDIGIT; return 0;
    case 'n': *cp = '\n'; return 0;
    case 'r': *cp = '\r'; return 0;
    case 't': *cp = '\t'; return 0;
    case 'f': *cp = '\f'; return 0;
    case 'v': *cp = '\v'; return 0;
    case 'b': *cp = 0x08; return 0; /* [\b] is a backspace */
    case '-': *cp = '-'; return 0;
    default:
        if (is_syntax(c)) {
            *cp = (unsigned)c;
            return 0;
        }
        return fail(p, "escape outside the supported subset");
    }
}

static int parse_class(Parser *p) {
    Class *k;
    int idx;
    if (p->classes_used >= MAX_CLASSES) return fail(p, "too many classes");
    idx = p->classes_used++;
    k = &p->classes[idx];
    memset(k, 0, sizeof(*k));
    ++p->pos; /* [ */
    if (peek(p) == '^') {
        k->negated = 1;
        ++p->pos;
    }
    while (peek(p) != ']') {
        unsigned lo = 0, hi, sets = 0, sets_hi = 0;
        int c = peek(p);
        if (c < 0) return fail(p, "unterminated class");
        if (c >= 0x80) return fail(p, "non-ASCII character in the pattern");
        if (c == '\\') {
            ++p->pos;
            if (class_escape(p, &lo, &sets) < 0) return -1;
        } else {
            lo = (unsigned)c;
            ++p->pos;
        }
        if (sets) {
            k->sets |= sets;
            if (peek(p) == '-' && p->pos + 1 < p->len && p->s[p->pos + 1] != ']')
                return fail(p, "range with a class escape");
            continue;
        }
        hi = lo;
        if (peek(p) == '-' && p->pos + 1 < p->len && p->s[p->pos + 1] != ']') {
            int d;
            ++p->pos;
            d = peek(p);
            if (d >= 0x80) return fail(p, "non-ASCII character in the pattern");
            if (d == '\\') {
                ++p->pos;
                if (class_escape(p, &hi, &sets_hi) < 0) return -1;
                if (sets_hi) return fail(p, "range with a class escape");
            } else {
                hi = (unsigned)d;
                ++p->pos;
            }
            if (hi < lo) return fail(p, "range out of order");
        }
        if (k->ranges >= MAX_RANGES) return fail(p, "class too large");
        k->lo[k->ranges] = lo;
        k->hi[k->ranges] = hi;
        ++k->ranges;
    }
    ++p->pos; /* ] */
    if (k->negated) {
        /* JavaScriptCore does not fold U+017F and U+212A into a class's letters; the matcher does. Negated, that
         * difference would miss what JavaScript matches, so a negated class with letters is outside the subset
         * (the list has none; [^\n] is fine). */
        int i;
        for (i = 0; i < k->ranges; ++i)
            if ((k->lo[i] <= 'z' && k->hi[i] >= 'a') || (k->lo[i] <= 'Z' && k->hi[i] >= 'A'))
                return fail(p, "negated class with letters outside the supported subset");
    }
    return node(p, N_CLASS, -1, -1, (unsigned)idx);
}

static int parse_alt(Parser *p);

static int parse_atom(Parser *p) {
    int c = peek(p);
    if (c == '(') {
        int inner;
        ++p->pos;
        if (peek(p) == '?') {
            if (p->pos + 1 < p->len && p->s[p->pos + 1] == ':') p->pos += 2;
            else return fail(p, "group kind outside the supported subset");
        }
        inner = parse_alt(p);
        if (inner < 0) return -1;
        if (peek(p) != ')') return fail(p, "unbalanced parenthesis");
        ++p->pos;
        return inner;
    }
    if (c == '[') return parse_class(p);
    if (c == '.') {
        ++p->pos;
        return node(p, N_ANY, -1, -1, 0);
    }
    if (c == '^') {
        ++p->pos;
        return node(p, N_BOL, -1, -1, 0);
    }
    if (c == '$') {
        ++p->pos;
        return node(p, N_EOL, -1, -1, 0);
    }
    if (c == '\\') {
        int e;
        ++p->pos;
        e = peek(p);
        ++p->pos;
        switch (e) {
        case 'b': return node(p, N_WORDB, -1, -1, 0);
        case 'B': return node(p, N_NWORDB, -1, -1, 0);
        case 's': case 'S': case 'w': case 'W': case 'd': case 'D': {
            unsigned sets;
            Class *k;
            int idx;
            if (p->classes_used >= MAX_CLASSES) return fail(p, "too many classes");
            idx = p->classes_used++;
            k = &p->classes[idx];
            memset(k, 0, sizeof(*k));
            sets = e == 's' ? SET_SPACE : e == 'S' ? SET_NSPACE : e == 'w' ? SET_WORD : e == 'W' ? SET_NWORD
                 : e == 'd' ? SET_DIGIT : SET_NDIGIT;
            k->sets = sets;
            return node(p, N_CLASS, -1, -1, (unsigned)idx);
        }
        case 'n': return node(p, N_CHAR, -1, -1, '\n');
        case 'r': return node(p, N_CHAR, -1, -1, '\r');
        case 't': return node(p, N_CHAR, -1, -1, '\t');
        case 'f': return node(p, N_CHAR, -1, -1, '\f');
        case 'v': return node(p, N_CHAR, -1, -1, '\v');
        default:
            if (is_syntax(e)) return node(p, N_CHAR, -1, -1, (unsigned)e);
            --p->pos;
            return fail(p, "escape outside the supported subset");
        }
    }
    if (c >= 0x80) return fail(p, "non-ASCII character in the pattern");
    if (c < 0 || strchr("|)*+?{}]", c)) return fail(p, "unexpected character");
    ++p->pos;
    return node(p, N_CHAR, -1, -1, (unsigned)c);
}

static int parse_repeat(Parser *p) {
    int atom = parse_atom(p), c;
    if (atom < 0) return -1;
    c = peek(p);
    if (c == '*' || c == '+' || c == '?') {
        int t = p->nodes[atom].type;
        if (t == N_WORDB || t == N_NWORDB || t == N_BOL || t == N_EOL) return fail(p, "nothing to repeat");
        ++p->pos;
        if (peek(p) == '?') return fail(p, "lazy quantifier outside the supported subset");
        if (peek(p) == '*' || peek(p) == '+' || peek(p) == '{') return fail(p, "nothing to repeat");
        atom = node(p, c == '*' ? N_STAR : c == '+' ? N_PLUS : N_QUEST, atom, -1, 0);
    } else if (c == '{') {
        return fail(p, "{m,n} outside the supported subset");
    }
    return atom;
}

static int parse_concat(Parser *p) {
    int left = node(p, N_EMPTY, -1, -1, 0);
    while (!p->failed && peek(p) >= 0 && peek(p) != '|' && peek(p) != ')') {
        int right = parse_repeat(p);
        if (right < 0) return -1;
        left = node(p, N_CAT, left, right, 0);
    }
    return p->failed ? -1 : left;
}

static int parse_alt(Parser *p) {
    int left = parse_concat(p);
    while (left >= 0 && peek(p) == '|') {
        int right;
        ++p->pos;
        right = parse_concat(p);
        if (right < 0) return -1;
        left = node(p, N_ALT, left, right, 0);
    }
    return left;
}

/* ---- program ------------------------------------------------------------------ */

enum { I_CHAR, I_CLASS, I_ANY, I_SPLIT, I_JMP, I_WORDB, I_NWORDB, I_BOL, I_EOL, I_MATCH };

typedef struct Inst {
    int op, x, y;
    unsigned value;
} Inst;

struct Regex {
    Inst *insts;
    int count;
    Class *classes;
};

typedef struct Emitter {
    Inst *insts;
    int count;
    int failed;
} Emitter;

static int emit(Emitter *e, int op, int x, int y, unsigned value) {
    if (e->count >= MAX_INSTS) {
        e->failed = 1;
        return 0;
    }
    e->insts[e->count].op = op;
    e->insts[e->count].x = x;
    e->insts[e->count].y = y;
    e->insts[e->count].value = value;
    return e->count++;
}

static void gen(Emitter *e, const Node *nodes, int i) {
    const Node *n = &nodes[i];
    int a, b;
    if (e->failed) return;
    switch (n->type) {
    case N_EMPTY: break;
    case N_CHAR: emit(e, I_CHAR, 0, 0, fold(n->value)); break;
    case N_CLASS: emit(e, I_CLASS, 0, 0, n->value); break;
    case N_ANY: emit(e, I_ANY, 0, 0, 0); break;
    case N_WORDB: emit(e, I_WORDB, 0, 0, 0); break;
    case N_NWORDB: emit(e, I_NWORDB, 0, 0, 0); break;
    case N_BOL: emit(e, I_BOL, 0, 0, 0); break;
    case N_EOL: emit(e, I_EOL, 0, 0, 0); break;
    case N_CAT: gen(e, nodes, n->a); gen(e, nodes, n->b); break;
    case N_ALT:
        a = emit(e, I_SPLIT, 0, 0, 0);
        e->insts[a].x = e->count;
        gen(e, nodes, n->a);
        b = emit(e, I_JMP, 0, 0, 0);
        e->insts[a].y = e->count;
        gen(e, nodes, n->b);
        e->insts[b].x = e->count;
        break;
    case N_STAR:
        a = emit(e, I_SPLIT, 0, 0, 0);
        e->insts[a].x = e->count;
        gen(e, nodes, n->a);
        emit(e, I_JMP, a, 0, 0);
        e->insts[a].y = e->count;
        break;
    case N_PLUS:
        a = e->count;
        gen(e, nodes, n->a);
        b = emit(e, I_SPLIT, a, 0, 0);
        e->insts[b].y = e->count;
        break;
    case N_QUEST:
        a = emit(e, I_SPLIT, 0, 0, 0);
        e->insts[a].x = e->count;
        gen(e, nodes, n->a);
        e->insts[a].y = e->count;
        break;
    default: e->failed = 1;
    }
}

Regex *regex_compile(const char *pattern, char *err, size_t errlen) {
    Parser p;
    Emitter e;
    Regex *re;
    int root;
    memset(&p, 0, sizeof(p));
    p.s = pattern;
    p.len = strlen(pattern);
    p.err = err;
    p.errlen = errlen;
    if (errlen) err[0] = 0;
    if (p.len > MAX_PATTERN) {
        snprintf(err, errlen, "regex: pattern longer than %d bytes", MAX_PATTERN);
        return NULL;
    }
    p.nodes = (Node *)malloc(sizeof(Node) * MAX_NODES);
    p.classes = (Class *)malloc(sizeof(Class) * MAX_CLASSES);
    if (!p.nodes || !p.classes) die("allocation failed");
    root = parse_alt(&p);
    if (root >= 0 && p.pos != p.len) root = fail(&p, peek(&p) == ')' ? "unbalanced parenthesis" : "unexpected character");
    if (root < 0) {
        free(p.nodes);
        free(p.classes);
        return NULL;
    }
    e.insts = (Inst *)malloc(sizeof(Inst) * MAX_INSTS);
    if (!e.insts) die("allocation failed");
    e.count = 0;
    e.failed = 0;
    gen(&e, p.nodes, root);
    emit(&e, I_MATCH, 0, 0, 0);
    free(p.nodes);
    if (e.failed) {
        snprintf(err, errlen, "regex: pattern too large");
        free(e.insts);
        free(p.classes);
        return NULL;
    }
    re = (Regex *)malloc(sizeof(*re));
    if (!re) die("allocation failed");
    re->insts = e.insts;
    re->count = e.count;
    re->classes = p.classes;
    return re;
}

void regex_free(Regex *re) {
    if (!re) return;
    free(re->insts);
    free(re->classes);
    free(re);
}

/* ---- matching ------------------------------------------------------------------ */

/* One code point from s[*i..n): a well-formed UTF-8 sequence, or BAD_BYTE + the byte. */
static unsigned next_cp(const unsigned char *s, size_t n, size_t *i) {
    unsigned c = s[*i], lo = 0x80, hi = 0xBF, cp;
    size_t len, k;
    if (c < 0x80) {
        ++*i;
        return c;
    }
    if (c >= 0xC2 && c <= 0xDF) len = 2, cp = c & 0x1F;
    else if (c >= 0xE0 && c <= 0xEF) len = 3, cp = c & 0x0F, lo = c == 0xE0 ? 0xA0 : 0x80, hi = c == 0xED ? 0x9F : 0xBF;
    else if (c >= 0xF0 && c <= 0xF4) len = 4, cp = c & 0x07, lo = c == 0xF0 ? 0x90 : 0x80, hi = c == 0xF4 ? 0x8F : 0xBF;
    else {
        ++*i;
        return BAD_BYTE + c;
    }
    if (*i + len > n || s[*i + 1] < lo || s[*i + 1] > hi) {
        ++*i;
        return BAD_BYTE + c;
    }
    for (k = 1; k < len; ++k) {
        if ((s[*i + k] & 0xC0) != 0x80) {
            ++*i;
            return BAD_BYTE + c;
        }
        cp = (cp << 6) | (s[*i + k] & 0x3Fu);
    }
    *i += len;
    return cp;
}

typedef struct Threads {
    int *pc;
    int count;
} Threads;

/* Adds pc and everything it reaches without reading (jumps, splits, true assertions) to the list. prev/cur are the
 * code points around the position (-1: none). Returns 1 when a match is reached. */
static int add(const Regex *re, Threads *t, int *mark, int gen_id, int *stack, int pc0, long prev, long cur, int at_start) {
    int sp = 0;
    stack[sp++] = pc0;
    while (sp) {
        int pc = stack[--sp];
        const Inst *in;
        if (mark[pc] == gen_id) continue;
        mark[pc] = gen_id;
        in = &re->insts[pc];
        switch (in->op) {
        case I_MATCH: return 1;
        case I_JMP: stack[sp++] = in->x; break;
        case I_SPLIT: stack[sp++] = in->y; stack[sp++] = in->x; break;
        case I_WORDB:
        case I_NWORDB: {
            int boundary = (prev >= 0 && is_word((unsigned)prev)) != (cur >= 0 && is_word((unsigned)cur));
            if (boundary == (in->op == I_WORDB)) stack[sp++] = pc + 1;
            break;
        }
        case I_BOL: if (at_start) stack[sp++] = pc + 1; break;
        case I_EOL: if (cur < 0) stack[sp++] = pc + 1; break;
        default: t->pc[t->count++] = pc; break;
        }
    }
    return 0;
}

int regex_search(const Regex *re, const char *text, size_t n) {
    const unsigned char *s = (const unsigned char *)text;
    /* Two thread lists, the marks and the stack (each instruction is expanded once per position, pushing at most two
     * more): count + count + count + 2 * count + 1. */
    int *buf = (int *)malloc(sizeof(int) * ((size_t)re->count * 5 + 1)), *mark, *stack, gen_id = 1, found = 0;
    Threads cur, nxt, tmp;
    size_t i = 0;
    long prev = -1;
    if (!buf) die("allocation failed");
    cur.pc = buf;
    nxt.pc = buf + re->count;
    mark = buf + 2 * re->count;
    stack = buf + 3 * re->count;
    memset(mark, 0, sizeof(int) * (size_t)re->count);
    cur.count = 0;
    for (;;) {
        size_t after = i;
        long c = i < n ? (long)next_cp(s, n, &after) : -1;
        int k;
        /* A new attempt starts at every position (the search is unanchored), in the same generation as the threads
         * that arrived here, so no instruction enters the list twice. */
        if (add(re, &cur, mark, gen_id, stack, 0, prev, c, i == 0)) {
            found = 1;
            break;
        }
        if (c < 0) break;
        ++gen_id;
        nxt.count = 0;
        for (k = 0; k < cur.count && !found; ++k) {
            const Inst *in = &re->insts[cur.pc[k]];
            int ok = in->op == I_CHAR    ? fold((unsigned)c) == in->value
                   : in->op == I_CLASS   ? class_match(&re->classes[in->value], (unsigned)c)
                   : in->op == I_ANY     ? !is_line_terminator((unsigned)c)
                                         : 0;
            if (ok) {
                size_t after2 = after;
                long c2 = after < n ? (long)next_cp(s, n, &after2) : -1;
                found = add(re, &nxt, mark, gen_id, stack, cur.pc[k] + 1, c, c2, 0);
            }
        }
        if (found) break;
        tmp = cur;
        cur = nxt;
        nxt = tmp;
        prev = c;
        i = after;
    }
    free(buf);
    return found;
}
