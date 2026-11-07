#include "calc.h"
#include "textfold.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define MAX_NUMBERS 32 /* numbers collected from an expression or a message */
#define MAX_DEPTH 32   /* nested parentheses */

typedef struct Parser {
    const char *p, *context;
    char err[96];
    double nums[MAX_NUMBERS];
    int count, depth, percent;
} Parser;

static int fail(Parser *ps, const char *fmt, ...) {
    va_list ap;
    if (ps->err[0]) return 0; /* keep the first reason */
    va_start(ap, fmt);
    vsnprintf(ps->err, sizeof(ps->err), fmt, ap);
    va_end(ap);
    return 0;
}

static int is_digit(char c) { return c >= '0' && c <= '9'; }

/* At most `max` bytes of s, never splitting a UTF-8 character (for messages). */
static int quote_len(const char *s, int max) {
    int n = 0;
    while (s[n] && n < max) ++n;
    while (n > 0 && s[n] && ((unsigned char)s[n] & 0xc0) == 0x80) --n;
    return n;
}

static void skip_spaces(Parser *ps) {
    while (*ps->p == ' ' || *ps->p == '\t') ++ps->p;
}

/* Matches an operator spelled in ASCII or UTF-8 at the cursor. */
static int take(Parser *ps, const char *op) {
    size_t n = strlen(op);
    skip_spaces(ps);
    if (strncmp(ps->p, op, n)) return 0;
    ps->p += n;
    return 1;
}

/* The number `token` (len bytes, not terminated) appears in the user's
 * message as a whole number, not inside a longer one. */
static int written_by_user(const char *context, const char *token, size_t len) {
    char t[64];
    const char *hit;
    if (!context || len >= sizeof(t)) return 0;
    memcpy(t, token, len);
    t[len] = 0;
    for (hit = strstr(context, t); hit; hit = strstr(hit + 1, t)) {
        int before = hit == context || (!is_digit(hit[-1]) && hit[-1] != '.' && hit[-1] != ',');
        int after = !is_digit(hit[len]) && !((hit[len] == '.' || hit[len] == ',') && is_digit(hit[len + 1]));
        if (before && after) return 1;
    }
    return 0;
}

/* "1.234.567" up to `end`: a first group of 1 to 3 digits that is not "0",
 * then groups of exactly three ("0.125" and "25.12.2024" are not thousands). */
static int thousand_groups(const char *s, size_t end) {
    size_t i, run = 0, groups = 0;
    for (i = 0; i < end; ++i) {
        if (s[i] != '.') {
            ++run;
            continue;
        }
        if (groups ? run != 3 : (run < 1 || run > 3 || s[0] == '0')) return 0;
        ++groups;
        run = 0;
    }
    return groups && run == 3;
}

/* Digits with ',' and '.' separators, read the Brazilian way (see calc.h). */
static int number_value(const char *s, size_t len, const char *context, double *out) {
    char buf[64];
    size_t i, n = 0, dots = 0, commas = 0, last_dot = 0, comma = 0;
    int thousands;
    if (!len || len >= sizeof(buf)) return 0;
    for (i = 0; i < len; ++i) {
        if (s[i] == '.') ++dots, last_dot = i;
        if (s[i] == ',') ++commas, comma = i;
    }
    if (commas > 1 || (commas && dots && comma < last_dot)) return 0; /* "1,234,567", "1,234.56" */
    if (dots > 1 || (dots && commas)) {
        if (!thousand_groups(s, commas ? comma : len)) return 0;
        thousands = 1;
    } else {
        thousands = dots == 1 && thousand_groups(s, len) && written_by_user(context, s, len);
    }
    for (i = 0; i < len; ++i) {
        if (s[i] == '.' && thousands) continue;
        buf[n++] = s[i] == ',' ? '.' : s[i];
    }
    buf[n] = 0;
    return sscanf(buf, "%lf", out) == 1;
}

/* Every operand is recorded for calc_grounded: past the cap, refuse rather
 * than let an unchecked number through. */
static int parse_number(Parser *ps, double *out) {
    const char *start = ps->p;
    while (is_digit(*ps->p) || ((*ps->p == '.' || *ps->p == ',') && is_digit(ps->p[1]))) ++ps->p;
    if (!number_value(start, (size_t)(ps->p - start), ps->context, out))
        return fail(ps, "expressão inválida perto de '%.*s'", quote_len(start, (int)(ps->p - start)), start);
    if (ps->count == MAX_NUMBERS) return fail(ps, "números demais na conta");
    ps->nums[ps->count++] = *out;
    return 1;
}

static int finite_or_fail(Parser *ps, double v) { return isfinite(v) ? 1 : fail(ps, "resultado grande demais"); }

/* Like take(), for an operator written as a word: "de" must not match "desconto". */
static int take_word(Parser *ps, const char *word) {
    size_t n = strlen(word);
    char next;
    skip_spaces(ps);
    if (strncmp(ps->p, word, n)) return 0;
    next = ps->p[n];
    if ((next >= 'a' && next <= 'z') || (next >= 'A' && next <= 'Z') || (unsigned char)next >= 0x80) return 0;
    ps->p += n;
    return 1;
}

static int parse_sum(Parser *ps, double *out);

static int parse_primary(Parser *ps, double *out) {
    int root;
    skip_spaces(ps);
    if (!*ps->p) return fail(ps, "expressão incompleta");
    if (is_digit(*ps->p)) return parse_number(ps, out);
    root = take(ps, "sqrt(") || take(ps, "raiz(");
    if (!root && !take(ps, "(")) return fail(ps, "expressão inválida perto de '%.*s'", quote_len(ps->p, 12), ps->p);
    if (++ps->depth > MAX_DEPTH) return fail(ps, "parênteses demais");
    if (!parse_sum(ps, out)) return 0;
    --ps->depth;
    if (!take(ps, ")")) return fail(ps, "falta fechar um parêntese");
    if (root && *out < 0) return fail(ps, "raiz de número negativo");
    if (root) *out = sqrt(*out);
    return 1;
}

static int parse_unary(Parser *ps, double *out);

/* primary, percent signs, then a right-associative power: 2 ^ 3 ^ 2 = 2 ^ 9.
 * ps->percent tells the caller the term was a lone percentage ("10%"). */
static int parse_power(Parser *ps, double *out) {
    double exponent;
    int percent = 0;
    if (!parse_primary(ps, out)) return 0;
    while (take(ps, "%") || take_word(ps, "por cento") || take_word(ps, "porcento")) {
        *out /= 100;
        percent = 1;
    }
    if (take(ps, "^") || take(ps, "**")) {
        if (!parse_unary(ps, &exponent)) return 0;
        *out = pow(*out, exponent);
        percent = 0;
        if (!finite_or_fail(ps, *out)) return 0;
    }
    ps->percent = percent;
    return 1;
}

/* Signs bind looser than powers: -2 ^ 2 is -(2^2), as written in math. */
static int parse_unary(Parser *ps, double *out) {
    if (take(ps, "-") || take(ps, "\xe2\x88\x92")) {
        if (!parse_unary(ps, out)) return 0;
        *out = -*out;
        return 1;
    }
    if (take(ps, "+")) return parse_unary(ps, out);
    return parse_power(ps, out);
}

static int parse_product(Parser *ps, double *out) {
    int lone;
    if (!parse_unary(ps, out)) return 0;
    lone = ps->percent;
    for (;;) {
        double rhs;
        int divide;
        /* "15% de 230", "3/4 de 200": "de" is "of", a product. */
        if (take(ps, "*") || take(ps, "x") || take(ps, "X") || take(ps, "\xc3\x97") || take_word(ps, "vezes") ||
            take_word(ps, "de"))
            divide = 0;
        else if (take(ps, "/") || take(ps, "\xc3\xb7") || take(ps, ":") || take_word(ps, "dividido por") ||
                 take_word(ps, "divido por"))
            divide = 1;
        else {
            ps->percent = lone;
            return 1;
        }
        if (!parse_unary(ps, &rhs)) return 0;
        if (divide && rhs == 0) return fail(ps, "divisão por zero");
        *out = divide ? *out / rhs : *out * rhs;
        lone = 0;
        if (!finite_or_fail(ps, *out)) return 0;
    }
}

/* "100 - 10%" is 90, as on a pocket calculator: a lone percentage added to or
 * taken from a value is a share of that value. */
static int parse_sum(Parser *ps, double *out) {
    if (!parse_product(ps, out)) return 0;
    for (;;) {
        double rhs;
        int minus;
        if (take(ps, "+") || take_word(ps, "mais")) minus = 0;
        else if (take(ps, "-") || take(ps, "\xe2\x88\x92") || take_word(ps, "menos")) minus = 1;
        else return 1;
        if (!parse_product(ps, &rhs)) return 0;
        if (ps->percent) rhs *= *out;
        *out = minus ? *out - rhs : *out + rhs;
        if (!finite_or_fail(ps, *out)) return 0;
    }
}

static int evaluate(Parser *ps, const char *expr, const char *context, double *out) {
    memset(ps, 0, sizeof(*ps));
    ps->p = expr;
    ps->context = context;
    if (strlen(expr) > CALC_EXPR_MAX) return fail(ps, "expressão longa demais");
    skip_spaces(ps);
    if (!*ps->p) return fail(ps, "expressão vazia");
    if (!parse_sum(ps, out)) return 0;
    skip_spaces(ps);
    if (*ps->p == ')') return fail(ps, "parêntese sem abertura");
    if (*ps->p) return fail(ps, "expressão inválida perto de '%.*s'", quote_len(ps->p, 12), ps->p);
    if (!isfinite(*out)) return fail(ps, "resultado grande demais");
    return 1;
}

int calc_eval(const char *expr, const char *context, double *out, char *err, size_t cap) {
    Parser ps;
    int ok = evaluate(&ps, expr, context, out);
    if (cap) snprintf(err, cap, "%s", ok ? "" : ps.err);
    return ok;
}

/* ---- Brazilian formatting -------------------------------------------------- */

void calc_format(double v, char *out, size_t cap) {
    char raw[64], *dot;
    size_t n = 0, i, int_len, start;
    int decimals;
    double a = fabs(v);
    if (!cap) return;
    if (!isfinite(v) || a >= 1e15 || (a > 0 && a < 1e-9)) { /* "0" would be false for 1e-14 */
        snprintf(out, cap, "%g", v);
        for (dot = out; *dot; ++dot)
            if (*dot == '.') *dot = ',';
        return;
    }
    decimals = a >= 1 || a == 0 ? 2 : 3 - (int)floor(log10(a));
    snprintf(raw, sizeof(raw), "%.*f", decimals, v);
    dot = strchr(raw, '.');
    if (dot) {
        char *end = raw + strlen(raw) - 1;
        while (end > dot && *end == '0') *end-- = 0;
        if (end == dot) *end = 0;
    }
    if (!strcmp(raw, "-0")) strcpy(raw, "0");
    dot = strchr(raw, '.');
    start = raw[0] == '-';
    int_len = (dot ? (size_t)(dot - raw) : strlen(raw)) - start;
    for (i = 0; raw[i] && n + 2 < cap; ++i) {
        size_t pos = i - start; /* digit index within the integer part */
        if (raw[i] == '.') {
            out[n++] = ',';
            continue;
        }
        if (i >= start && pos < int_len && pos && (int_len - pos) % 3 == 0) out[n++] = '.';
        out[n++] = raw[i];
    }
    out[n] = 0;
}

/* ---- numbers in the user's message ----------------------------------------- */

typedef struct NumberWord {
    const char *word;
    double value;
    int kind; /* 0 adds to the group, 1 multiplies it (mil, milhão), 2 stands alone (metade, dobro) */
} NumberWord;

static const NumberWord NUMBER_WORDS[] = {
    {"zero", 0, 0}, {"um", 1, 0}, {"uma", 1, 0}, {"dois", 2, 0}, {"duas", 2, 0}, {"tres", 3, 0},
    {"quatro", 4, 0}, {"cinco", 5, 0}, {"seis", 6, 0}, {"sete", 7, 0}, {"oito", 8, 0}, {"nove", 9, 0},
    {"dez", 10, 0}, {"onze", 11, 0}, {"doze", 12, 0}, {"treze", 13, 0}, {"catorze", 14, 0}, {"quatorze", 14, 0},
    {"quinze", 15, 0}, {"dezesseis", 16, 0}, {"dezessete", 17, 0}, {"dezoito", 18, 0}, {"dezenove", 19, 0},
    {"vinte", 20, 0}, {"trinta", 30, 0}, {"quarenta", 40, 0}, {"cinquenta", 50, 0}, {"sessenta", 60, 0},
    {"setenta", 70, 0}, {"oitenta", 80, 0}, {"noventa", 90, 0}, {"cem", 100, 0}, {"cento", 100, 0},
    {"duzentos", 200, 0}, {"duzentas", 200, 0}, {"trezentos", 300, 0}, {"trezentas", 300, 0},
    {"quatrocentos", 400, 0}, {"quatrocentas", 400, 0}, {"quinhentos", 500, 0}, {"quinhentas", 500, 0},
    {"seiscentos", 600, 0}, {"seiscentas", 600, 0}, {"setecentos", 700, 0}, {"setecentas", 700, 0},
    {"oitocentos", 800, 0}, {"oitocentas", 800, 0}, {"novecentos", 900, 0}, {"novecentas", 900, 0},
    {"mil", 1000, 1}, {"milhao", 1e6, 1}, {"milhoes", 1e6, 1},
    {"metade", 2, 2}, {"meio", 0.5, 2}, {"meia", 0.5, 2}, {"dobro", 2, 2}, {"triplo", 3, 2}, {"terco", 3, 2},
    {"duzia", 12, 2}, {"dezena", 10, 2}, {"centena", 100, 2},
};

#define MAX_SAID 96 /* numbers read from one message (a value can have two readings) */

typedef struct Numbers {
    double v[MAX_SAID];
    int n;
} Numbers;

static void add_number(Numbers *ns, double v) {
    int i;
    for (i = 0; i < ns->n; ++i)
        if (ns->v[i] == v) return;
    if (ns->n < MAX_SAID) ns->v[ns->n++] = v;
}

static const NumberWord *number_word(const char *w, size_t len) {
    size_t i;
    for (i = 0; i < sizeof(NUMBER_WORDS) / sizeof(NUMBER_WORDS[0]); ++i)
        if (strlen(NUMBER_WORDS[i].word) == len && !strncmp(w, NUMBER_WORDS[i].word, len)) return &NUMBER_WORDS[i];
    return NULL;
}

/* "dois mil e quinhentos" -> 2500; each word's own value is kept as well. */
static void words_in(const char *folded, Numbers *ns) {
    const char *s = folded;
    double total = 0, group = 0;
    int active = 0;
    while (*s) {
        const char *w;
        const NumberWord *nw;
        size_t len;
        while (*s && !text_is_letter(*s)) ++s;
        w = s;
        while (text_is_letter(*s)) ++s;
        len = (size_t)(s - w);
        if (!len) break;
        nw = number_word(w, len);
        if (nw && nw->kind == 2) add_number(ns, nw->value);
        if (nw && nw->kind == 0) {
            group += nw->value;
            active = 1;
            add_number(ns, nw->value);
            continue;
        }
        if (nw && nw->kind == 1) {
            total += (group != 0 ? group : 1) * nw->value;
            group = 0;
            active = 1;
            continue;
        }
        if (active && len == 1 && w[0] == 'e') continue;
        if (active) add_number(ns, total + group);
        total = group = 0;
        active = 0;
    }
    if (active) add_number(ns, total + group);
}

/* "2 mil", "1,5 milhão": the scale word written after digits, else 1. "milhões" is
 * spelled in octal: a hex escape would swallow the "e" that follows it. */
static double scale_after(const char *s) {
    static const char *const MILLIONS[] = {"milh\xc3\xa3o", "milhao", "milh\303\265es", "milhoes"};
    size_t i, n;
    while (*s == ' ') ++s;
    for (i = 0; i < 4; ++i) {
        n = strlen(MILLIONS[i]);
        if (!strncmp(s, MILLIONS[i], n) && !text_is_letter((char)(s[n] | 32))) return 1e6;
    }
    return !strncmp(s, "mil", 3) && !text_is_letter((char)(s[3] | 32)) ? 1000 : 1;
}

/* Digit runs with separators, both readings when ambiguous ("1.500"), and
 * times a scale word that follows ("2 mil" is also 2000). */
static void digits_in(const char *text, Numbers *ns) {
    const char *s = text;
    while (*s) {
        const char *start;
        double v, scale;
        int k;
        while (*s && !is_digit(*s)) ++s;
        if (!*s) break;
        start = s;
        while (is_digit(*s) || ((*s == '.' || *s == ',') && is_digit(s[1]))) ++s;
        scale = scale_after(s);
        for (k = 0; k < 2; ++k)
            if (number_value(start, (size_t)(s - start), k ? NULL : text, &v)) {
                add_number(ns, v);
                if (scale != 1) add_number(ns, v * scale);
            }
    }
}

static int has_percent(const char *text, const char *folded) {
    static const char *const words[] = {"porcento", "porcentagem", "percentual"};
    return strchr(text, '%') || strstr(folded, "por cento") || text_has_word(folded, words, 3);
}

/* ---- relevance and grounding -------------------------------------------------- */

static const char *const CALC_WORDS[] = {
    "quanto", "quanta", "quantos", "quantas", "qnto", "qto", "qtos", "mais", "menos", "vezes", "multiplicado",
    "multiplicada", "multiplique", "multiplicar", "dividido", "dividida", "divididos", "divida", "dividir",
    "divide", "divido", "dividindo", "soma", "somar", "some", "somando", "subtraia", "subtrair", "subtraido",
    "porcento", "porcentagem", "percentual", "cento", "desconto", "juros", "juro", "raiz", "elevado", "elevada",
    "potencia", "quadrado", "cubo", "metade", "dobro", "triplo", "terco", "media", "total", "parcela",
    "parcelas", "prestacao", "prestacoes", "troco", "gorjeta", "calcule", "calcular", "calcula", "conta",
    "resultado", "rende", "render", "rendimento", "lucro", "aumento", "reajuste", "acrescimo", "fica", "custa"};

/* The nearest non-space byte before t[i] (backward) or after it. */
static char neighbor(const char *t, size_t i, int forward) {
    if (forward) {
        for (++i; t[i] == ' '; ++i) {}
        return t[i];
    }
    while (i > 0 && t[i - 1] == ' ') --i;
    return i > 0 ? t[i - 1] : 0;
}

/* An operator between two operands ("2+2", "3 * (4"): "+55 11" is a phone
 * number, "?a=1" a link. "-", "/" and "x" also write phone numbers, dates and
 * words, so they count only when spaced on both sides. */
static int has_operator(const char *t) {
    size_t i;
    if (strstr(t, "\xc3\x97") || strstr(t, "\xc3\xb7")) return 1;
    for (i = 0; t[i]; ++i) {
        char before, after;
        int spaced = i > 0 && t[i - 1] == ' ' && t[i + 1] == ' ';
        if (!strchr("+*^", t[i]) && !(strchr("-/x", t[i]) && spaced)) continue;
        before = neighbor(t, i, 0);
        after = neighbor(t, i, 1);
        if ((is_digit(before) || before == ')') && (is_digit(after) || after == '(')) return 1;
    }
    return 0;
}

/* A number other than the article "um"/"uma" ("Seja breve uma vez mais"). */
static int has_number(const char *text, const char *folded) {
    size_t i;
    if (strpbrk(text, "0123456789")) return 1;
    for (i = 0; i < sizeof(NUMBER_WORDS) / sizeof(NUMBER_WORDS[0]); ++i)
        if (NUMBER_WORDS[i].value != 1 && text_find_word(folded, folded, NUMBER_WORDS[i].word)) return 1;
    return 0;
}

int calc_relevant(const char *text) {
    char folded[FOLD_MAX];
    text_fold_tail(text, folded, sizeof(folded));
    if (!has_number(text, folded)) return 0;
    return has_operator(text) || text_has_word(folded, CALC_WORDS, sizeof(CALC_WORDS) / sizeof(CALC_WORDS[0]));
}

static int near(double a, double b) { return fabs(a - b) <= 1e-9 * (fabs(b) > 1 ? fabs(b) : 1); }

static int in_list(double v, const double *list, int n) {
    int i;
    for (i = 0; i < n; ++i)
        if (near(v, list[i])) return 1;
    return 0;
}

int calc_grounded(const char *expr, const char *text) {
    static const double CONSTANTS[] = {0.5, 1, 2, 3, 4, 10, 12, 100, 1000};
    char folded[FOLD_MAX];
    double value, derived[3 * MAX_SAID];
    Numbers said;
    Parser ps;
    int i, own = 0, nd = 0;
    if (!evaluate(&ps, expr, text, &value)) return 0;
    memset(&said, 0, sizeof(said));
    text_fold_tail(text, folded, sizeof(folded));
    digits_in(text, &said);
    words_in(folded, &said);
    if (has_percent(text, folded))
        for (i = 0; i < said.n; ++i) {
            derived[nd++] = said.v[i] / 100;
            derived[nd++] = 1 - said.v[i] / 100;
            derived[nd++] = 1 + said.v[i] / 100;
        }
    for (i = 0; i < ps.count; ++i) {
        double v = ps.nums[i];
        if (in_list(v, said.v, said.n)) {
            own = 1;
            continue;
        }
        if (!in_list(v, derived, nd) && !in_list(v, CONSTANTS, (int)(sizeof(CONSTANTS) / sizeof(CONSTANTS[0]))))
            return 0;
    }
    return own;
}
