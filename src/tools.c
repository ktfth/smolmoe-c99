#include "tools.h"
#include "calc.h"
#include "calendar.h"
#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX_ARGS 8
#define ARG_KEY_MAX 32
#define ARG_VALUE_MAX 96
#define NESTED_ARGS_MAX 512 /* "arguments" given as a JSON string */
#define MAX_MENTIONED 8     /* dates looked at in one user message */

static Date date_of(const ToolClock *c) {
    Date d;
    d.year = c->year;
    d.month = c->month;
    d.day = c->day;
    return d;
}

static int digits(const char *s, int n) {
    int v = 0, i;
    for (i = 0; i < n; ++i) {
        if (s[i] < '0' || s[i] > '9') return -1;
        v = v * 10 + (s[i] - '0');
    }
    return v;
}

int tool_clock_parse(const char *s, ToolClock *out) {
    ToolClock c;
    if (strlen(s) != 16 || s[4] != '-' || s[7] != '-' || s[10] != 'T' || s[13] != ':') return 0;
    c.year = digits(s, 4);
    c.month = digits(s + 5, 2);
    c.day = digits(s + 8, 2);
    c.hour = digits(s + 11, 2);
    c.minute = digits(s + 14, 2);
    if (!date_valid(date_of(&c)) || c.hour < 0 || c.hour > 23 || c.minute < 0 || c.minute > 59) return 0;
    *out = c;
    return 1;
}

ToolClock tool_clock_now(void) {
    time_t t = time(NULL);
    const struct tm *tm = t == (time_t)-1 ? NULL : localtime(&t);
    ToolClock c;
    if (!tm) die("cannot read the local time");
    c.year = tm->tm_year + 1900;
    c.month = tm->tm_mon + 1;
    c.day = tm->tm_mday;
    c.hour = tm->tm_hour;
    c.minute = tm->tm_min;
    return c;
}

/* ---- strict JSON for one call --------------------------------------------- */

typedef enum { ARG_STRING, ARG_NUMBER, ARG_LITERAL } ArgKind;

typedef struct Arg {
    char key[ARG_KEY_MAX], value[ARG_VALUE_MAX];
    ArgKind kind;
} Arg;

typedef struct Call {
    char name[TOOL_NAME_MAX];
    Arg args[MAX_ARGS];
    int nargs, has_name, has_args;
} Call;

typedef struct Json {
    const char *p, *end;
} Json;

static void skip_ws(Json *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' || *j->p == '\n' || *j->p == '\r')) ++j->p;
}

static int eat(Json *j, char c) {
    skip_ws(j);
    if (j->p < j->end && *j->p == c) {
        ++j->p;
        return 1;
    }
    return 0;
}

static int peek(Json *j, char c) {
    skip_ws(j);
    return j->p < j->end && *j->p == c;
}

static int hex4(Json *j, unsigned *v) {
    int i;
    *v = 0;
    if (j->end - j->p < 4) return 0;
    for (i = 0; i < 4; ++i) {
        char c = *j->p++;
        unsigned d = c >= '0' && c <= '9' ? (unsigned)(c - '0')
                   : c >= 'a' && c <= 'f' ? (unsigned)(c - 'a' + 10)
                   : c >= 'A' && c <= 'F' ? (unsigned)(c - 'A' + 10) : 16u;
        if (d > 15) return 0;
        *v = *v << 4 | d;
    }
    return 1;
}

static int put_utf8(char *out, size_t cap, size_t *n, unsigned cp) {
    unsigned char b[4];
    size_t len = cp < 0x80 ? 1 : cp < 0x800 ? 2 : cp < 0x10000 ? 3 : 4, i;
    if (!cp || *n + len >= cap) return 0; /* no NUL: names and values are C strings */
    if (len == 1) b[0] = (unsigned char)cp;
    else if (len == 2) { b[0] = (unsigned char)(0xc0 | cp >> 6); b[1] = (unsigned char)(0x80 | (cp & 0x3f)); }
    else if (len == 3) {
        b[0] = (unsigned char)(0xe0 | cp >> 12);
        b[1] = (unsigned char)(0x80 | (cp >> 6 & 0x3f));
        b[2] = (unsigned char)(0x80 | (cp & 0x3f));
    } else {
        b[0] = (unsigned char)(0xf0 | cp >> 18);
        b[1] = (unsigned char)(0x80 | (cp >> 12 & 0x3f));
        b[2] = (unsigned char)(0x80 | (cp >> 6 & 0x3f));
        b[3] = (unsigned char)(0x80 | (cp & 0x3f));
    }
    for (i = 0; i < len; ++i) out[(*n)++] = (char)b[i];
    return 1;
}

/* \uXXXX, joining a surrogate pair; a lone surrogate is invalid. */
static int unicode_escape(Json *j, unsigned *cp) {
    unsigned lo;
    if (!hex4(j, cp) || (*cp >= 0xdc00 && *cp <= 0xdfff)) return 0;
    if (*cp < 0xd800 || *cp > 0xdbff) return 1;
    if (j->end - j->p < 2 || j->p[0] != '\\' || j->p[1] != 'u') return 0;
    j->p += 2;
    if (!hex4(j, &lo) || lo < 0xdc00 || lo > 0xdfff) return 0;
    *cp = 0x10000 + ((*cp - 0xd800) << 10) + (lo - 0xdc00);
    return 1;
}

static int escape_char(Json *j, unsigned *cp) {
    char c;
    if (j->p >= j->end) return 0;
    c = *j->p++;
    switch (c) {
    case '"': case '\\': case '/': *cp = (unsigned char)c; return 1;
    case 'b': *cp = '\b'; return 1;
    case 'f': *cp = '\f'; return 1;
    case 'n': *cp = '\n'; return 1;
    case 'r': *cp = '\r'; return 1;
    case 't': *cp = '\t'; return 1;
    case 'u': return unicode_escape(j, cp);
    default: return 0;
    }
}

/* A JSON string into out (UTF-8, NUL-terminated); 0 if malformed or too long. */
static int parse_string(Json *j, char *out, size_t cap) {
    size_t n = 0;
    if (!eat(j, '"')) return 0;
    while (j->p < j->end) {
        unsigned char c = (unsigned char)*j->p++;
        unsigned cp;
        if (c == '"') {
            out[n] = 0;
            return 1;
        }
        if (c < 0x20) return 0;
        if (c != '\\') {
            if (n + 1 >= cap) return 0;
            out[n++] = (char)c;
            continue;
        }
        if (!escape_char(j, &cp) || !put_utf8(out, cap, &n, cp)) return 0;
    }
    return 0;
}

/* JSON number grammar, kept as text: -?(0|[1-9]d*)(.d+)?([eE][+-]?d+)? */
static int parse_number(Json *j, char *out, size_t cap) {
    const char *s = j->p, *p = s;
    size_t n;
    if (p < j->end && *p == '-') ++p;
    if (p >= j->end || *p < '0' || *p > '9') return 0;
    if (*p == '0') ++p;
    else while (p < j->end && *p >= '0' && *p <= '9') ++p;
    if (p < j->end && *p == '.') {
        if (++p >= j->end || *p < '0' || *p > '9') return 0;
        while (p < j->end && *p >= '0' && *p <= '9') ++p;
    }
    if (p < j->end && (*p == 'e' || *p == 'E')) {
        if (++p < j->end && (*p == '+' || *p == '-')) ++p;
        if (p >= j->end || *p < '0' || *p > '9') return 0;
        while (p < j->end && *p >= '0' && *p <= '9') ++p;
    }
    n = (size_t)(p - s);
    if (n >= cap) return 0;
    memcpy(out, s, n);
    out[n] = 0;
    j->p = p;
    return 1;
}

static int parse_literal(Json *j, Arg *a) {
    static const char *const words[3] = {"true", "false", "null"};
    int i;
    for (i = 0; i < 3; ++i) {
        size_t n = strlen(words[i]);
        if ((size_t)(j->end - j->p) >= n && !memcmp(j->p, words[i], n)) {
            memcpy(a->value, words[i], n + 1);
            a->kind = ARG_LITERAL;
            j->p += n;
            return 1;
        }
    }
    return 0;
}

/* Arguments are flat: strings, numbers, true/false/null. */
static int parse_scalar(Json *j, Arg *a) {
    skip_ws(j);
    if (peek(j, '"')) {
        a->kind = ARG_STRING;
        return parse_string(j, a->value, sizeof(a->value));
    }
    if (j->p < j->end && (*j->p == '-' || (*j->p >= '0' && *j->p <= '9'))) {
        a->kind = ARG_NUMBER;
        return parse_number(j, a->value, sizeof(a->value));
    }
    return parse_literal(j, a);
}

static int has_key(const Call *c, int count, const char *key) {
    int i;
    for (i = 0; i < count; ++i)
        if (!strcmp(c->args[i].key, key)) return 1;
    return 0;
}

static int parse_arguments(Json *j, Call *c) {
    if (!eat(j, '{')) return 0;
    if (eat(j, '}')) return 1;
    do {
        Arg *a;
        if (c->nargs >= MAX_ARGS) return 0;
        a = &c->args[c->nargs];
        if (!parse_string(j, a->key, sizeof(a->key)) || has_key(c, c->nargs, a->key)) return 0;
        if (!eat(j, ':') || !parse_scalar(j, a)) return 0;
        ++c->nargs;
    } while (eat(j, ','));
    return eat(j, '}');
}

/* "arguments": {...}, or the same object encoded as a JSON string. */
static int parse_arguments_value(Json *j, Call *c) {
    char inner[NESTED_ARGS_MAX];
    Json k;
    if (c->has_args) return 0;
    c->has_args = 1;
    if (!peek(j, '"')) return parse_arguments(j, c);
    if (!parse_string(j, inner, sizeof(inner))) return 0;
    k.p = inner;
    k.end = inner + strlen(inner);
    if (!parse_arguments(&k, c)) return 0;
    skip_ws(&k);
    return k.p == k.end;
}

static int parse_member(Json *j, Call *c) {
    char key[16];
    if (!parse_string(j, key, sizeof(key)) || !eat(j, ':')) return 0;
    if (!strcmp(key, "arguments")) return parse_arguments_value(j, c);
    if (strcmp(key, "name") || c->has_name) return 0;
    c->has_name = 1;
    return parse_string(j, c->name, sizeof(c->name)) && c->name[0];
}

/* {"name": "...", "arguments": {...}} and nothing else around it. */
static int parse_call(const char *s, size_t n, Call *c) {
    Json j;
    j.p = s;
    j.end = s + n;
    memset(c, 0, sizeof(*c));
    if (!eat(&j, '{')) return 0;
    if (!peek(&j, '}')) {
        do {
            if (!parse_member(&j, c)) return 0;
        } while (eat(&j, ','));
    }
    if (!eat(&j, '}')) return 0;
    skip_ws(&j);
    return j.p == j.end && c->has_name;
}

/* ---- tools ----------------------------------------------------------------- */

static const Arg *find_arg(const Call *c, const char *key) {
    int i;
    for (i = 0; i < c->nargs; ++i)
        if (!strcmp(c->args[i].key, key)) return &c->args[i];
    return NULL;
}

/* A non-negative integer, written as a number or quoted (small models quote numbers). */
static int int_arg(const Call *c, const char *key, int lo, int hi, int *out) {
    const Arg *a = find_arg(c, key);
    const char *s;
    long v = 0;
    int count = 0;
    if (!a || a->kind == ARG_LITERAL) return 0;
    for (s = a->value; *s; ++s) {
        if (*s < '0' || *s > '9' || ++count > 6) return 0;
        v = v * 10 + (*s - '0');
    }
    if (!count || v < lo || v > hi) return 0;
    *out = (int)v;
    return 1;
}

/* JSON string contents: quotes, backslashes and control characters escaped. */
static void json_escape(char *out, size_t cap, const char *s) {
    size_t n = 0;
    for (; *s && n + 7 < cap; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            out[n++] = '\\';
            out[n++] = (char)c;
        } else if (c < 0x20) {
            n += (size_t)snprintf(out + n, cap - n, "\\u%04x", c);
        } else {
            out[n++] = (char)c;
        }
    }
    out[n] = 0;
}

static int error_result(char result[TOOL_RESULT_MAX], const char *fmt, ...) SM_PRINTF(2, 3);
static int error_result(char result[TOOL_RESULT_MAX], const char *fmt, ...) {
    char message[TOOL_RESULT_MAX], escaped[TOOL_RESULT_MAX - 16]; /* room for {"erro": ""} */
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(message, sizeof(message), fmt, ap);
    va_end(ap);
    json_escape(escaped, sizeof(escaped), message);
    snprintf(result, TOOL_RESULT_MAX, "{\"erro\": \"%s\"}", escaped);
    return 0;
}

/* Results carry a ready sentence ("resumo"): small models copy it far more
 * reliably than they phrase a number themselves. */
static int current_date(const ToolClock *now, char result[TOOL_RESULT_MAX]) {
    char date[64];
    Date today = date_of(now);
    const char *weekday = date_weekday(today);
    date_long(today, date, sizeof(date));
    snprintf(result, TOOL_RESULT_MAX,
             "{\"resumo\": \"Hoje é %s, %s. Agora são %02d:%02d.\", "
             "\"data\": \"%s\", \"dia_da_semana\": \"%s\", \"hora\": \"%02d:%02d\"}",
             weekday, date, now->hour, now->minute, date, weekday, now->hour, now->minute);
    return 1;
}

static int next_occurrence(int month, int day, Date today, Date *out) {
    char numeric[24];
    snprintf(numeric, sizeof(numeric), "%d/%d", day, month);
    return date_parse(numeric, today, out);
}

/* The model's argument: the date as text ("25 de dezembro", "Natal") or, as
 * small models sometimes send, month and day integers. A past date it gives
 * (an invented year) becomes the next occurrence of that day. */
static int model_date(const Call *c, Date today, Date *out) {
    const Arg *text = find_arg(c, "data");
    int m, d;
    if (text && text->kind == ARG_STRING) {
        if (!date_parse(text->value, today, out)) return 0;
        return date_days(*out) >= date_days(today) || next_occurrence(out->month, out->day, today, out);
    }
    return int_arg(c, "mes", 1, 12, &m) && int_arg(c, "dia", 1, 31, &d) && next_occurrence(m, d, today, out);
}

/* The user's own words are the ground truth. One date in the message: that one.
 * Several: the model's argument picks among them, else the last one mentioned
 * (the question usually ends with it). None: the model's argument. */
static int target_date(const Call *c, const char *user_text, Date today, Date *out) {
    Date said[MAX_MENTIONED], proposed;
    int n = date_find(user_text, today, said, MAX_MENTIONED), has_proposal = model_date(c, today, &proposed), i;
    if (!n) {
        if (has_proposal) *out = proposed;
        return has_proposal;
    }
    for (i = 0; n > 1 && has_proposal && i < n; ++i)
        if (date_days(said[i]) == date_days(proposed)) {
            *out = proposed;
            return 1;
        }
    *out = said[n - 1];
    return 1;
}

/* Length of the longest prefix within max bytes that ends on a UTF-8 character boundary. */
static int utf8_prefix(const char *s, int max) {
    int n = (int)strlen(s);
    if (n <= max) return n;
    for (n = max; n > 0 && ((unsigned char)s[n] & 0xc0) == 0x80; --n) {}
    return n;
}

/* Only the count. Prefixing today's date ("Hoje é ...") was measured and made
 * the model write "49 dias atrás" again (single session: 10/17 -> 4/17). */
static void summary(char *out, size_t cap, long days, const char *date, const char *weekday) {
    if (days > 1) snprintf(out, cap, "Faltam %ld dias para %s (%s).", days, date, weekday);
    else if (days == 1) snprintf(out, cap, "Falta 1 dia para %s (%s).", date, weekday);
    else if (days == 0) snprintf(out, cap, "É hoje: %s (%s).", date, weekday);
    else if (days == -1) snprintf(out, cap, "Essa data já passou: %s (%s) foi ontem.", date, weekday);
    else snprintf(out, cap, "Essa data já passou: %s (%s) foi há %ld dias.", date, weekday, -days);
}

/* Next occurrence on or after today (0 days: today), or the exact date when a year is given. */
static int days_until(const Call *c, const char *user_text, const ToolClock *now, char result[TOOL_RESULT_MAX]) {
    char date[64], text[160];
    const Arg *given = find_arg(c, "data");
    Date today = date_of(now), target;
    long days;
    if (!target_date(c, user_text, today, &target)) {
        const char *shown = given && given->kind == ARG_STRING ? given->value : "";
        return error_result(result, "não entendi a data \"%.*s\"; use, por exemplo, 25 de dezembro ou 25/12",
                            utf8_prefix(shown, 60), shown);
    }
    days = date_days(target) - date_days(today);
    date_long(target, date, sizeof(date));
    summary(text, sizeof(text), days, date, date_weekday(target));
    snprintf(result, TOOL_RESULT_MAX, "{\"resumo\": \"%s\", \"data\": \"%s\", \"dia_da_semana\": \"%s\", \"dias\": %ld}",
             text, date, date_weekday(target), days);
    return 1;
}

/* ---- calcular --------------------------------------------------------------- */

static const char *expression_arg(const Call *c) {
    const Arg *a = find_arg(c, "expressao");
    if (!a) a = find_arg(c, "expression");
    return a && (a->kind == ARG_STRING || a->kind == ARG_NUMBER) ? a->value : NULL;
}

/* The runtime does the arithmetic; the user's message decides what "1.500" means. */
static int calculate(const Call *c, const char *user_text, char result[TOOL_RESULT_MAX]) {
    char err[96], formatted[64], escaped[2 * ARG_VALUE_MAX];
    double value;
    const char *expr = expression_arg(c);
    if (!expr) return error_result(result, "falta o argumento \"expressao\", com a conta a fazer");
    if (!calc_eval(expr, user_text, &value, err, sizeof(err))) return error_result(result, "não consegui calcular: %s", err);
    calc_format(value, formatted, sizeof(formatted));
    json_escape(escaped, sizeof(escaped), expr);
    snprintf(result, TOOL_RESULT_MAX, "{\"resumo\": \"O resultado é %s.\", \"expressao\": \"%s\", \"resultado\": \"%s\"}",
             formatted, escaped, formatted);
    return 1;
}

int tools_relevant(const char *user_text) { return date_or_time_mentioned(user_text) || calc_relevant(user_text); }

const char *tools_forced_choice(const char *user_text) {
    return calc_relevant(user_text) && !date_question(user_text) ? "calcular" : NULL;
}

size_t tools_call_prefix(const char *tool, char *out, size_t cap) {
    int w;
    if (strcmp(tool, "calcular")) return 0;
    w = snprintf(out, cap, "\n{\"name\": \"calcular\", \"arguments\": {\"expressao\": \"");
    return w < 0 || (size_t)w >= cap ? 0 : (size_t)w;
}

int tool_call_grounded(const char *call, size_t n, const char *user_text) {
    Call c;
    Date named, reference;
    reference.year = 2000; /* any valid day: only whether a date is named matters */
    reference.month = 1;
    reference.day = 1;
    if (!parse_call(call, n, &c)) return 0;
    if (!strcmp(c.name, "data_atual")) return 1;
    if (!strcmp(c.name, "calcular")) return expression_arg(&c) && calc_grounded(expression_arg(&c), user_text);
    /* "3/4 de 200" is a fraction, not the 3rd of April. */
    return !strcmp(c.name, "dias_ate") && date_question(user_text) && date_find(user_text, reference, &named, 1) > 0;
}

int tool_run(const char *call, size_t n, const char *user_text, const ToolClock *now, char name[TOOL_NAME_MAX],
             char result[TOOL_RESULT_MAX]) {
    Call c;
    if (!parse_call(call, n, &c)) {
        snprintf(name, TOOL_NAME_MAX, "?");
        return error_result(result, "chamada inválida: use {\"name\": ..., \"arguments\": {...}}");
    }
    snprintf(name, TOOL_NAME_MAX, "%s", c.name);
    if (!strcmp(c.name, "data_atual")) return current_date(now, result);
    if (!strcmp(c.name, "dias_ate")) return days_until(&c, user_text, now, result);
    if (!strcmp(c.name, "calcular")) return calculate(&c, user_text, result);
    return error_result(result, "função desconhecida: %s", c.name);
}

/* ---- system prompt -------------------------------------------------------- */

/* Same layout as the Qwen2.5 chat template (json.dumps with ", " and ": "). */
static const char *const TOOL_SPECS[] = {
    "{\"type\": \"function\", \"function\": {\"name\": \"data_atual\", \"description\": "
    "\"Data de hoje, dia da semana e hora atual. Use só para perguntas sobre hoje, a data ou as horas.\", "
    "\"parameters\": {\"type\": \"object\", \"properties\": {}, \"required\": []}}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"dias_ate\", \"description\": "
    "\"Quantos dias faltam, a partir de hoje, até uma data ou um feriado. Use só quando o usuário perguntar "
    "quanto tempo falta para uma data.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"data\": "
    "{\"type\": \"string\", \"description\": \"A data ou o feriado exatamente como o usuário escreveu\"}}, "
    "\"required\": [\"data\"]}}}",
    /* No example expression: models copy example numbers (DECISIONS, iteration 4). */
    "{\"type\": \"function\", \"function\": {\"name\": \"calcular\", \"description\": "
    "\"Faz uma conta exata com os números do usuário: soma, subtração, multiplicação, divisão, porcentagem, "
    "potência e raiz. Use sempre que a resposta depender de um cálculo.\", \"parameters\": {\"type\": \"object\", "
    "\"properties\": {\"expressao\": {\"type\": \"string\", \"description\": \"A conta, com + - * / ^ e "
    "parênteses\"}}, \"required\": [\"expressao\"]}}}",
};

size_t tools_system_prompt(const char *system, char *out, size_t cap) {
    size_t n, i;
    int w = snprintf(out, cap, "%s\n\n# Tools\n\nYou may call one or more functions to assist with the user query."
                               "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
                               "<tools>", system);
    if (w < 0 || (size_t)w >= cap) return 0;
    n = (size_t)w;
    for (i = 0; i < sizeof(TOOL_SPECS) / sizeof(TOOL_SPECS[0]); ++i) {
        w = snprintf(out + n, cap - n, "\n%s", TOOL_SPECS[i]);
        if (w < 0 || (size_t)w >= cap - n) return 0;
        n += (size_t)w;
    }
    w = snprintf(out + n, cap - n, "\n</tools>\n\nFor each function call, return a json object with function name "
                                   "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
                                   "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>");
    if (w < 0 || (size_t)w >= cap - n) return 0;
    return n + (size_t)w;
}
