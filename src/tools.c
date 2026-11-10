#include "tools.h"
#include "calc.h"
#include "calendar.h"
#include "textfold.h"
#include "util.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
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
    if (!tm) sm_fail(SM_ERR_IO, "cannot read the local time");
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

static int only_closing_braces(const char *p, const char *end) {
    for (; p < end; ++p)
        if (*p != '}' && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') return 0;
    return 1;
}

/* A JSON string into out (UTF-8, NUL-terminated); 0 if malformed or too long. */
static int parse_string_val(Json *j, char *out, size_t cap, int allow_unquoted_end) {
    size_t n = 0;
    if (!eat(j, '"')) return 0;
    while (j->p < j->end) {
        unsigned char c = (unsigned char)*j->p++;
        unsigned cp;
        if (c == '"') {
            out[n] = 0;
            return 1;
        }
        if (allow_unquoted_end && c == '}' && only_closing_braces(j->p - 1, j->end)) {
            --j->p;
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

static int parse_string(Json *j, char *out, size_t cap) { return parse_string_val(j, out, cap, 0); }

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
        return parse_string_val(j, a->value, sizeof(a->value), 1);
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
    if (!eat(&j, '}') && !(c->has_name && c->has_args)) return 0;
    skip_ws(&j);
    return j.p == j.end && c->has_name;
}

/* A flat result value: a string (into out when wanted), a number or a literal. */
static int result_value(Json *j, char *out, size_t cap) {
    char skipped[TOOL_RESULT_MAX];
    Arg literal;
    skip_ws(j);
    if (peek(j, '"')) return out ? parse_string(j, out, cap) : parse_string(j, skipped, sizeof(skipped));
    if (out) return 0;
    if (j->p < j->end && (*j->p == '-' || (*j->p >= '0' && *j->p <= '9')))
        return parse_number(j, skipped, sizeof(skipped));
    return parse_literal(j, &literal);
}

static int result_string(const char *result, const char *key, char *out, size_t cap) {
    char name[ARG_KEY_MAX];
    int found = 0;
    Json j;
    j.p = result;
    j.end = result + strlen(result);
    if (!eat(&j, '{')) return 0;
    if (!peek(&j, '}')) {
        do {
            int wanted;
            if (!parse_string(&j, name, sizeof(name)) || !eat(&j, ':')) return 0;
            wanted = !found && !strcmp(name, key);
            if (!result_value(&j, wanted ? out : NULL, cap)) return 0;
            found |= wanted;
        } while (eat(&j, ','));
    }
    if (!eat(&j, '}')) return 0;
    skip_ws(&j);
    return found && j.p == j.end;
}

int tool_result_string(const char *result, const char *key, char *out, size_t cap) {
    int ok = cap && result_string(result, key, out, cap);
    if (cap && !ok) out[0] = 0; /* a failed parse may have written part of a value */
    return ok;
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

/* ---- converter (unidades) ------------------------------------------------- */

typedef enum { UNIT_LEN, UNIT_MASS, UNIT_TEMP, UNIT_SPEED, UNIT_VOL, UNIT_DATA } UnitKind;

typedef struct Unit {
    const char *name, *symbol;
    UnitKind kind;
    double factor;
    int short_form; /* 1: single-letter or short token that must follow a number/marker in free text */
} Unit;

/* Ordered longest-first when prefixes overlap ("km/h" before "km", "milimetro" before "m"). */
static const Unit UNITS[] = {
    {"quilometros", "km", UNIT_LEN, 1000.0, 0},
    {"quilometro", "km", UNIT_LEN, 1000.0, 0},
    {"centimetros", "cm", UNIT_LEN, 0.01, 0},
    {"centimetro", "cm", UNIT_LEN, 0.01, 0},
    {"milimetros", "mm", UNIT_LEN, 0.001, 0},
    {"milimetro", "mm", UNIT_LEN, 0.001, 0},
    {"polegadas", "pol", UNIT_LEN, 0.0254, 0},
    {"polegada", "pol", UNIT_LEN, 0.0254, 0},
    {"metros", "m", UNIT_LEN, 1.0, 0},
    {"metro", "m", UNIT_LEN, 1.0, 0},
    {"milhas", "mi", UNIT_LEN, 1609.344, 0},
    {"milha", "mi", UNIT_LEN, 1609.344, 0},
    {"jardas", "yd", UNIT_LEN, 0.9144, 0},
    {"jarda", "yd", UNIT_LEN, 0.9144, 0},
    {"pes", "ft", UNIT_LEN, 0.3048, 0},
    {"pol", "pol", UNIT_LEN, 0.0254, 0},
    {"km/h", "km/h", UNIT_SPEED, 1.0 / 3.6, 0},
    {"kmh", "km/h", UNIT_SPEED, 1.0 / 3.6, 0},
    {"m/s", "m/s", UNIT_SPEED, 1.0, 0},
    {"mph", "mph", UNIT_SPEED, 0.44704, 0},
    {"km", "km", UNIT_LEN, 1000.0, 0},
    {"cm", "cm", UNIT_LEN, 0.01, 0},
    {"mm", "mm", UNIT_LEN, 0.001, 0},
    {"mi", "mi", UNIT_LEN, 1609.344, 0},
    {"yd", "yd", UNIT_LEN, 0.9144, 0},
    {"ft", "ft", UNIT_LEN, 0.3048, 0},
    {"pe", "ft", UNIT_LEN, 0.3048, 1},
    {"in", "pol", UNIT_LEN, 0.0254, 1},
    {"m", "m", UNIT_LEN, 1.0, 1},
    {"quilogramas", "kg", UNIT_MASS, 1000.0, 0},
    {"quilograma", "kg", UNIT_MASS, 1000.0, 0},
    {"miligramas", "mg", UNIT_MASS, 0.001, 0},
    {"miligrama", "mg", UNIT_MASS, 0.001, 0},
    {"toneladas", "t", UNIT_MASS, 1000000.0, 0},
    {"tonelada", "t", UNIT_MASS, 1000000.0, 0},
    {"quilos", "kg", UNIT_MASS, 1000.0, 0},
    {"quilo", "kg", UNIT_MASS, 1000.0, 0},
    {"gramas", "g", UNIT_MASS, 1.0, 0},
    {"grama", "g", UNIT_MASS, 1.0, 0},
    {"libras", "lb", UNIT_MASS, 453.59237, 0},
    {"libra", "lb", UNIT_MASS, 453.59237, 0},
    {"oncas", "oz", UNIT_MASS, 28.349523125, 0},
    {"onca", "oz", UNIT_MASS, 28.349523125, 0},
    {"ton", "t", UNIT_MASS, 1000000.0, 0},
    {"lbs", "lb", UNIT_MASS, 453.59237, 0},
    {"kg", "kg", UNIT_MASS, 1000.0, 0},
    {"mg", "mg", UNIT_MASS, 0.001, 0},
    {"lb", "lb", UNIT_MASS, 453.59237, 0},
    {"oz", "oz", UNIT_MASS, 28.349523125, 0},
    {"t", "t", UNIT_MASS, 1000000.0, 1},
    {"g", "g", UNIT_MASS, 1.0, 1},
    {"fahrenheit", "°F", UNIT_TEMP, 1.0, 0},
    {"celsius", "°C", UNIT_TEMP, 1.0, 0},
    {"kelvins", "K", UNIT_TEMP, 1.0, 0},
    {"kelvin", "K", UNIT_TEMP, 1.0, 0},
    {"c", "°C", UNIT_TEMP, 1.0, 1},
    {"f", "°F", UNIT_TEMP, 1.0, 1},
    {"k", "K", UNIT_TEMP, 1.0, 1},
    {"mililitros", "mL", UNIT_VOL, 0.001, 0},
    {"mililitro", "mL", UNIT_VOL, 0.001, 0},
    {"litros", "L", UNIT_VOL, 1.0, 0},
    {"litro", "L", UNIT_VOL, 1.0, 0},
    {"galoes", "gal", UNIT_VOL, 3.785411784, 0},
    {"galao", "gal", UNIT_VOL, 3.785411784, 0},
    {"gal", "gal", UNIT_VOL, 3.785411784, 0},
    {"ml", "mL", UNIT_VOL, 0.001, 0},
    {"l", "L", UNIT_VOL, 1.0, 1},
    {"terabytes", "TB", UNIT_DATA, 1099511627776.0, 0},
    {"terabyte", "TB", UNIT_DATA, 1099511627776.0, 0},
    {"gigabytes", "GB", UNIT_DATA, 1073741824.0, 0},
    {"gigabyte", "GB", UNIT_DATA, 1073741824.0, 0},
    {"megabytes", "MB", UNIT_DATA, 1048576.0, 0},
    {"megabyte", "MB", UNIT_DATA, 1048576.0, 0},
    {"kilobytes", "KB", UNIT_DATA, 1024.0, 0},
    {"kilobyte", "KB", UNIT_DATA, 1024.0, 0},
    {"bytes", "B", UNIT_DATA, 1.0, 0},
    {"byte", "B", UNIT_DATA, 1.0, 0},
    {"tib", "TB", UNIT_DATA, 1099511627776.0, 0},
    {"gib", "GB", UNIT_DATA, 1073741824.0, 0},
    {"mib", "MB", UNIT_DATA, 1048576.0, 0},
    {"kib", "KB", UNIT_DATA, 1024.0, 0},
    {"tb", "TB", UNIT_DATA, 1099511627776.0, 0},
    {"gb", "GB", UNIT_DATA, 1073741824.0, 0},
    {"mb", "MB", UNIT_DATA, 1048576.0, 0},
    {"kb", "KB", UNIT_DATA, 1024.0, 0},
    {"b", "B", UNIT_DATA, 1.0, 1},
};

static const char *skip_unit_lead(const char *p) {
    for (;;) {
        while (*p == ' ' || *p == '?') ++p; /* '?' comes from '°' in text_fold */
        if (!strncmp(p, "graus ", 6)) p += 6;
        else if (!strncmp(p, "grau ", 5)) p += 5;
        else if (!strncmp(p, "de ", 3)) p += 3;
        else break;
    }
    return p;
}

static const Unit *match_unit_at(const char *p, int allow_short, const char **after) {
    size_t i;
    p = skip_unit_lead(p);
    for (i = 0; i < sizeof(UNITS) / sizeof(UNITS[0]); ++i) {
        size_t len = strlen(UNITS[i].name);
        if (UNITS[i].short_form && !allow_short) continue;
        if (!strncmp(p, UNITS[i].name, len) && !text_is_letter(p[len]) && p[len] != '/') {
            if (after) *after = p + len;
            return &UNITS[i];
        }
    }
    return NULL;
}

static const Unit *parse_unit(const char *raw) {
    char folded[64];
    const char *after;
    const Unit *u;
    if (!raw || !text_fold(raw, folded, sizeof(folded))) return NULL;
    u = match_unit_at(folded, 1, &after);
    if (!u) return NULL;
    while (*after == ' ' || *after == '?') ++after;
    return *after == 0 ? u : NULL;
}

static int parse_folded_number(const char *p, double *out, const char **after) {
    char buf[64];
    size_t n = 0;
    const char *s = p;
    char *end;
    if (*s == '-' && s[1] >= '0' && s[1] <= '9') buf[n++] = *s++;
    if (*s < '0' || *s > '9') return 0;
    while (*s >= '0' && *s <= '9' && n + 2 < sizeof(buf)) buf[n++] = *s++;
    if ((*s == ',' || *s == '.') && s[1] >= '0' && s[1] <= '9') {
        buf[n++] = '.';
        ++s;
        while (*s >= '0' && *s <= '9' && n + 1 < sizeof(buf)) buf[n++] = *s++;
    }
    buf[n] = 0;
    *out = strtod(buf, &end);
    if (end != buf + n) return 0;
    *after = s;
    return 1;
}

/* Extracts <val> <from_unit> ... <to_unit> (or "quantos <to_unit> ... <val> <from_unit>")
 * when both units belong to the same category and are distinct. */
static int convert_extract(const char *user_text, double *val_out, const Unit **from_out, const Unit **to_out) {
    static const char *const preps[] = {"em ", "para ", "pra ", "a "};
    static const char *const asks[] = {"quantos ", "quantas "};
    char folded[FOLD_MAX];
    const char *p, *after_num, *after_u1;
    const Unit *early_to = NULL;
    size_t i;
    text_fold_tail(user_text, folded, sizeof(folded));
    for (p = folded; *p; ++p) {
        if (p == folded || !text_is_letter(p[-1])) {
            for (i = 0; i < sizeof(asks) / sizeof(asks[0]); ++i) {
                size_t len = strlen(asks[i]);
                if (!strncmp(p, asks[i], len)) {
                    const Unit *u = match_unit_at(p + len, 0, NULL);
                    if (u) early_to = u;
                }
            }
        }
    }
    for (p = folded; *p; ++p) {
        double v;
        const Unit *u1;
        int deg;
        if (p > folded && (text_is_letter(p[-1]) || (p[-1] >= '0' && p[-1] <= '9'))) continue;
        if (!parse_folded_number(p, &v, &after_num)) continue;
        deg = after_num[0] == '?' || !strncmp(after_num, " ?", 2) || !strncmp(after_num, " grau", 5);
        u1 = match_unit_at(after_num, 1, &after_u1);
        if (!u1) continue;
        if (u1->short_form && u1->kind == UNIT_TEMP && !deg && after_num == skip_unit_lead(after_num)) continue;
        if (early_to && early_to->kind == u1->kind && strcmp(early_to->symbol, u1->symbol) != 0) {
            *val_out = v;
            *from_out = u1;
            *to_out = early_to;
            return 1;
        }
        for (; *after_u1; ++after_u1) {
            if (after_u1 > folded && text_is_letter(after_u1[-1])) continue;
            for (i = 0; i < sizeof(preps) / sizeof(preps[0]); ++i) {
                size_t len = strlen(preps[i]);
                if (!strncmp(after_u1, preps[i], len)) {
                    const Unit *u2 = match_unit_at(after_u1 + len, 1, NULL);
                    if (u2 && u2->kind == u1->kind && strcmp(u2->symbol, u1->symbol) != 0) {
                        *val_out = v;
                        *from_out = u1;
                        *to_out = u2;
                        return 1;
                    }
                }
            }
        }
    }
    return 0;
}

static int convert_relevant(const char *user_text) {
    double v;
    const Unit *from, *to;
    return convert_extract(user_text, &v, &from, &to);
}

static double convert_math(double val, const Unit *from, const Unit *to) {
    if (from->kind == UNIT_TEMP) {
        double c = !strcmp(from->symbol, "°F") ? (val - 32.0) * 5.0 / 9.0
                 : !strcmp(from->symbol, "K")  ? val - 273.15
                                               : val;
        return !strcmp(to->symbol, "°F") ? c * 9.0 / 5.0 + 32.0
             : !strcmp(to->symbol, "K")  ? c + 273.15
                                         : c;
    }
    return val * (from->factor / to->factor);
}

static int model_convert_args(const Call *c, double *val_out, const Unit **from_out, const Unit **to_out) {
    const Arg *v = find_arg(c, "valor");
    const Arg *de = find_arg(c, "de");
    const Arg *para = find_arg(c, "para");
    const char *after;
    if (!v) v = find_arg(c, "value");
    if (!de) de = find_arg(c, "origem");
    if (!para) para = find_arg(c, "destino");
    if (!v || (v->kind != ARG_NUMBER && v->kind != ARG_STRING) || !parse_folded_number(v->value, val_out, &after))
        return 0;
    *from_out = de && de->kind == ARG_STRING ? parse_unit(de->value) : NULL;
    *to_out = para && para->kind == ARG_STRING ? parse_unit(para->value) : NULL;
    return *from_out != NULL && *to_out != NULL;
}

static int convert_unit(const Call *c, const char *user_text, char result[TOOL_RESULT_MAX]) {
    double val, out;
    const Unit *from = NULL, *to = NULL;
    char in_fmt[64], out_fmt[64];
    if (!convert_extract(user_text, &val, &from, &to) && !model_convert_args(c, &val, &from, &to))
        return error_result(result, "informe \"valor\", \"de\" e \"para\" com unidades conhecidas (ex.: km, mi, C, F)");
    if (from->kind != to->kind)
        return error_result(result, "unidades incompatíveis: %s e %s", from->symbol, to->symbol);
    out = convert_math(val, from, to);
    calc_format(val, in_fmt, sizeof(in_fmt));
    calc_format(out, out_fmt, sizeof(out_fmt));
    snprintf(result, TOOL_RESULT_MAX,
             "{\"resumo\": \"%s %s equivalem a %s %s.\", \"valor\": \"%s\", \"de\": \"%s\", \"para\": \"%s\", "
             "\"resultado\": \"%s\"}",
             in_fmt, from->symbol, out_fmt, to->symbol, in_fmt, from->symbol, to->symbol, out_fmt);
    return 1;
}

/* ---- contar (letras, palavras e caracteres) ------------------------------- */

static int count_relevant(const char *user_text) {
    char folded[FOLD_MAX];
    text_fold_tail(user_text, folded, sizeof(folded));
    return strstr(folded, "quantas letras") != NULL || strstr(folded, "quantos caracteres") != NULL ||
           strstr(folded, "quantas palavras") != NULL || strstr(folded, "quantas vezes a letra") != NULL ||
           strstr(folded, "ocorrencias da letra") != NULL;
}

static const char *count_text_arg(const Call *c) {
    const Arg *a = find_arg(c, "texto");
    if (!a) a = find_arg(c, "palavra");
    if (!a) a = find_arg(c, "text");
    return a && a->kind == ARG_STRING && a->value[0] ? a->value : NULL;
}

static const char *count_target_arg(const Call *c) {
    const Arg *a = find_arg(c, "letra");
    if (!a) a = find_arg(c, "alvo");
    if (!a) a = find_arg(c, "caractere");
    return a && a->kind == ARG_STRING && a->value[0] ? a->value : NULL;
}

/* When the user asks "quantas letras 'r' tem em morango?" and the model omits "letra",
 * recover the single letter from the user's question. */
static void infer_target_letter(const char *user_text, char out[8]) {
    char folded[FOLD_MAX];
    const char *p;
    out[0] = 0;
    text_fold_tail(user_text, folded, sizeof(folded));
    p = strstr(folded, "letra");
    if (!p) return;
    p += 5;
    if (*p == 's') ++p;
    while (*p == ' ' || *p == '\'' || *p == '"') ++p;
    if (text_is_letter(p[0]) && (!text_is_letter(p[1]))) {
        if (p[-1] == '\'' || p[-1] == '"' || !strncmp(p + 1, " tem ", 5) || !strncmp(p + 1, " ha ", 4) ||
            !strncmp(p + 1, " aparece", 8) || !strncmp(p + 1, " em ", 4) || !strncmp(p + 1, " na ", 4) ||
            !strncmp(p + 1, " no ", 4)) {
            out[0] = p[0];
            out[1] = 0;
        }
    }
}

static int count_grounded(const Call *c, const char *user_text) {
    char folded_user[FOLD_MAX], folded_text[FOLD_MAX];
    const char *texto = count_text_arg(c);
    if (!texto || !text_fold(texto, folded_text, sizeof(folded_text)) || !folded_text[0]) return 0;
    if (!user_text[0]) return 1;
    if (!count_relevant(user_text)) return 0;
    text_fold_tail(user_text, folded_user, sizeof(folded_user));
    return strstr(folded_user, folded_text) != NULL;
}

static void analyze_utf8(const char *s, int *chars, int *letters, int *words) {
    const unsigned char *p = (const unsigned char *)s;
    int in_word = 0;
    *chars = *letters = *words = 0;
    while (*p) {
        int is_let = 0, is_alnum = 0;
        if ((*p & 0x80) == 0) {
            is_let = (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z');
            is_alnum = is_let || (*p >= '0' && *p <= '9');
            ++p;
        } else if (*p == 0xc3 && p[1] >= 0x80 && p[1] <= 0xbf) {
            is_let = p[1] != 0x97 && p[1] != 0xb7; /* Latin-1 letters except × and ÷ */
            is_alnum = is_let;
            p += 2;
        } else {
            ++p;
            while ((*p & 0xc0) == 0x80) ++p;
        }
        ++(*chars);
        if (is_let) ++(*letters);
        if (is_alnum && !in_word) {
            ++(*words);
            in_word = 1;
        } else if (!is_alnum) {
            in_word = 0;
        }
    }
}

static int count_text(const Call *c, const char *user_text, char result[TOOL_RESULT_MAX]) {
    const char *texto = count_text_arg(c), *letra = count_target_arg(c);
    char inferred[8], folded_text[FOLD_MAX], folded_target[64], esc_text[64], esc_target[16];
    int chars, letters, words, hits = 0;
    if (!texto) return error_result(result, "falta o argumento \"texto\" com a palavra ou frase a contar");
    if (!letra) {
        infer_target_letter(user_text, inferred);
        if (inferred[0]) letra = inferred;
    }
    analyze_utf8(texto, &chars, &letters, &words);
    json_escape(esc_text, sizeof(esc_text), texto);
    if (letra && text_fold(texto, folded_text, sizeof(folded_text)) &&
        text_fold(letra, folded_target, sizeof(folded_target)) && folded_target[0]) {
        size_t tlen = strlen(folded_target);
        const char *p = folded_text;
        while ((p = strstr(p, folded_target)) != NULL) {
            ++hits;
            p += tlen;
        }
        json_escape(esc_target, sizeof(esc_target), letra);
        snprintf(result, TOOL_RESULT_MAX,
                 "{\"resumo\": \"\\\"%s\\\" aparece %d %s em \\\"%s\\\" (%d %s, %d %s e %d %s).\", "
                 "\"texto\": \"%s\", \"letra\": \"%s\", \"ocorrencias\": %d, \"letras\": %d, "
                 "\"caracteres\": %d, \"palavras\": %d}",
                 esc_target, hits, hits == 1 ? "vez" : "vezes", esc_text, letters,
                 letters == 1 ? "letra" : "letras", chars, chars == 1 ? "caractere" : "caracteres", words,
                 words == 1 ? "palavra" : "palavras", esc_text, esc_target, hits, letters, chars, words);
        return 1;
    }
    snprintf(result, TOOL_RESULT_MAX,
             "{\"resumo\": \"\\\"%s\\\" tem %d %s, %d %s e %d %s.\", \"texto\": \"%s\", \"letras\": %d, "
             "\"caracteres\": %d, \"palavras\": %d}",
             esc_text, letters, letters == 1 ? "letra" : "letras", chars, chars == 1 ? "caractere" : "caracteres",
             words, words == 1 ? "palavra" : "palavras", esc_text, letters, chars, words);
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
    if (!calc_eval(expr, user_text, &value, err, sizeof(err))) {
        if (convert_relevant(user_text)) return convert_unit(c, user_text, result);
        return error_result(result, "não consegui calcular: %s", err);
    }
    calc_format(value, formatted, sizeof(formatted));
    json_escape(escaped, sizeof(escaped), expr);
    snprintf(result, TOOL_RESULT_MAX, "{\"resumo\": \"O resultado é %s.\", \"expressao\": \"%s\", \"resultado\": \"%s\"}",
             formatted, escaped, formatted);
    return 1;
}

int tools_relevant(const char *user_text) {
    return date_or_time_mentioned(user_text) || calc_relevant(user_text) || convert_relevant(user_text) ||
           count_relevant(user_text);
}

static int current_date_question(const char *user_text) {
    static const char *const now_words[] = {"hoje", "amanha", "ontem", "agora", "atual", "estamos"};
    char s[FOLD_MAX];
    text_fold_tail(user_text, s, sizeof(s));
    if (text_has_word(s, now_words, sizeof(now_words) / sizeof(now_words[0]))) return 1;
    return strstr(s, "que horas") != NULL || strstr(s, "qual a hora") != NULL || strstr(s, "qual e a hora") != NULL ||
           strstr(s, "que dia") != NULL || strstr(s, "qual a data") != NULL || strstr(s, "qual e a data") != NULL;
}

const char *tools_forced_choice(const char *user_text) {
    if (date_question(user_text)) return NULL;
    if (count_relevant(user_text)) return "contar";
    if (convert_relevant(user_text)) return "converter";
    if (calc_relevant(user_text)) return "calcular";
    return NULL;
}

const char *tools_date_choice(const char *user_text) {
    Date named, reference;
    reference.year = 2000;
    reference.month = 1;
    reference.day = 1;
    if (!date_question(user_text) || calc_relevant(user_text) || convert_relevant(user_text) ||
        count_relevant(user_text))
        return NULL;
    if (date_find(user_text, reference, &named, 1) > 0) return "dias_ate";
    if (current_date_question(user_text)) return "data_atual";
    return NULL;
}

size_t tools_call_prefix(const char *tool, char *out, size_t cap) {
    int w;
    if (!strcmp(tool, "calcular"))
        w = snprintf(out, cap, "\n{\"name\": \"calcular\", \"arguments\": {\"expressao\": \"");
    else if (!strcmp(tool, "converter"))
        w = snprintf(out, cap, "\n{\"name\": \"converter\", \"arguments\": {\"valor\": ");
    else if (!strcmp(tool, "contar"))
        w = snprintf(out, cap, "\n{\"name\": \"contar\", \"arguments\": {\"texto\": \"");
    else if (!strcmp(tool, "dias_ate"))
        w = snprintf(out, cap, "\n{\"name\": \"dias_ate\", \"arguments\": {}}");
    else if (!strcmp(tool, "data_atual"))
        w = snprintf(out, cap, "\n{\"name\": \"data_atual\", \"arguments\": {}}");
    else
        return 0;
    return w < 0 || (size_t)w >= cap ? 0 : (size_t)w;
}

int tool_call_grounded(const char *call, size_t n, const char *user_text) {
    Call c;
    Date named, reference;
    double val;
    const Unit *from, *to;
    reference.year = 2000; /* any valid day: only whether a date is named matters */
    reference.month = 1;
    reference.day = 1;
    if (!parse_call(call, n, &c)) return 0;
    if (!strcmp(c.name, "data_atual")) return 1;
    if (!strcmp(c.name, "calcular")) return expression_arg(&c) && calc_grounded(expression_arg(&c), user_text);
    if (!strcmp(c.name, "converter"))
        return convert_relevant(user_text) ||
               (!user_text[0] && model_convert_args(&c, &val, &from, &to) && from->kind == to->kind);
    if (!strcmp(c.name, "contar")) return count_grounded(&c, user_text);
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
    if (!strcmp(c.name, "converter")) return convert_unit(&c, user_text, result);
    if (!strcmp(c.name, "contar")) return count_text(&c, user_text, result);
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
    "{\"type\": \"function\", \"function\": {\"name\": \"converter\", \"description\": "
    "\"Converte um valor entre unidades de medida (comprimento, massa, temperatura, velocidade, volume ou dados). "
    "Use quando o usuário pedir para converter unidades.\", \"parameters\": {\"type\": \"object\", "
    "\"properties\": {\"valor\": {\"type\": \"number\", \"description\": \"Valor numérico a converter\"}, "
    "\"de\": {\"type\": \"string\", \"description\": \"Unidade de origem\"}, "
    "\"para\": {\"type\": \"string\", \"description\": \"Unidade de destino\"}}, "
    "\"required\": [\"valor\", \"de\", \"para\"]}}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"contar\", \"description\": "
    "\"Conta letras, ocorrências de uma letra, palavras e caracteres em uma palavra ou frase. Use quando o "
    "usuário perguntar quantas letras, palavras ou caracteres há em um texto.\", \"parameters\": "
    "{\"type\": \"object\", \"properties\": {\"texto\": {\"type\": \"string\", \"description\": "
    "\"A palavra ou frase exatamente como o usuário escreveu\"}, \"letra\": {\"type\": \"string\", "
    "\"description\": \"Letra específica a contar (opcional)\"}}, \"required\": [\"texto\"]}}}",
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
