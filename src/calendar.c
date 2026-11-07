#include "calendar.h"
#include "textfold.h"

#include <stdio.h>
#include <string.h>

#define SEARCH_YEARS 8   /* 29 Feb can be 8 years away (2096 -> 2104) */
#define MAX_MENTIONS 16  /* dates looked at in one text */

static const char *const MONTHS[12] = {"janeiro", "fevereiro", "março", "abril", "maio", "junho",
                                       "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"};
static const char *const MONTH_KEYS[12] = {"janeiro", "fevereiro", "marco", "abril", "maio", "junho",
                                           "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"};
static const char *const WEEKDAYS[7] = {"domingo", "segunda-feira", "terça-feira", "quarta-feira",
                                        "quinta-feira", "sexta-feira", "sábado"};

/* ---- arithmetic (H. Hinnant's civil algorithms) --------------------------- */

static int is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }

static int month_days(int y, int m) {
    static const int days[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return m == 2 && is_leap(y) ? 29 : days[m - 1];
}

int date_valid(Date d) {
    return d.year >= 1 && d.year <= 9999 && d.month >= 1 && d.month <= 12 && d.day >= 1 &&
           d.day <= month_days(d.year, d.month);
}

long date_days(Date d) {
    long y = d.year - (d.month <= 2), era = (y >= 0 ? y : y - 399) / 400, yoe = y - era * 400;
    long doy = (153 * (d.month + (d.month > 2 ? -3 : 9)) + 2) / 5 + d.day - 1;
    return era * 146097 + yoe * 365 + yoe / 4 - yoe / 100 + doy - 719468;
}

Date date_from_days(long z) {
    long era, doe, yoe, doy, mp;
    Date d;
    z += 719468;
    era = (z >= 0 ? z : z - 146096) / 146097;
    doe = z - era * 146097;
    yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    mp = (5 * doy + 2) / 153;
    d.day = (int)(doy - (153 * mp + 2) / 5 + 1);
    d.month = (int)(mp < 10 ? mp + 3 : mp - 9);
    d.year = (int)(yoe + era * 400 + (d.month <= 2));
    return d;
}

static int weekday_index(Date d) { return (int)((date_days(d) % 7 + 11) % 7); } /* 0 = domingo */

const char *date_weekday(Date d) { return WEEKDAYS[weekday_index(d)]; }

void date_long(Date d, char *out, size_t cap) {
    snprintf(out, cap, "%d de %s de %d", d.day, MONTHS[d.month - 1], d.year);
}

/* ---- holidays ---------------------------------------------------------------- */

typedef enum { FIXED, EASTER, NTH_SUNDAY } HolidayKind;

typedef struct Holiday {
    const char *name; /* folded: lowercase, no accents */
    HolidayKind kind;
    int month, day;   /* EASTER: day = offset from Easter Sunday; NTH_SUNDAY: day = n */
} Holiday;

/* Whole words. A name contained in another comes after it, and a match inside
 * an earlier one is ignored ("vespera de natal" is not also "natal"). */
static const Holiday HOLIDAYS[] = {
    {"vespera de natal", FIXED, 12, 24},        {"natal", FIXED, 12, 25},
    {"reveillon", FIXED, 12, 31},               {"ano novo", FIXED, 1, 1},
    {"carnaval", EASTER, 0, -47},               {"sexta-feira santa", EASTER, 0, -2},
    {"sexta feira santa", EASTER, 0, -2},       {"pascoa", EASTER, 0, 0},
    {"corpus christi", EASTER, 0, 60},          {"tiradentes", FIXED, 4, 21},
    {"dia do trabalho", FIXED, 5, 1},           {"dia do trabalhador", FIXED, 5, 1},
    {"dia das maes", NTH_SUNDAY, 5, 2},
    {"dia dos namorados", FIXED, 6, 12},        {"dia dos pais", NTH_SUNDAY, 8, 2},
    {"independencia", FIXED, 9, 7},             {"dia das criancas", FIXED, 10, 12},
    {"nossa senhora aparecida", FIXED, 10, 12}, {"finados", FIXED, 11, 2},
    {"proclamacao da republica", FIXED, 11, 15}, {"consciencia negra", FIXED, 11, 20},
};

/* Easter Sunday (anonymous Gregorian algorithm). */
static Date easter(int y) {
    int a = y % 19, b = y / 100, c = y % 100, d = b / 4, e = b % 4, f = (b + 8) / 25, g = (b - f + 1) / 3;
    int h = (19 * a + b - d - g + 15) % 30, i = c / 4, k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7;
    int m = (a + 11 * h + 22 * l) / 451, n = h + l - 7 * m + 114;
    Date out;
    out.year = y;
    out.month = n / 31;
    out.day = n % 31 + 1;
    return out;
}

static Date holiday_in(const Holiday *h, int y) {
    Date d;
    d.year = y;
    d.month = h->month;
    d.day = h->day;
    if (h->kind == EASTER) return date_from_days(date_days(easter(y)) + h->day);
    if (h->kind == NTH_SUNDAY) {
        d.day = 1;
        d.day = 1 + (7 - weekday_index(d)) % 7 + 7 * (h->day - 1);
    }
    return d;
}

/* ---- parsing ------------------------------------------------------------------ */

static int read_number(const char **p, int max_digits, int *value) {
    int n = 0, v = 0;
    while (**p >= '0' && **p <= '9') {
        if (++n > max_digits) return 0;
        v = v * 10 + (*(*p)++ - '0');
    }
    *value = v;
    return n;
}

static int is_digit(char c) { return c >= '0' && c <= '9'; }

static int month_at(const char **p, int *month) {
    int i;
    for (i = 0; i < 12; ++i) {
        size_t n = strlen(MONTH_KEYS[i]);
        if (!strncmp(*p, MONTH_KEYS[i], n) && !text_is_letter((*p)[n])) {
            *month = i + 1;
            *p += n;
            return 1;
        }
    }
    return 0;
}

/* "25/12", "25/12/2026", "2026-12-25": y = 0 when no year is given. */
static int parse_separated(const char *p, int a, int na, int *y, int *m, int *d) {
    char sep = *p++;
    int b, c, nc;
    if (!read_number(&p, 2, &b)) return 0;
    *y = 0;
    if (*p != sep) {
        if (na > 2) return 0;
        *d = a;
        *m = b;
        return 1;
    }
    ++p;
    nc = read_number(&p, 4, &c);
    if (na == 4 && nc && nc <= 2) { *y = a; *m = b; *d = c; return 1; }
    if (na > 2 || (nc != 2 && nc != 4)) return 0;
    *y = nc == 2 ? 2000 + c : c;
    *m = b;
    *d = a;
    return 1;
}

/* A date starting at the digit run p: "25 de dezembro [de 2026]", "25 dezembro",
 * "25/12[/2026]", "2026-12-25". y = 0 when no year is given. */
static int numeric_at(const char *p, int *y, int *m, int *d) {
    int a, na, c;
    if (!(na = read_number(&p, 4, &a))) return 0;
    if (*p == '/' || *p == '-' || *p == '.') return parse_separated(p, a, na, y, m, d);
    if (na > 2) return 0;
    if (!strncmp(p, " de ", 4)) p += 4;
    else if (*p == ' ') ++p;
    if (!month_at(&p, m)) return 0;
    *d = a;
    *y = 0;
    if (!strncmp(p, " de ", 4)) {
        p += 4;
        if (read_number(&p, 4, &c) == 4) *y = c;
    }
    return 1;
}

static int on_or_after(Date candidate, long today, Date *out) {
    if (!date_valid(candidate) || date_days(candidate) < today) return 0;
    *out = candidate;
    return 1;
}

/* The exact date when a year is given, else the next occurrence from today. */
static int resolve(int y, int m, int d, Date today, Date *out) {
    Date x;
    x.month = m;
    x.day = d;
    if (m < 1 || m > 12) return 0;
    if (y) {
        x.year = y;
        if (!date_valid(x)) return 0;
        *out = x;
        return 1;
    }
    for (x.year = today.year; x.year <= today.year + SEARCH_YEARS; ++x.year)
        if (on_or_after(x, date_days(today), out)) return 1;
    return 0;
}

static int next_holiday(const Holiday *h, Date today, Date *out) {
    int y;
    for (y = today.year; y <= today.year + 1; ++y)
        if (on_or_after(holiday_in(h, y), date_days(today), out)) return 1;
    return 0;
}

/* A date named in the text, where it starts and how long its name is. */
typedef struct Mention {
    size_t at, len;
    Date date;
} Mention;

static int inside(const Mention *found, int n, size_t at) {
    int i;
    for (i = 0; i < n; ++i)
        if (at >= found[i].at && at < found[i].at + found[i].len) return 1;
    return 0;
}

static int add_holidays(const char *s, Date today, Mention *found, int n, int max) {
    size_t i;
    for (i = 0; i < sizeof(HOLIDAYS) / sizeof(HOLIDAYS[0]) && n < max; ++i) {
        const char *at = text_find_word(s, s, HOLIDAYS[i].name);
        /* Skip occurrences inside an earlier, longer name ("vespera de natal"). */
        while (at && inside(found, n, (size_t)(at - s))) at = text_find_word(s, at + 1, HOLIDAYS[i].name);
        if (!at || !next_holiday(&HOLIDAYS[i], today, &found[n].date)) continue;
        found[n].at = (size_t)(at - s);
        found[n].len = strlen(HOLIDAYS[i].name);
        ++n;
    }
    return n;
}

/* Every digit run that starts a date ("dia 3." or "30 anos" do not). */
static int add_numeric(const char *s, Date today, Mention *found, int n, int max) {
    const char *p = s;
    while (*p && n < max) {
        int y, m, d;
        if (!is_digit(*p)) {
            ++p;
            continue;
        }
        if (numeric_at(p, &y, &m, &d) && resolve(y, m, d, today, &found[n].date)) {
            found[n].at = (size_t)(p - s);
            found[n].len = 1;
            ++n;
        }
        while (is_digit(*p)) ++p;
    }
    return n;
}

/* Dates in order of mention, each date once. */
static int mentions(const char *text, Date today, Date *out, int max) {
    char s[FOLD_MAX];
    Mention found[MAX_MENTIONS];
    int n, i, j, count = 0;
    text_fold_tail(text, s, sizeof(s));
    n = add_holidays(s, today, found, 0, MAX_MENTIONS);
    n = add_numeric(s, today, found, n, MAX_MENTIONS);
    for (i = 1; i < n; ++i) /* insertion sort by position */
        for (j = i; j > 0 && found[j].at < found[j - 1].at; --j) {
            Mention t = found[j];
            found[j] = found[j - 1];
            found[j - 1] = t;
        }
    for (i = 0; i < n && count < max; ++i) {
        for (j = 0; j < count && date_days(out[j]) != date_days(found[i].date); ++j) {}
        if (j == count) out[count++] = found[i].date;
    }
    return count;
}

int date_find(const char *text, Date today, Date *out, int max) { return mentions(text, today, out, max); }

int date_parse(const char *text, Date today, Date *out) { return mentions(text, today, out, 1) == 1; }

/* Whole words, folded. "dia"/"dias" alone are too common ("dias melhores"). */
static const char *const TIME_WORDS[] = {"hoje", "amanha", "ontem", "agora", "hora", "horas", "horario",
                                         "data", "datas", "semana", "mes", "falta", "faltam", "faltando",
                                         "faltar", "aniversario", "calendario", "feriado", "feriados"};

static int has_time_word(const char *s) { return text_has_word(s, TIME_WORDS, sizeof(TIME_WORDS) / sizeof(TIME_WORDS[0])); }

int date_or_time_mentioned(const char *text) {
    char s[FOLD_MAX];
    Date any, reference;
    reference.year = 2000; /* any valid day: only whether a date is named matters */
    reference.month = 1;
    reference.day = 1;
    text_fold_tail(text, s, sizeof(s));
    return has_time_word(s) || date_parse(text, reference, &any);
}

/* Words that make a bare "3/4" a date ("dias até 3/4") rather than a fraction ("3/4 de 200"). */
static const char *const DATE_CONTEXT[] = {"dia", "dias", "ate", "data", "prazo", "vencimento", "quando"};

static int names_month_or_holiday(const char *s) {
    size_t i;
    for (i = 0; i < 12; ++i)
        if (text_find_word(s, s, MONTH_KEYS[i])) return 1;
    for (i = 0; i < sizeof(HOLIDAYS) / sizeof(HOLIDAYS[0]); ++i)
        if (text_find_word(s, s, HOLIDAYS[i].name)) return 1;
    return 0;
}

int date_question(const char *text) {
    char s[FOLD_MAX];
    Date any, reference;
    reference.year = 2000;
    reference.month = 1;
    reference.day = 1;
    text_fold_tail(text, s, sizeof(s));
    if (has_time_word(s) || names_month_or_holiday(s)) return 1;
    return date_parse(text, reference, &any) &&
           text_has_word(s, DATE_CONTEXT, sizeof(DATE_CONTEXT) / sizeof(DATE_CONTEXT[0]));
}
