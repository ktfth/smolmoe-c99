/* Unit tests for the chat tools: clock parsing, date arithmetic, strict call
 * parsing of untrusted model output and the system-prompt section. */
#include <stdio.h>
#include <string.h>

#include "calendar.h"
#include "tools.h"

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

static ToolClock clock_at(const char *s) {
    ToolClock c;
    if (!tool_clock_parse(s, &c)) {
        fprintf(stderr, "bad test clock %s\n", s);
        memset(&c, 0, sizeof(c));
        ++failures;
    }
    return c;
}

/* Runs `call` for a user message at `now` and compares name, success and the exact result. */
static void expect_grounded(const char *now, const char *user, const char *call, const char *name, int ok,
                            const char *result) {
    ToolClock c = clock_at(now);
    char got_name[TOOL_NAME_MAX], got[TOOL_RESULT_MAX];
    int ran = tool_run(call, strlen(call), user, &c, got_name, got);
    CHECK(ran == ok, "%s: ran=%d, expected %d (%s)", call, ran, ok, got);
    CHECK(!strcmp(got_name, name), "%s: name %s, expected %s", call, got_name, name);
    if (result) CHECK(!strcmp(got, result), "%s:\n  got      %s\n  expected %s", call, got, result);
    else CHECK(!strncmp(got, "{\"erro\": \"", 10), "%s: expected an erro result, got %s", call, got);
}

/* A user message without a date: the call's own arguments decide. */
static void expect_call(const char *now, const char *call, const char *name, int ok, const char *result) {
    expect_grounded(now, "", call, name, ok, result);
}

static void test_clock(void) {
    ToolClock c;
    CHECK(tool_clock_parse("2026-11-06T10:30", &c) && c.year == 2026 && c.month == 11 && c.day == 6 &&
              c.hour == 10 && c.minute == 30, "valid clock");
    CHECK(tool_clock_parse("2028-02-29T23:59", &c), "leap day");
    CHECK(!tool_clock_parse("2026-02-29T00:00", &c), "29 Feb in a common year");
    CHECK(!tool_clock_parse("2026-13-01T00:00", &c), "month 13");
    CHECK(!tool_clock_parse("2026-11-06T24:00", &c), "hour 24");
    CHECK(!tool_clock_parse("2026-11-06", &c), "date without time");
    CHECK(!tool_clock_parse("2026-11-06T10:30x", &c), "trailing text");
    CHECK(!tool_clock_parse("", &c), "empty");
}

static Date ymd(int y, int m, int d) {
    Date out;
    out.year = y;
    out.month = m;
    out.day = d;
    return out;
}

/* date_parse(text) at 2026-11-06 must give y-m-d, or fail when y == 0. */
static void expect_date(const char *text, int y, int m, int d) {
    Date got;
    int ok = date_parse(text, ymd(2026, 11, 6), &got);
    if (!y) {
        CHECK(!ok, "\"%s\" should not parse, got %d-%02d-%02d", text, got.year, got.month, got.day);
        return;
    }
    CHECK(ok && got.year == y && got.month == m && got.day == d, "\"%s\": got %s%d-%02d-%02d, expected %d-%02d-%02d",
          text, ok ? "" : "(failed) ", got.year, got.month, got.day, y, m, d);
}

static void test_calendar(void) {
    long z, first = date_days(ymd(1, 1, 1)), last = date_days(ymd(9999, 12, 31));
    CHECK(date_from_days(first - 1).year == 0 && !date_valid(date_from_days(first - 1)), "year 0 is out of range");
    for (z = first; z <= last; z += 997) {
        Date d = date_from_days(z);
        CHECK(date_valid(d) && date_days(d) == z, "day %ld round trip", z);
    }
    CHECK(!strcmp(date_weekday(ymd(1970, 1, 1)), "quinta-feira"), "1970-01-01 weekday");
    CHECK(!strcmp(date_weekday(ymd(2000, 2, 29)), "terça-feira"), "2000-02-29 weekday");
    CHECK(!date_valid(ymd(2100, 2, 29)) && date_valid(ymd(2000, 2, 29)), "century leap rule");
}

static void test_date_parse(void) {
    expect_date("25 de dezembro", 2026, 12, 25);
    expect_date("dia 25 de Dezembro", 2026, 12, 25);
    expect_date("25 dezembro", 2026, 12, 25);
    expect_date("25/12", 2026, 12, 25);
    expect_date("6 de novembro", 2026, 11, 6);   /* today counts */
    expect_date("5 de novembro", 2027, 11, 5);   /* already passed: next year */
    expect_date("3 de março", 2027, 3, 3);
    expect_date("3 de marco", 2027, 3, 3);
    expect_date("12 de Outubro", 2027, 10, 12);
    expect_date("1º de maio", 2027, 5, 1);
    expect_date("29 de fevereiro", 2028, 2, 29);
    expect_date("1/1/2027", 2027, 1, 1);
    expect_date("25/12/26", 2026, 12, 25);
    expect_date("2026-12-25", 2026, 12, 25);
    expect_date("1 de janeiro de 2026", 2026, 1, 1); /* an explicit year may be in the past */
    expect_date("Natal", 2026, 12, 25);
    expect_date("o natal", 2026, 12, 25);
    expect_date("Véspera de Natal", 2026, 12, 24);
    expect_date("ano novo", 2027, 1, 1);
    expect_date("Páscoa", 2027, 3, 28);
    expect_date("carnaval", 2027, 2, 9);
    expect_date("Sexta-feira Santa", 2027, 3, 26);
    expect_date("Corpus Christi", 2027, 5, 27);
    expect_date("Dia das Mães", 2027, 5, 9);
    expect_date("dia dos pais", 2027, 8, 8);
    expect_date("Dia das Crianças", 2027, 10, 12);
    expect_date("Proclamação da República", 2026, 11, 15);
    expect_date("Consciência Negra", 2026, 11, 20);
    expect_date("Finados", 2027, 11, 2);
    expect_date("Hoje é dia 3. Quanto falta para 25/12?", 2026, 12, 25);
    expect_date("tenho 30 anos e faço 3 de março", 2027, 3, 3);
    expect_date("aniversário da Natália", 0, 0, 0);  /* "natal" only as a whole word */
    expect_date("taxa de natalidade", 0, 0, 0);
    expect_date("25 de maiolica", 0, 0, 0);
    expect_date("32 de dezembro", 0, 0, 0);
    expect_date("30 de fevereiro", 0, 0, 0);
    expect_date("13/13", 0, 0, 0);
    expect_date("dezembro", 0, 0, 0);
    expect_date("amanhã de manhã", 0, 0, 0);
    expect_date("", 0, 0, 0);
}

/* The summary comes first: the model copies it more reliably there (measured). */
static void test_current_date(void) {
    expect_call("2026-11-06T10:30", "{\"name\": \"data_atual\", \"arguments\": {}}", "data_atual", 1,
                "{\"resumo\": \"Hoje é sexta-feira, 6 de novembro de 2026. Agora são 10:30.\", "
                "\"data\": \"6 de novembro de 2026\", \"dia_da_semana\": \"sexta-feira\", \"hora\": \"10:30\"}");
    expect_call("2026-11-06T07:05", "\n{\"name\": \"data_atual\"}\n", "data_atual", 1,
                "{\"resumo\": \"Hoje é sexta-feira, 6 de novembro de 2026. Agora são 07:05.\", "
                "\"data\": \"6 de novembro de 2026\", \"dia_da_semana\": \"sexta-feira\", \"hora\": \"07:05\"}");
}

#define NATAL_2026 "{\"resumo\": \"Faltam 49 dias para 25 de dezembro de 2026 (sexta-feira).\", " \
                   "\"data\": \"25 de dezembro de 2026\", \"dia_da_semana\": \"sexta-feira\", \"dias\": 49}"

static void test_days_until(void) {
    const char *natal = "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"Natal\"}}";
    expect_call("2026-11-06T10:30", natal, "dias_ate", 1, NATAL_2026);
    expect_call("2026-12-24T08:00", natal, "dias_ate", 1,
                "{\"resumo\": \"Falta 1 dia para 25 de dezembro de 2026 (sexta-feira).\", "
                "\"data\": \"25 de dezembro de 2026\", \"dia_da_semana\": \"sexta-feira\", \"dias\": 1}");
    expect_call("2026-12-25T08:00", natal, "dias_ate", 1,
                "{\"resumo\": \"É hoje: 25 de dezembro de 2026 (sexta-feira).\", "
                "\"data\": \"25 de dezembro de 2026\", \"dia_da_semana\": \"sexta-feira\", \"dias\": 0}");
    expect_call("2026-12-26T08:00", natal, "dias_ate", 1,
                "{\"resumo\": \"Faltam 364 dias para 25 de dezembro de 2027 (sábado).\", "
                "\"data\": \"25 de dezembro de 2027\", \"dia_da_semana\": \"sábado\", \"dias\": 364}");
    expect_call("2026-03-01T00:00", "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"29/02\"}}", "dias_ate", 1,
                "{\"resumo\": \"Faltam 730 dias para 29 de fevereiro de 2028 (terça-feira).\", "
                "\"data\": \"29 de fevereiro de 2028\", \"dia_da_semana\": \"terça-feira\", \"dias\": 730}");
    /* A past year in the model's argument (not the user's) is dropped. */
    expect_call("2026-11-06T10:30", "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"1 de janeiro de 2026\"}}",
                "dias_ate", 1,
                "{\"resumo\": \"Faltam 56 dias para 1 de janeiro de 2027 (sexta-feira).\", "
                "\"data\": \"1 de janeiro de 2027\", \"dia_da_semana\": \"sexta-feira\", \"dias\": 56}");
}

/* The user's own words win over the date the model wrote; a past year the model
 * invents is dropped. An explicit year the user wrote is kept. */
static void test_grounding(void) {
    const char *now = "2026-11-06T10:30";
    expect_grounded(now, "Quanto tempo falta para o natal?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"1 de dezembro de 2023\"}}", "dias_ate", 1,
                    NATAL_2026);
    expect_grounded(now, "Quantos dias até o dia das crianças, 12 de outubro?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": 12, \"dia\": 12}}", "dias_ate", 1,
                    "{\"resumo\": \"Faltam 340 dias para 12 de outubro de 2027 (terça-feira).\", "
                    "\"data\": \"12 de outubro de 2027\", \"dia_da_semana\": \"terça-feira\", \"dias\": 340}");
    expect_grounded(now, "e aquela data?", "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"25/12/2023\"}}",
                    "dias_ate", 1, NATAL_2026);
    /* Several dates in the message: the model's argument picks among them, else the last one asked. */
    expect_grounded(now, "Nasci em 3 de maio, quanto falta para 20/11?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"20/11\"}}", "dias_ate", 1,
                    "{\"resumo\": \"Faltam 14 dias para 20 de novembro de 2026 (sexta-feira).\", "
                    "\"data\": \"20 de novembro de 2026\", \"dia_da_semana\": \"sexta-feira\", \"dias\": 14}");
    expect_grounded(now, "Nasci em 3 de maio, quanto falta para 20/11?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"Natal\"}}", "dias_ate", 1,
                    "{\"resumo\": \"Faltam 14 dias para 20 de novembro de 2026 (sexta-feira).\", "
                    "\"data\": \"20 de novembro de 2026\", \"dia_da_semana\": \"sexta-feira\", \"dias\": 14}");
    expect_grounded(now, "Quanto falta para o 3 de maio, e não para o Natal?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"3 de maio\"}}", "dias_ate", 1,
                    "{\"resumo\": \"Faltam 178 dias para 3 de maio de 2027 (segunda-feira).\", "
                    "\"data\": \"3 de maio de 2027\", \"dia_da_semana\": \"segunda-feira\", \"dias\": 178}");
    expect_grounded(now, "Quantos dias faltam para a véspera de natal e para o natal?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"25/12\"}}", "dias_ate", 1, NATAL_2026);
    {
        /* A long message: the date at its end still counts. */
        char user[1400];
        memset(user, 'a', 1000);
        user[999] = ' ';
        strcpy(user + 1000, "e no fim: quanto falta para o natal?");
        CHECK(tool_call_grounded("{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"natal\"}}", 52, user),
              "date at the end of a 1 KB message");
        expect_grounded(now, user, "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"1/1\"}}", "dias_ate", 1,
                        NATAL_2026);
    }
    expect_grounded(now, "Quantos dias faltam para o aniversário da Natália?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"15/11\"}}", "dias_ate", 1,
                    "{\"resumo\": \"Faltam 9 dias para 15 de novembro de 2026 (domingo).\", "
                    "\"data\": \"15 de novembro de 2026\", \"dia_da_semana\": \"domingo\", \"dias\": 9}");
    expect_grounded(now, "Quantos dias faltavam para 1 de janeiro de 2026?",
                    "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"Natal\"}}", "dias_ate", 1,
                    "{\"resumo\": \"Essa data já passou: 1 de janeiro de 2026 (quinta-feira) foi há 309 dias.\", "
                    "\"data\": \"1 de janeiro de 2026\", \"dia_da_semana\": \"quinta-feira\", \"dias\": -309}");
}

/* Tools are offered only for messages about dates or time. */
static void test_relevance(void) {
    static const char *const yes[] = {"Que dia é hoje?", "Que horas são?", "Estamos em que mês?",
                                      "Quanto falta para o carnaval?", "Quantos dias faltam para 25/12?",
                                      "Em que dia da semana estamos?", "falta quanto pro meu aniversário?",
                                      "Qual é a data de hoje?", "QUE HORAS SAO"};
    static const char *const no[] = {"Qual é a capital da França?", "Quanto é 7 mais 5?",
                                     "Quem escreveu Dom Casmurro?", "Me dê uma dica curta para estudar melhor.",
                                     "Olá, tudo bem?", "Dias melhores virão é uma música?", ""};
    size_t i;
    for (i = 0; i < sizeof(yes) / sizeof(yes[0]); ++i) CHECK(tools_relevant(yes[i]), "relevant: %s", yes[i]);
    for (i = 0; i < sizeof(no) / sizeof(no[0]); ++i) CHECK(!tools_relevant(no[i]), "not relevant: %s", no[i]);
}

/* A call must fit the message: dias_ate needs a date or holiday in it. */
static void test_call_grounded(void) {
    const char *days = "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"1 de janeiro\"}}";
    const char *now = "{\"name\": \"data_atual\", \"arguments\": {}}";
    CHECK(tool_call_grounded(days, strlen(days), "Quanto falta para o Natal?"), "holiday in the message");
    CHECK(tool_call_grounded(days, strlen(days), "Quantos dias até 25/12?"), "date in the message");
    CHECK(!tool_call_grounded(days, strlen(days), "Quantas horas tem um dia?"), "no date: dias_ate discarded");
    CHECK(!tool_call_grounded(days, strlen(days), "O que é um feriado?"), "the word feriado is not a date");
    CHECK(tool_call_grounded(now, strlen(now), "Que horas são?"), "data_atual needs no date");
    CHECK(!tool_call_grounded("{\"name\": \"x\"}", 13, "Que horas são?"), "unknown tool");
    CHECK(!tool_call_grounded("{\"name\": ", 9, "Que horas são?"), "unreadable call");
}

/* Small models send month/day integers, quote numbers or encode the arguments as a JSON string. */
static void test_lenient_arguments(void) {
    expect_call("2026-11-06T10:30", "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": 12, \"dia\": 25}}",
                "dias_ate", 1, NATAL_2026);
    expect_call("2026-11-06T10:30", "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": \"12\", \"dia\": \"25\"}}",
                "dias_ate", 1, NATAL_2026);
    expect_call("2026-11-06T10:30", "{\"arguments\": \"{\\\"data\\\": \\\"25/12\\\"}\", \"name\": \"dias_ate\"}",
                "dias_ate", 1, NATAL_2026);
    expect_call("2026-11-06T10:30", "{\"name\": \"\\u0064ias_ate\", \"arguments\": {\"data\": \"25 de dezembro\"}}",
                "dias_ate", 1, NATAL_2026);
}

static void test_invalid_calls(void) {
    const char *now = "2026-11-06T10:30";
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"30 de fevereiro\"}}", "dias_ate", 0, NULL);
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"data\": \"\\\"quando\\\"\"}}", "dias_ate", 0,
                "{\"erro\": \"não entendi a data \\\"\\\"quando\\\"\\\"; use, por exemplo, 25 de dezembro ou 25/12\"}");
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {}}", "dias_ate", 0, NULL);
    /* The echoed text is cut at a character boundary: 59 'a' then "ç" would split at 60 bytes. */
    expect_call(now,
                "{\"name\": \"dias_ate\", \"arguments\": {\"data\": "
                "\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaçx\"}}",
                "dias_ate", 0,
                "{\"erro\": \"não entendi a data \\\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\\"; "
                "use, por exemplo, 25 de dezembro ou 25/12\"}");
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": 2, \"dia\": 30}}", "dias_ate", 0, NULL);
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": 13, \"dia\": 1}}", "dias_ate", 0, NULL);
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": 12}}", "dias_ate", 0, NULL);
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": 12.5, \"dia\": 25}}", "dias_ate", 0, NULL);
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": -1, \"dia\": 25}}", "dias_ate", 0, NULL);
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": {\"x\": 1}, \"dia\": 25}}", "?", 0, NULL);
    expect_call(now, "{\"name\": \"previsao\", \"arguments\": {}}", "previsao", 0,
                "{\"erro\": \"função desconhecida: previsao\"}");
    expect_call(now, "{\"name\": \"dias_ate\", \"arguments\": {\"mes\": 12", "?", 0, NULL);
    expect_call(now, "{\"name\": \"data_atual\"} extra", "?", 0, NULL);
    expect_call(now, "", "?", 0, NULL);
    expect_call(now, "[1, 2]", "?", 0, NULL);
    expect_call(now, "{\"name\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}", "?", 0, NULL);
    expect_call(now, "{\"name\": \"data_atual\", \"name\": \"dias_ate\"}", "?", 0, NULL);
    expect_call(now, "{\"name\": \"bad\\ud800\"}", "?", 0, NULL);
    expect_call(now, "{\"name\": 7}", "?", 0, NULL);
}

/* A name the model invents is echoed back: it must not break the JSON result. */
static void test_result_is_valid_json(void) {
    expect_call("2026-11-06T10:30", "{\"name\": \"a\\\"b\\\\c\"}", "a\"b\\c", 0,
                "{\"erro\": \"função desconhecida: a\\\"b\\\\c\"}");
}

static void test_system_prompt(void) {
    char out[4096];
    size_t n = tools_system_prompt("Seja breve.", out, sizeof(out));
    CHECK(n > 0 && n == strlen(out), "system prompt length");
    CHECK(!strncmp(out, "Seja breve.\n\n# Tools\n\n", 22), "starts with the system text and the Tools header");
    CHECK(strstr(out, "<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"data_atual\"") != NULL,
          "declares data_atual first");
    CHECK(strstr(out, "\"name\": \"dias_ate\"") != NULL, "declares dias_ate");
    CHECK(n > 12 && !strcmp(out + n - 12, "</tool_call>"), "ends with the call format");
    CHECK(tools_system_prompt("Seja breve.", out, 64) == 0, "too small a buffer is refused");
}

int main(void) {
    test_calendar();
    test_date_parse();
    test_clock();
    test_current_date();
    test_days_until();
    test_grounding();
    test_relevance();
    test_call_grounded();
    test_lenient_arguments();
    test_invalid_calls();
    test_result_is_valid_json();
    test_system_prompt();
    if (failures) {
        fprintf(stderr, "%d tool test(s) failed\n", failures);
        return 1;
    }
    puts("tool tests passed");
    return 0;
}
