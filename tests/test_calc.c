/* Unit tests for the calculator tool: evaluation of model-written expressions,
 * Brazilian number formatting, and the checks that tie a call to the user's
 * own message. */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "calc.h"

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

static void expect_value(const char *expr, const char *context, double want) {
    double got = 0;
    char err[128] = "";
    int ok = calc_eval(expr, context, &got, err, sizeof(err));
    CHECK(ok && fabs(got - want) <= 1e-9 * (fabs(want) + 1), "%s = %.12g (want %.12g) %s", expr, got, want, err);
}

static void expect_error(const char *expr, const char *needle) {
    double got = 0;
    char err[128] = "";
    CHECK(!calc_eval(expr, NULL, &got, err, sizeof(err)) && strstr(err, needle), "%s: expected error with '%s', got '%s'",
          expr, needle, err);
}

static void test_eval(void) {
    expect_value("2 + 3 * 4", NULL, 14);
    expect_value("(2 + 3) * 4", NULL, 20);
    expect_value("10 / 4", NULL, 2.5);
    expect_value("2 ^ 10", NULL, 1024);
    expect_value("2 ** 3", NULL, 8);
    expect_value("2 ^ 3 ^ 2", NULL, 512); /* right associative */
    expect_value("-3 + 5", NULL, 2);
    expect_value("-2 ^ 2", NULL, -4);     /* as written in math: -(2^2) */
    expect_value("230 * 15 / 100", NULL, 34.5);
    expect_value("15% * 230", NULL, 34.5);
    expect_value("230 * 15%", NULL, 34.5);
    expect_value("sqrt(144)", NULL, 12);
    expect_value("raiz(2)", NULL, 1.4142135623730951);
    expect_value("3,5 * 2", NULL, 7);              /* decimal comma */
    expect_value("1.234,56 + 0,44", NULL, 1235);   /* Brazilian thousands and decimals */
    expect_value("1.234.567 + 1", NULL, 1234568);
    expect_value("2 × 3 ÷ 4", NULL, 1.5);
    expect_value("8 x 7", NULL, 56);
    expect_value("10 − 4", NULL, 6);               /* U+2212 minus */
    expect_value("1000 * (1 + 0.01) ^ 12", NULL, 1126.8250301319697);
    expect_value("1.5 * 2", NULL, 3);              /* a lone dot is a decimal point... */
    expect_value("1.500 * 2", "Comprei por R$ 1.500 em 2 vezes", 3000); /* ...unless the user wrote it as thousands */
    expect_value("1.500 * 2", "a taxa é 1.500 por cento", 3000);
    expect_value("1.500 * 2", "quanto é 1.5 vezes 2", 3);
    /* Small models write the operation in Portuguese: "15% de 230". */
    expect_value("15% de 230", NULL, 34.5);
    expect_value("3/4 de 200", NULL, 150);
    expect_value("15 por cento de 230", NULL, 34.5);
    expect_value("7 vezes 8", NULL, 56);
    expect_value("10 mais 5 menos 3", NULL, 12);
    expect_value("100 dividido por 4", NULL, 25);
    expect_error("15 desconto", "inválid"); /* "de" must be a whole word */
    /* A percentage added to or taken from a value is a share of that value, as on a pocket calculator. */
    expect_value("100 - 10%", NULL, 90);
    expect_value("100 + 10%", NULL, 110);
    expect_value("200 - 5% * 2", NULL, 199.9); /* not a lone percentage: plain arithmetic */
    /* "0.125" is never a thousands group, whatever the message says. */
    expect_value("0.125 * 8", "uma fatia de 0.125 vezes 8", 1);
    expect_error("25.12.2024", "inválid"); /* thousands groups have three digits */
    expect_error("1 / (10 ^ 400)", "grande"); /* an overflow inside is not hidden by the result */
    {
        char many[160];
        int k, n = 0;
        for (k = 0; k < 33; ++k) n += sprintf(many + n, "1+");
        sprintf(many + n, "9");
        expect_error(many, "números demais");
    }
    expect_error("", "vazia");
    expect_error("2 +", "incompleta");
    expect_error("(2 + 3", "parêntese");
    expect_error("2 / 0", "zero");
    expect_error("sqrt(-1)", "negativo");
    expect_error("abc + 1", "inválid");
    expect_error("2 2", "inválid");
    expect_error("10 ^ 400", "grande");
    expect_error("1,234,567", "inválid");
    {
        char longer[400];
        memset(longer, '1', sizeof(longer) - 1);
        longer[sizeof(longer) - 1] = 0;
        expect_error(longer, "longa");
    }
}

static void expect_format(double v, const char *want) {
    char out[64];
    calc_format(v, out, sizeof(out));
    CHECK(!strcmp(out, want), "format %.17g -> '%s' (want '%s')", v, out, want);
}

static void test_format(void) {
    expect_format(34.5, "34,5");
    expect_format(1776, "1.776");
    expect_format(1000000, "1.000.000");
    expect_format(1126.8250301319697, "1.126,83");
    expect_format(-2.5, "-2,5");
    expect_format(2.0, "2");
    expect_format(33.333333333, "33,33");
    expect_format(0.3333333333, "0,3333");
    expect_format(0.000012346, "0,00001235"); /* 0.000012345 is 1.2344999...e-5 in binary */
    expect_format(-0.0, "0");
    expect_format(0, "0");
    expect_format(999.999, "1.000");
    expect_format(123456789012.5, "123.456.789.012,5");
    expect_format(2e20, "2e+20");
    expect_format(1e-14, "1e-14"); /* tiny but not zero */
}

static void test_relevant(void) {
    const char *yes[] = {"Quanto é 15% de 230?", "quanto é sete vezes oito?", "Divida 100 reais entre 3 pessoas",
                         "Qual a raiz quadrada de 144?", "2+2", "Um produto de 80 reais com 25% de desconto sai por quanto?",
                         "qnto é 12 x 12", "Se eu investir R$ 1.000 a 1% ao mês, quanto terei em 12 meses?",
                         "Qual a média de 7, 8 e 10?", "Quanto dá 3/4 de 200?"};
    const char *no[] = {"Em que ano o homem pisou na Lua?", "Me dê 3 dicas para dormir melhor", "Tenho 2 gatos",
                        "Que dia é hoje?", "Olá, tudo bem?", "Resuma em 2 frases a história do Brasil",
                        "Meu telefone é 11 98765-4321", "Seja breve uma vez mais",
                        "Quanto tempo leva para aprender uma língua?", "+55 11 99999-9999 me liga",
                        "Leia https://site.com/?a=1"};
    size_t i;
    for (i = 0; i < sizeof(yes) / sizeof(yes[0]); ++i) CHECK(calc_relevant(yes[i]), "should be relevant: %s", yes[i]);
    for (i = 0; i < sizeof(no) / sizeof(no[0]); ++i) CHECK(!calc_relevant(no[i]), "should not be relevant: %s", no[i]);
}

static void test_grounded(void) {
    CHECK(calc_grounded("230 * 15 / 100", "Quanto é 15% de 230?"), "percent of");
    CHECK(calc_grounded("230 * 0.15", "Quanto é 15% de 230?"), "percent as a fraction");
    CHECK(calc_grounded("80 * (1 - 0.25)", "Um produto de 80 reais com 25% de desconto sai por quanto?"), "discount");
    CHECK(calc_grounded("80 * 0.75", "Um produto de 80 reais com 25% de desconto sai por quanto?"), "discount factor");
    CHECK(calc_grounded("7 * 8", "quanto é sete vezes oito?"), "number words");
    CHECK(calc_grounded("25 + 125", "some vinte e cinco com cento e vinte e cinco"), "compound number words");
    CHECK(calc_grounded("2500 / 4", "dois mil e quinhentos divididos por quatro"), "thousands in words");
    CHECK(calc_grounded("100 / 3", "Divida 100 reais entre 3 pessoas"), "plain numbers");
    CHECK(calc_grounded("1000 * 1.01 ^ 12", "Se eu investir R$ 1.000 a 1% ao mês, quanto terei em 12 meses?"),
          "compound interest");
    CHECK(calc_grounded("1234.56 * 2", "Quanto é R$ 1.234,56 vezes 2?"), "Brazilian decimal in the text");
    CHECK(calc_grounded("200 / 2", "Quanto é a metade de 200?"), "half");
    CHECK(!calc_grounded("2 + 2", "Quanto é 15% de 230?"), "only constants: unrelated to the message");
    CHECK(!calc_grounded("230 * 17 / 100", "Quanto é 15% de 230?"), "invented number");
    CHECK(!calc_grounded("1 +", "Quanto é 1 + 1?"), "unreadable expression");
    CHECK(calc_grounded("2000 * 3", "quanto é 2 mil vezes 3?"), "digits followed by mil");
    CHECK(calc_grounded("1500000 / 3", "divida 1,5 milhão por 3"), "digits followed by milhão");
    CHECK(calc_grounded("2000000 / 4", "divida 2 milhões por 4"), "digits followed by milhões");
    CHECK(calc_grounded("2000000 / 4", "divida 2 milhoes por 4"), "digits followed by milhoes, no accent");
    CHECK(calc_grounded("(1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17) / 17",
                        "média de 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 e 17"), "seventeen numbers");
}

int main(void) {
    test_eval();
    test_format();
    test_relevant();
    test_grounded();
    if (failures) {
        fprintf(stderr, "%d calculator test(s) failed\n", failures);
        return 1;
    }
    puts("calculator tests passed");
    return 0;
}
