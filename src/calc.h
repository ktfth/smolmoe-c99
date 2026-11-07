/* The "calcular" chat tool: arithmetic done by the runtime, not by the model.
 *
 * The model writes an expression; it is evaluated here in double precision
 * and the result is formatted the Brazilian way ("1.126,83"). A call is only
 * accepted when its numbers come from the user's message (calc_grounded). That
 * stops invented operands, not every misuse: small constants and percentages
 * are free, and the model can still drop one of the user's numbers or pick
 * the wrong formula (compound vs simple interest). */
#ifndef SMOLMOE_CALC_H
#define SMOLMOE_CALC_H

#include <stddef.h>

#define CALC_EXPR_MAX 200 /* bytes of an expression */

/* Evaluates + - * / x × ÷ − ^ ** %, parentheses and sqrt()/raiz(), plus the
 * words "mais", "menos", "vezes", "de" (15% de 230), "dividido por" and
 * "por cento", which small models write instead of symbols. Numbers may
 * use a decimal comma and Brazilian thousands dots ("1.234,56"); a lone
 * "1.500" is 1.5 unless `context` (the user's message, may be NULL) writes
 * that same number. Returns 1 with *out, or 0 with a Portuguese reason. */
int calc_eval(const char *expr, const char *context, double *out, char *err, size_t cap);

/* Brazilian format: thousands dots, decimal comma, 2 decimals from 1 up and 4
 * significant digits below 1, trailing zeros trimmed; %g beyond 1e15. */
void calc_format(double v, char *out, size_t cap);

/* Whether a message asks for a calculation: a number (digits or words) plus a
 * calculation word or an unambiguous operator. */
int calc_relevant(const char *text);

/* Whether every number of a readable expression comes from the message: as
 * written (digits or words), as a percentage of one (n/100, 1 ± n/100) when
 * the message says "%", or a small constant (1/2, 1, 2, 3, 4, 10, 12, 100,
 * 1000); and at least one is the message's own. */
int calc_grounded(const char *expr, const char *text);

#endif
