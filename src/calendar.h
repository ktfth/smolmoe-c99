/* Gregorian calendar arithmetic, Brazilian holidays and dates written the way
 * people write them in Portuguese ("25 de dezembro", "25/12", "Natal").
 * Exact integer arithmetic: the chat model never computes a date itself. */
#ifndef SMOLMOE_CALENDAR_H
#define SMOLMOE_CALENDAR_H

#include <stddef.h>

typedef struct Date {
    int year, month, day;
} Date;

int date_valid(Date d);
long date_days(Date d);          /* days since 1970-01-01, proleptic Gregorian */
Date date_from_days(long days);
const char *date_weekday(Date d); /* "sexta-feira" */
void date_long(Date d, char *out, size_t cap); /* "25 de dezembro de 2026" */

/* Finds the dates `text` names, in order of mention and each once: the next
 * occurrence on or after `today`, or exactly that date when a year is given.
 * Understands "25 de dezembro [de 2026]", "dia 25 dezembro", "25/12[/2026]",
 * "2026-12-25" and holiday names (whole words). Returns how many were found. */
int date_find(const char *text, Date today, Date *out, int max);

/* The first date `text` names; 0 when it names none. */
int date_parse(const char *text, Date today, Date *out);

/* 1 when the text names a date or a holiday, or talks about time ("hoje",
 * "horas", "semana", "falta"...). A lexical cue, not understanding. */
int date_or_time_mentioned(const char *text);

/* Stricter: the text is about a date. A bare numeric date ("3/4") counts only
 * next to a date word ("dia", "até", "data"...), since "3/4 de 200" is a
 * fraction; time words, month names and holidays always count. */
int date_question(const char *text);

#endif
