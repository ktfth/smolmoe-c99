/* The machine's role, declared only by the SMOLMOE_ENV variable (plan,
 * proposal 4a, revision 22): dev/development or prod/production, case and
 * surrounding blanks ignored. Absent, empty or anything else is the
 * conservative mode, which acts as production, with a notice of its own for
 * each case. The chat shows the role on every prompt and, in production, a red
 * banner at start. */
#ifndef SMOLMOE_PAPEL_H
#define SMOLMOE_PAPEL_H

#include <stdio.h>

typedef enum { PAPEL_DEV, PAPEL_PRODUCAO, PAPEL_CONSERVADOR } Papel;

typedef enum {
    PAPEL_AVISO_NENHUM,       /* a valid value */
    PAPEL_AVISO_NAO_DEFINIDA, /* absent or empty */
    PAPEL_AVISO_INVALIDA      /* another value ("staging", "producao") */
} PapelAviso;

#define PAPEL_MOSTRA_MAX 64 /* characters of the value a notice shows */

typedef struct PapelLido {
    Papel papel;
    PapelAviso aviso;
    /* The value as a notice shows it: ASCII blanks trimmed, control bytes as \xNN (an ESC never reaches the
     * terminal), cut at PAPEL_MOSTRA_MAX characters with "…". */
    char mostra[PAPEL_MOSTRA_MAX * 4 + 8];
} PapelLido;

/* Reads a SMOLMOE_ENV value; NULL means the variable is absent. */
void papel_ler(const char *valor, PapelLido *out);

/* Reads SMOLMOE_ENV from the environment. */
void papel_do_ambiente(PapelLido *out);

/* "[PRODUÇÃO]", "[conservador]" or "[dev]": the start of every chat prompt. */
const char *papel_marca(Papel p);

/* Production and the conservative mode ask before a listed command runs. */
int papel_age_como_producao(Papel p);

/* The start-up lines: the production banner (red when `cor`), or the
 * conservative notice with how to set the variable on this system; nothing in
 * dev. */
void papel_anunciar(const PapelLido *p, int cor, FILE *f);

#endif
