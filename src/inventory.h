/* smolmoe-inventario: a read-only snapshot of what matters for defense on this
 * machine (listening ports, login users, admin groups, SUID binaries,
 * world-writable paths, persistence points, SSH authorized keys, security
 * configs) and the difference between two snapshots.
 *
 * Nothing here writes to the system or runs another program: everything is
 * read from files (/etc, /proc, homes). A report is plain text, one item per
 * line: "categoria\tchave\tdetalhe", sorted, so two runs on an unchanged
 * system give the same lines and `--comparar` shows only what changed.
 * Lines starting with '#' (host, date, unreadable paths) are not compared. */
#ifndef SMOLMOE_INVENTORY_H
#define SMOLMOE_INVENTORY_H

#include <stddef.h>
#include <stdio.h>

#define INV_FORMAT "smolmoe-inventario 1"
/* Longest item line, NUL included. inv_add and inv_finish cut items to fit
 * (marking " [cortado]" or ", ..."), so inventory_read accepts every report
 * inventory_write produces. */
#define INV_ITEM_MAX 8192

typedef struct InvList {
    char **items; /* "categoria\tchave\tdetalhe" */
    size_t count, cap;
} InvList;

void inv_init(InvList *l);
void inv_free(InvList *l);
/* Adds one item; tabs, line breaks and control bytes in a field become '?'.
 * The key is cut to 4096 bytes and the whole item to INV_ITEM_MAX - 1. */
void inv_add(InvList *l, const char *category, const char *key, const char *detail);
/* Sorts and merges items with the same category and key (details joined by ", "). */
void inv_finish(InvList *l);

/* Reads the system under root ("" or "/" for this machine; a mounted disk
 * otherwise). Items go to out (finished); notes (unreadable paths, missing
 * sources) go to notes as plain text lines. Linux layout; elsewhere the parts
 * that exist are read. */
void inventory_collect(const char *root, InvList *out, InvList *notes);

/* Writes a report: header comments, notes as comments, then the items. */
void inventory_write(FILE *f, const InvList *items, const InvList *notes, const char *host, const char *when,
                     const char *root);

/* Reads a report written by inventory_write. 1 on success; 0 with err filled. */
int inventory_read(const char *path, InvList *out, char *err, size_t cap);

/* Writes "+\tline" (new), "-\tline" (gone) and "~\tcategoria\tchave\tantes -> depois"
 * (changed detail) to f. Both lists must be finished. Returns the number of changes. */
size_t inventory_diff(const InvList *before, const InvList *after, FILE *f);

#endif
