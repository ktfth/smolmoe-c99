/* smolmoe-inventario: writes a read-only inventory of this machine, or compares two.
 * Exit codes: 0 ok / no change, 1 changes found (--comparar), 2 error. */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "inventory.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(_WIN32)
#  include <unistd.h>
#endif

static void usage(void) {
    fputs("uso: smolmoe-inventario [--raiz PASTA] [--saida ARQUIVO]\n"
          "     smolmoe-inventario --comparar ANTES DEPOIS\n"
          "\n"
          "  Sem opções, lê esta máquina (só leitura) e escreve o relatório no stdout:\n"
          "  portas em escuta, usuários com login, grupos administrativos, binários SUID,\n"
          "  caminhos graváveis por todos, pontos de inicialização, chaves SSH autorizadas\n"
          "  e configurações de segurança (com hash). Rode como root para cobertura completa.\n"
          "  --raiz PASTA     lê um sistema montado em PASTA (ex.: disco de outra máquina)\n"
          "  --saida ARQUIVO  grava o relatório em ARQUIVO (substituição atômica)\n"
          "  --comparar A B   mostra o que mudou de A para B: + novo, - removido, ~ alterado.\n"
          "                   Sai com 0 sem mudanças, 1 com mudanças, 2 em erro.\n",
          stderr);
    exit(2);
}

static void summary(const InvList *items, const InvList *notes) {
    size_t i = 0;
    fputs("smolmoe-inventario:", stderr);
    while (i < items->count) {
        const char *s = items->items[i];
        size_t len = strcspn(s, "\t"), n = 0;
        while (i < items->count && !strncmp(items->items[i], s, len) && items->items[i][len] == '\t') ++i, ++n;
        fprintf(stderr, " %.*s=%zu", (int)len, s, n);
    }
    fprintf(stderr, " (%zu notas)\n", notes->count);
}

static int compare(const char *a, const char *b) {
    InvList before, after;
    char err[512];
    size_t changes;
    inv_init(&before);
    inv_init(&after);
    if (!inventory_read(a, &before, err, sizeof(err)) || !inventory_read(b, &after, err, sizeof(err))) {
        fprintf(stderr, "smolmoe-inventario: %s\n", err);
        return 2;
    }
    changes = inventory_diff(&before, &after, stdout);
    fflush(stdout);
    fprintf(stderr, "smolmoe-inventario: %zu mudança(s)\n", changes);
    inv_free(&before);
    inv_free(&after);
    return changes ? 1 : 0;
}

int main(int argc, char **argv) {
    const char *root = "", *output = NULL;
    char host[256] = "?", when[32] = "?", tmp[4096];
    time_t now = time(NULL);
    struct tm *tm = localtime(&now);
    InvList items, notes;
    FILE *f = stdout;
    int a;

    for (a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "--comparar")) {
            if (a != 1 || argc != 4) usage();
            return compare(argv[2], argv[3]);
        }
        if (a + 1 >= argc) usage();
        if (!strcmp(argv[a], "--raiz")) root = argv[++a];
        else if (!strcmp(argv[a], "--saida")) output = argv[++a];
        else usage();
    }

#if defined(_WIN32)
    {
        const char *h = getenv("COMPUTERNAME");
        if (h) snprintf(host, sizeof(host), "%s", h);
    }
    (void)root;
    fputs("smolmoe-inventario: a coleta só existe para Linux; --comparar funciona aqui.\n", stderr);
    return 2;
#else
    if (gethostname(host, sizeof(host) - 1)) snprintf(host, sizeof(host), "?");
#endif
    if (tm) strftime(when, sizeof(when), "%Y-%m-%dT%H:%M", tm);

    inv_init(&items);
    inv_init(&notes);
    inventory_collect(root, &items, &notes);

    if (output) {
        if (snprintf(tmp, sizeof(tmp), "%s.tmp", output) >= (int)sizeof(tmp) || !(f = fopen(tmp, "wb"))) {
            fprintf(stderr, "smolmoe-inventario: %s: %s\n", output, strerror(errno));
            return 2;
        }
    }
    inventory_write(f, &items, &notes, host, when, root);
    if (output) {
        int bad = ferror(f);
        if (fclose(f)) bad = 1;
        if (bad || rename(tmp, output)) {
            fprintf(stderr, "smolmoe-inventario: %s: %s\n", output, strerror(errno));
            remove(tmp);
            return 2;
        }
    } else if (fflush(stdout) || ferror(stdout)) {
        return 2;
    }
    summary(&items, &notes);
    inv_free(&items);
    inv_free(&notes);
    return 0;
}
