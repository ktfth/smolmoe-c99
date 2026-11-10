/* test-papel: reads SMOLMOE_ENV the way the chat does and prints the role, the
 * notice and the value the notice shows, in the words of the adversary's
 * ambiente-casos.json ("producao nao_declarado", "dev nenhum",
 * "producao invalido staging"). tests/papel.test.ts runs it for every case.
 * With --hex BYTES it reads those bytes instead of the environment, for byte
 * sequences an environment cannot carry on every system (bad UTF-8). */
#include "papel.h"

#include <stdio.h>
#include <string.h>

static int hex_digit(char c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

int main(int argc, char **argv) {
    static const char *const avisos[] = {"nenhum", "nao_declarado", "invalido"};
    PapelLido p;
    if (argc == 3 && !strcmp(argv[1], "--hex")) {
        char bytes[1024];
        size_t i, n = strlen(argv[2]) / 2;
        if (n >= sizeof(bytes)) return 2;
        for (i = 0; i < n; ++i) {
            int a = hex_digit(argv[2][2 * i]), b = hex_digit(argv[2][2 * i + 1]);
            if (a < 0 || b < 0) return 2;
            bytes[i] = (char)(a * 16 + b);
        }
        bytes[n] = 0;
        papel_ler(bytes, &p);
    } else {
        papel_do_ambiente(&p);
    }
    printf("%s %s", p.papel == PAPEL_DEV ? "dev" : "producao", avisos[p.aviso]);
    if (p.aviso == PAPEL_AVISO_INVALIDA) printf(" %s", p.mostra);
    putchar('\n');
    return 0;
}
