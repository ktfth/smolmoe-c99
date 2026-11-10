#include "papel.h"
#include "platform.h"

#include <string.h>

static int espaco(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'; }

static int igual_sem_caixa(const char *a, size_t n, const char *b) {
    size_t i;
    if (strlen(b) != n) return 0;
    for (i = 0; i < n; ++i) {
        char x = a[i] >= 'A' && a[i] <= 'Z' ? (char)(a[i] - 'A' + 'a') : a[i];
        if (x != b[i]) return 0;
    }
    return 1;
}

/* Length of the well-formed UTF-8 sequence at s (n bytes left), or 0 for a byte that does not start one. The
 * second byte's range rules out overlong forms (E0 80 9B is an ESC in disguise), surrogates (ED A0..BF) and code
 * points above U+10FFFF, as in the Unicode table of well-formed byte sequences. */
static size_t utf8_len(const unsigned char *s, size_t n) {
    unsigned char lo = 0x80, hi = 0xBF;
    size_t len, i;
    if (s[0] < 0x80) return 1;
    if (s[0] >= 0xC2 && s[0] <= 0xDF) len = 2;
    else if (s[0] >= 0xE0 && s[0] <= 0xEF) {
        len = 3;
        if (s[0] == 0xE0) lo = 0xA0;
        if (s[0] == 0xED) hi = 0x9F;
    } else if (s[0] >= 0xF0 && s[0] <= 0xF4) {
        len = 4;
        if (s[0] == 0xF0) lo = 0x90;
        if (s[0] == 0xF4) hi = 0x8F;
    } else return 0;
    if (len > n || s[1] < lo || s[1] > hi) return 0;
    for (i = 2; i < len; ++i)
        if ((s[i] & 0xC0) != 0x80) return 0;
    return len;
}

/* The value for a notice: C0 and C1 controls (an ESC or an 8-bit CSI) and bytes that are not UTF-8 as \xNN, at most
 * PAPEL_MOSTRA_MAX characters (a UTF-8 sequence is one character and is never split), then "…" when cut. */
static void mostrar(const char *s, size_t n, char *out, size_t cap) {
    const unsigned char *u = (const unsigned char *)s;
    size_t i = 0, o = 0, chars = 0;
    out[0] = 0;
    while (i < n) {
        size_t len = utf8_len(u + i, n - i);
        unsigned cp = len == 2 ? ((u[i] & 0x1Fu) << 6) | (u[i + 1] & 0x3Fu) : len == 1 ? u[i] : 0x100;
        if (chars == PAPEL_MOSTRA_MAX) {
            snprintf(out + o, cap - o, "…");
            return;
        }
        if (!len) {
            o += (size_t)snprintf(out + o, cap - o, "\\x%02x", u[i]);
            len = 1;
        } else if (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F)) {
            o += (size_t)snprintf(out + o, cap - o, "\\x%02x", cp);
        } else {
            memcpy(out + o, s + i, len);
            o += len;
            out[o] = 0;
        }
        i += len;
        ++chars;
    }
}

void papel_ler(const char *valor, PapelLido *out) {
    size_t n;
    memset(out, 0, sizeof(*out));
    out->papel = PAPEL_CONSERVADOR;
    out->aviso = PAPEL_AVISO_NAO_DEFINIDA;
    if (!valor) return;
    while (espaco(*valor)) ++valor; /* ASCII blanks only: a no-break space before "dev" makes it invalid */
    n = strlen(valor);
    while (n && espaco(valor[n - 1])) --n;
    if (!n) return;
    mostrar(valor, n, out->mostra, sizeof(out->mostra));
    out->aviso = PAPEL_AVISO_NENHUM;
    if (igual_sem_caixa(valor, n, "dev") || igual_sem_caixa(valor, n, "development")) out->papel = PAPEL_DEV;
    else if (igual_sem_caixa(valor, n, "prod") || igual_sem_caixa(valor, n, "production")) out->papel = PAPEL_PRODUCAO;
    else out->aviso = PAPEL_AVISO_INVALIDA;
}

void papel_do_ambiente(PapelLido *out) {
    char valor[2048];
    int lido = env_utf8("SMOLMOE_ENV", valor, sizeof(valor));
    papel_ler(lido ? valor : NULL, out);
    if (lido == 2) {
        /* Only the start was read, and "dev" plus 2045 blanks plus "x" must not become dev: a value cut is invalid.
         * What the notice shows always ends in "…", as the start of something longer. */
        const char *inicio = valor;
        size_t n;
        while (espaco(*inicio)) ++inicio;
        mostrar(inicio, strlen(inicio), out->mostra, sizeof(out->mostra) - 4);
        n = strlen(out->mostra);
        if (n < 3 || strcmp(out->mostra + n - 3, "…")) snprintf(out->mostra + n, sizeof(out->mostra) - n, "…");
        out->papel = PAPEL_CONSERVADOR;
        out->aviso = PAPEL_AVISO_INVALIDA;
    }
}

const char *papel_marca(Papel p) {
    return p == PAPEL_PRODUCAO ? "[PRODUÇÃO]" : p == PAPEL_DEV ? "[dev]" : "[conservador]";
}

int papel_age_como_producao(Papel p) { return p != PAPEL_DEV; }

void papel_anunciar(const PapelLido *p, int cor, FILE *f) {
#if defined(_WIN32)
    static const char *const como = "setx SMOLMOE_ENV dev (só vale num console novo)";
#else
    static const char *const como = "export SMOLMOE_ENV=dev no ~/.bashrc ou no ~/.zshrc";
#endif
    if (p->papel == PAPEL_PRODUCAO) {
        if (cor) fprintf(f, "\033[1;97;41m PRODUÇÃO \033[0m");
        else fputs("[PRODUÇÃO]", f);
        fprintf(f, " Esta máquina é de produção (SMOLMOE_ENV=%s): comandos que mudam o sistema pedem confirmação.\n",
                p->mostra);
    } else if (p->aviso == PAPEL_AVISO_NAO_DEFINIDA) {
        fprintf(f, "SMOLMOE_ENV não está definida: modo conservador, como em produção. Numa máquina de desenvolvimento: %s.\n",
                como);
    } else if (p->aviso == PAPEL_AVISO_INVALIDA) {
        fprintf(f, "SMOLMOE_ENV=%s não é válido; use dev, development, prod ou production. Modo conservador, como em "
                   "produção.\n", p->mostra);
    }
}
