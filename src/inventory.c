#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#  define _DARWIN_C_SOURCE 1
#  define _FILE_OFFSET_BITS 64
#endif

#include "inventory.h"
#include "sha256.h"

#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#  include <dirent.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#define INV_PATH 4096
#define INV_LINE 8192 /* longest line read from a system file; longer lines are skipped whole */
#define HASH_BYTES 8 /* 16 hex digits: enough to notice a change */

/* ---- lists ---------------------------------------------------------------- */

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fputs("smolmoe-inventario: sem memória\n", stderr);
        exit(2);
    }
    return q;
}

void inv_init(InvList *l) { memset(l, 0, sizeof(*l)); }

void inv_free(InvList *l) {
    size_t i;
    for (i = 0; i < l->count; ++i) free(l->items[i]);
    free(l->items);
    inv_init(l);
}

static void push(InvList *l, char *s) {
    if (l->count == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 64;
        l->items = (char **)xrealloc(l->items, l->cap * sizeof(*l->items));
    }
    l->items[l->count++] = s;
}

/* Copies s into out at *n, replacing control bytes (tab and newline included). */
static void put_clean(char *out, size_t *n, const char *s) {
    for (; *s; ++s) out[(*n)++] = (unsigned char)*s < 32 || *s == 127 ? '?' : *s;
}

/* Cuts s to at most max bytes without splitting a UTF-8 character. */
static void cut_utf8(char *s, size_t max) {
    size_t n = strlen(s);
    if (n <= max) return;
    while (max && ((unsigned char)s[max] & 0xC0) == 0x80) --max;
    s[max] = 0;
}

#define INV_KEY_MAX 4096 /* a path; longer keys are cut */
#define INV_CUT " [cortado]"
#define INV_MORE ", ..."

void inv_add(InvList *l, const char *category, const char *key, const char *detail) {
    size_t n = 0, len = strlen(category) + strlen(key) + strlen(detail) + 3, keep;
    char *s = (char *)xrealloc(NULL, len), *k;
    put_clean(s, &n, category);
    s[n++] = '\t';
    k = s + n;
    put_clean(s, &n, key);
    s[n] = 0;
    if (strlen(k) > INV_KEY_MAX) {
        cut_utf8(k, INV_KEY_MAX);
        n = strlen(s);
    }
    s[n++] = '\t';
    keep = n;
    put_clean(s, &n, detail);
    s[n] = 0;
    /* Every item must fit in a report line that inventory_read accepts. */
    if (n > INV_ITEM_MAX - 1) {
        size_t room = INV_ITEM_MAX - 1 - strlen(INV_CUT);
        cut_utf8(s, room > keep ? room : keep); /* cut before shrinking the buffer */
        s = (char *)xrealloc(s, INV_ITEM_MAX);
        strcat(s, INV_CUT);
    }
    push(l, s);
}

static void note(InvList *notes, const char *fmt, ...) {
    char buf[INV_PATH + 128];
    size_t n = 0;
    char *s;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    s = (char *)xrealloc(NULL, strlen(buf) + 1);
    put_clean(s, &n, buf);
    s[n] = 0;
    push(notes, s);
}

static int ends_with_str(const char *s, const char *tail) {
    size_t a = strlen(s), b = strlen(tail);
    return a >= b && !strcmp(s + a - b, tail);
}

/* Length of "categoria\tchave" (up to the second tab). */
static size_t key_len(const char *s) {
    const char *t = strchr(s, '\t');
    if (t) t = strchr(t + 1, '\t');
    return t ? (size_t)(t - s) : strlen(s);
}

static int key_cmp(const char *a, const char *b) {
    size_t la = key_len(a), lb = key_len(b);
    int r = memcmp(a, b, la < lb ? la : lb);
    return r ? r : (la > lb) - (la < lb);
}

static int line_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

void inv_finish(InvList *l) {
    size_t i, kept = 0;
    if (l->count > 1) qsort(l->items, l->count, sizeof(*l->items), line_cmp);
    /* Exact duplicates first, so a merged line never repeats a detail. */
    for (i = 0; i < l->count; ++i) {
        if (kept && !strcmp(l->items[kept - 1], l->items[i])) free(l->items[i]);
        else l->items[kept++] = l->items[i];
    }
    l->count = kept;
    kept = 0;
    for (i = 0; i < l->count; ++i) {
        char *s = l->items[i];
        if (kept && !key_cmp(l->items[kept - 1], s)) {
            char *prev = l->items[kept - 1];
            const char *detail = s + key_len(s) + 1;
            size_t pl = strlen(prev), dl = strlen(detail);
            if (!ends_with_str(prev, INV_MORE)) {
                if (pl + 2 + dl <= INV_ITEM_MAX - 1) {
                    prev = (char *)xrealloc(prev, pl + dl + 3);
                    memcpy(prev + pl, ", ", 2);
                    strcpy(prev + pl + 2, detail);
                } else {
                    /* Too many details for one line: say there are more instead of growing. */
                    cut_utf8(prev, INV_ITEM_MAX - 1 - strlen(INV_MORE));
                    prev = (char *)xrealloc(prev, INV_ITEM_MAX);
                    strcat(prev, INV_MORE);
                }
                l->items[kept - 1] = prev;
            }
            free(s);
        } else {
            l->items[kept++] = s;
        }
    }
    l->count = kept;
}

/* One line of a system file into buf, without the line ending. Returns 1, 0 at
 * the end, or -1 for a line that is longer than cap-1 bytes or holds a NUL:
 * that line is consumed whole and must be skipped, so its tail is never read
 * as a line of its own (a long gecos field must not become a fake account). */
static int read_line(FILE *f, char *buf, size_t cap) {
    size_t n = 0;
    int ch, bad = 0;
    while ((ch = getc(f)) != EOF && ch != '\n') {
        if (ch == 0 || n + 1 >= cap) bad = 1;
        else buf[n++] = (char)ch;
    }
    buf[n] = 0;
    if (ch == EOF && n == 0 && !bad) return 0;
    if (n && buf[n - 1] == '\r') buf[--n] = 0;
    return bad ? -1 : 1;
}

/* ---- report files --------------------------------------------------------- */

void inventory_write(FILE *f, const InvList *items, const InvList *notes, const char *host, const char *when,
                     const char *root) {
    size_t i;
    fprintf(f, "# %s\n# host: %s\n# data: %s\n# raiz: %s\n# itens: %zu\n", INV_FORMAT, host, when,
            root && *root ? root : "/", items->count);
    for (i = 0; notes && i < notes->count; ++i) fprintf(f, "# nota: %s\n", notes->items[i]);
    for (i = 0; i < items->count; ++i) fprintf(f, "%s\n", items->items[i]);
}

int inventory_read(const char *path, InvList *out, char *err, size_t cap) {
    char line[INV_ITEM_MAX + 1];
    unsigned long number = 0;
    int header = 0, r;
    FILE *f = fopen(path, "rb");
    if (!f) {
        snprintf(err, cap, "%s: %s", path, strerror(errno));
        return 0;
    }
    while ((r = read_line(f, line, sizeof(line))) != 0) {
        size_t n = strlen(line), i;
        const char *t1, *t2;
        char *s;
        ++number;
        if (r < 0) {
            snprintf(err, cap, "%s:%lu: linha longa demais ou com byte nulo", path, number);
            fclose(f);
            return 0;
        }
        if (!header) {
            if (strcmp(line, "# " INV_FORMAT)) {
                snprintf(err, cap, "%s: não é um relatório do smolmoe-inventario", path);
                fclose(f);
                return 0;
            }
            header = 1;
            continue;
        }
        if (!n || line[0] == '#') continue;
        /* inventory_write never writes control bytes besides the two tabs: anything else is damage. */
        for (i = 0; i < n; ++i)
            if (((unsigned char)line[i] < 32 && line[i] != '\t') || line[i] == 127) {
                snprintf(err, cap, "%s:%lu: byte de controle", path, number);
                fclose(f);
                return 0;
            }
        t1 = strchr(line, '\t');
        t2 = t1 ? strchr(t1 + 1, '\t') : NULL;
        if (!t2 || strchr(t2 + 1, '\t')) {
            snprintf(err, cap, "%s:%lu: esperado 'categoria<TAB>chave<TAB>detalhe'", path, number);
            fclose(f);
            return 0;
        }
        s = (char *)xrealloc(NULL, n + 1);
        memcpy(s, line, n + 1);
        push(out, s);
    }
    fclose(f);
    if (!header) {
        snprintf(err, cap, "%s: arquivo vazio", path);
        return 0;
    }
    inv_finish(out);
    return 1;
}

size_t inventory_diff(const InvList *before, const InvList *after, FILE *f) {
    size_t i = 0, j = 0, changes = 0;
    while (i < before->count || j < after->count) {
        int c = i == before->count ? 1 : j == after->count ? -1 : key_cmp(before->items[i], after->items[j]);
        if (c < 0) {
            fprintf(f, "-\t%s\n", before->items[i++]);
            ++changes;
        } else if (c > 0) {
            fprintf(f, "+\t%s\n", after->items[j++]);
            ++changes;
        } else {
            const char *a = before->items[i], *b = after->items[j];
            if (strcmp(a, b)) {
                size_t k = key_len(a);
                fprintf(f, "~\t%.*s\t%s -> %s\n", (int)k, a, a + k + 1, b + k + 1);
                ++changes;
            }
            ++i, ++j;
        }
    }
    return changes;
}

/* ---- collection (POSIX) --------------------------------------------------- */

#if defined(_WIN32)

void inventory_collect(const char *root, InvList *out, InvList *notes) {
    (void)root;
    (void)out;
    note(notes, "coleta disponível só em Linux (e parcial em outros sistemas POSIX)");
}

#else

typedef struct User {
    char name[64], home[512], shell[256];
    unsigned long uid;
    int login;
} User;

typedef struct Ctx {
    char root[INV_PATH];
    InvList *out, *notes;
    User *users;
    size_t nusers;
} Ctx;

static int full_path(const Ctx *c, const char *shown, char *out) {
    int n = snprintf(out, INV_PATH, "%s%s", c->root, shown);
    return n > 0 && n < INV_PATH;
}

static FILE *open_shown(Ctx *c, const char *shown) {
    char p[INV_PATH];
    FILE *f;
    if (!full_path(c, shown, p)) return NULL;
    f = fopen(p, "rb");
    if (!f && errno == EACCES) note(c->notes, "sem acesso: %s", shown);
    return f;
}

/* 16 hex digits of the file's SHA-256 into hex; 0 when it cannot be read. */
static int hash_file(const char *path, char hex[2 * HASH_BYTES + 1]) {
    static const char digits[] = "0123456789abcdef";
    unsigned char buf[65536], d[32];
    Sha256 s;
    size_t n;
    int i;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    sha256_init(&s);
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) sha256_update(&s, buf, n);
    if (ferror(f)) {
        fclose(f);
        return 0;
    }
    fclose(f);
    sha256_final(&s, d);
    for (i = 0; i < HASH_BYTES; ++i) hex[2 * i] = digits[d[i] >> 4], hex[2 * i + 1] = digits[d[i] & 15];
    hex[2 * HASH_BYTES] = 0;
    return 1;
}

/* Splits a colon-separated line in place; returns the number of fields. */
static int split_colons(char *s, char **fields, int max) {
    int n = 0;
    fields[n++] = s;
    while (n < max && (s = strchr(s, ':')) != NULL) {
        *s++ = 0;
        fields[n++] = s;
    }
    return n;
}

/* A uid: 1 to 10 decimal digits and nothing else. strtoul alone would read "" or "x" as 0 (root). */
static int parse_uid(const char *s, unsigned long *out) {
    size_t n = strlen(s), i;
    if (!n || n > 10) return 0;
    for (i = 0; i < n; ++i)
        if (s[i] < '0' || s[i] > '9') return 0;
    *out = strtoul(s, NULL, 10);
    return 1;
}

static int ends_with(const char *s, const char *tail) {
    size_t a = strlen(s), b = strlen(tail);
    return a >= b && !strcmp(s + a - b, tail);
}

static int login_shell(const char *shell) {
    static const char *const none[] = {"nologin", "/false", "/true", "/sync", "/shutdown", "/halt"};
    size_t i;
    for (i = 0; i < sizeof(none) / sizeof(none[0]); ++i)
        if (ends_with(shell, none[i])) return 0;
    return 1;
}

static void collect_users(Ctx *c) {
    char line[INV_LINE], detail[512], key[INV_LINE + 32];
    int r;
    FILE *f = open_shown(c, "/etc/passwd");
    if (!f) {
        note(c->notes, "usuários: /etc/passwd ausente");
        return;
    }
    while ((r = read_line(f, line, sizeof(line))) != 0) {
        char *fl[7];
        User u;
        if (r < 0) { note(c->notes, "linha longa ou inválida ignorada: /etc/passwd"); continue; }
        if (!line[0] || line[0] == '#' || split_colons(line, fl, 7) != 7) continue;
        memset(&u, 0, sizeof(u));
        /* Malformed or oversized fields: skip the account rather than guess or cut it. */
        if (!fl[0][0] || strlen(fl[0]) >= sizeof(u.name) || strlen(fl[5]) >= sizeof(u.home) ||
            strlen(fl[6]) >= sizeof(u.shell) || !parse_uid(fl[2], &u.uid)) {
            note(c->notes, "conta malformada ignorada em /etc/passwd");
            continue;
        }
        snprintf(u.name, sizeof(u.name), "%s", fl[0]);
        snprintf(u.home, sizeof(u.home), "%s", fl[5]);
        snprintf(u.shell, sizeof(u.shell), "%s", fl[6][0] ? fl[6] : "/bin/sh");
        u.login = login_shell(u.shell);
        if (u.login || u.uid == 0) {
            snprintf(detail, sizeof(detail), "uid=%lu shell=%s", u.uid, u.shell);
            inv_add(c->out, "usuario", u.name, detail);
        }
        if (u.uid == 0 && strcmp(u.name, "root")) {
            snprintf(key, sizeof(key), "uid0:%s", u.name);
            inv_add(c->out, "alerta", key, "outro usuário com uid 0");
        }
        c->users = (User *)xrealloc(c->users, (c->nusers + 1) * sizeof(*c->users));
        c->users[c->nusers++] = u;
    }
    fclose(f);

    f = open_shown(c, "/etc/shadow");
    if (!f) return;
    while ((r = read_line(f, line, sizeof(line))) != 0) {
        char *fl[3];
        if (r < 0) continue;
        if (split_colons(line, fl, 3) < 2 || fl[1][0]) continue;
        snprintf(key, sizeof(key), "senha_vazia:%s", fl[0]);
        inv_add(c->out, "alerta", key, "entra sem senha");
    }
    fclose(f);
}

/* Members of groups that amount to root (or read every disk). */
static void collect_admins(Ctx *c) {
    static const char *const groups[] = {"root", "sudo", "wheel", "admin", "docker", "lxd", "libvirt", "disk"};
    char line[INV_LINE], key[INV_LINE + 32];
    int r;
    FILE *f = open_shown(c, "/etc/group");
    if (!f) return;
    while ((r = read_line(f, line, sizeof(line))) != 0) {
        char *fl[4], *m, *next;
        size_t g;
        if (r < 0) { note(c->notes, "linha longa ou inválida ignorada: /etc/group"); continue; }
        if (split_colons(line, fl, 4) != 4) continue;
        for (g = 0; g < sizeof(groups) / sizeof(groups[0]); ++g)
            if (!strcmp(fl[0], groups[g])) break;
        if (g == sizeof(groups) / sizeof(groups[0])) continue;
        for (m = fl[3]; m && *m; m = next) {
            next = strchr(m, ',');
            if (next) *next++ = 0;
            if (!*m) continue;
            snprintf(key, sizeof(key), "%s:%s", fl[0], m);
            inv_add(c->out, "admin", key, "membro");
        }
    }
    fclose(f);
}

/* ---- ports from /proc/net ---- */

typedef struct Sock {
    unsigned long inode;
    char comm[32];
    unsigned long pid;
} Sock;

typedef struct SockMap {
    Sock *v;
    size_t n;
} SockMap;

static int is_digits(const char *s) {
    if (!*s) return 0;
    for (; *s; ++s)
        if (*s < '0' || *s > '9') return 0;
    return 1;
}

static void read_comm(Ctx *c, const char *pid, char *out, size_t cap) {
    char p[INV_PATH], *nl;
    FILE *f;
    snprintf(out, cap, "?");
    if (snprintf(p, sizeof(p), "%s/proc/%s/comm", c->root, pid) >= (int)sizeof(p)) return;
    f = fopen(p, "rb");
    if (!f) return;
    if (fgets(out, (int)cap, f)) {
        nl = strchr(out, '\n');
        if (nl) *nl = 0;
        if (!*out) snprintf(out, cap, "?");
    }
    fclose(f);
}

/* Socket inode -> process, from /proc/<pid>/fd. Other users' processes need root. */
static SockMap socket_owners(Ctx *c) {
    SockMap m = {NULL, 0};
    char p[INV_PATH], link[128];
    DIR *proc, *fds;
    struct dirent *e, *fe;
    unsigned long denied = 0;
    if (snprintf(p, sizeof(p), "%s/proc", c->root) >= (int)sizeof(p) || !(proc = opendir(p))) return m;
    while ((e = readdir(proc)) != NULL) {
        char comm[32] = "";
        if (!is_digits(e->d_name)) continue;
        if (snprintf(p, sizeof(p), "%s/proc/%s/fd", c->root, e->d_name) >= (int)sizeof(p)) continue;
        if (!(fds = opendir(p))) {
            if (errno == EACCES) ++denied;
            continue;
        }
        while ((fe = readdir(fds)) != NULL) {
            char lp[INV_PATH];
            ssize_t n;
            unsigned long inode;
            if (!is_digits(fe->d_name)) continue;
            if (snprintf(lp, sizeof(lp), "%s/%s", p, fe->d_name) >= (int)sizeof(lp)) continue;
            n = readlink(lp, link, sizeof(link) - 1);
            if (n <= 0) continue;
            link[n] = 0;
            if (sscanf(link, "socket:[%lu]", &inode) != 1) continue;
            if (!comm[0]) read_comm(c, e->d_name, comm, sizeof(comm));
            m.v = (Sock *)xrealloc(m.v, (m.n + 1) * sizeof(*m.v));
            m.v[m.n].inode = inode;
            m.v[m.n].pid = strtoul(e->d_name, NULL, 10);
            snprintf(m.v[m.n].comm, sizeof(m.v[m.n].comm), "%s", comm);
            ++m.n;
        }
        closedir(fds);
    }
    closedir(proc);
    if (denied) note(c->notes, "portas: dono de socket desconhecido para %lu processos (rode como root)", denied);
    return m;
}

static const char *owner_of(const SockMap *m, unsigned long inode) {
    size_t i;
    for (i = 0; i < m->n; ++i)
        if (m->v[i].inode == inode) return m->v[i].comm;
    return "?";
}

/* /proc/net address (hex, 32-bit words in host order, i.e. little-endian on Linux targets) to text. */
static int format_addr(const char *hex, int v6, char *out, size_t cap, int *loopback) {
    unsigned char b[16];
    int words = v6 ? 4 : 1, i, allzero = 1;
    if (strlen(hex) != (size_t)(8 * words)) return 0;
    for (i = 0; i < words; ++i) {
        char w[9];
        unsigned long x;
        memcpy(w, hex + 8 * i, 8);
        w[8] = 0;
        x = strtoul(w, NULL, 16);
        b[4 * i] = (unsigned char)x, b[4 * i + 1] = (unsigned char)(x >> 8);
        b[4 * i + 2] = (unsigned char)(x >> 16), b[4 * i + 3] = (unsigned char)(x >> 24);
    }
    if (!v6) {
        *loopback = b[0] == 127;
        snprintf(out, cap, "%u.%u.%u.%u", b[0], b[1], b[2], b[3]);
        return 1;
    }
    for (i = 0; i < 16; ++i) allzero &= b[i] == 0;
    *loopback = 0;
    if (allzero) {
        snprintf(out, cap, "[::]");
        return 1;
    }
    for (i = 0; i < 15 && !b[i]; ++i) {}
    if (i == 15 && b[15] == 1) {
        *loopback = 1;
        snprintf(out, cap, "[::1]");
        return 1;
    }
    for (i = 0; i < 10 && !b[i]; ++i) {}
    if (i == 10 && b[10] == 0xff && b[11] == 0xff) {
        *loopback = b[12] == 127;
        snprintf(out, cap, "[::ffff:%u.%u.%u.%u]", b[12], b[13], b[14], b[15]);
        return 1;
    }
    snprintf(out, cap, "[%x:%x:%x:%x:%x:%x:%x:%x]", b[0] << 8 | b[1], b[2] << 8 | b[3], b[4] << 8 | b[5],
             b[6] << 8 | b[7], b[8] << 8 | b[9], b[10] << 8 | b[11], b[12] << 8 | b[13], b[14] << 8 | b[15]);
    return 1;
}

static void collect_ports(Ctx *c) {
    static const struct { const char *file, *proto; int v6, udp; } tables[] = {
        {"/proc/net/tcp", "tcp", 0, 0}, {"/proc/net/tcp6", "tcp6", 1, 0},
        {"/proc/net/udp", "udp", 0, 1}, {"/proc/net/udp6", "udp6", 1, 1}};
    SockMap owners;
    size_t t;
    int any = 0, r;
    char probe[INV_PATH];
    struct stat st;
    if (!full_path(c, "/proc/net/tcp", probe) || stat(probe, &st)) {
        note(c->notes, "portas: /proc/net indisponível");
        return;
    }
    owners = socket_owners(c);
    for (t = 0; t < sizeof(tables) / sizeof(tables[0]); ++t) {
        char line[1024];
        FILE *f = open_shown(c, tables[t].file);
        if (!f) continue;
        any = 1;
        if (read_line(f, line, sizeof(line)) == 0) { /* header */
            fclose(f);
            continue;
        }
        while ((r = read_line(f, line, sizeof(line))) != 0) {
            char local[40], remote[40], lhex[8], rhex[8], shex[8], addr[64], key[96], detail[128];
            unsigned lport, rport, state;
            unsigned long uid, inode;
            int loop, k, remote_zero = 1;
            if (r < 0) continue;
            /* Ports are exactly 4 hex digits and the state 2: a longer field is not this format. */
            if (sscanf(line, " %*d: %39[0-9A-Fa-f]:%7[0-9A-Fa-f] %39[0-9A-Fa-f]:%7[0-9A-Fa-f] %7[0-9A-Fa-f] %*s %*s %*s %lu %*d %lu",
                       local, lhex, remote, rhex, shex, &uid, &inode) != 7 ||
                strlen(lhex) != 4 || strlen(rhex) != 4 || strlen(shex) != 2)
                continue;
            lport = (unsigned)strtoul(lhex, NULL, 16);
            rport = (unsigned)strtoul(rhex, NULL, 16);
            state = (unsigned)strtoul(shex, NULL, 16);
            for (k = 0; remote[k]; ++k) remote_zero &= remote[k] == '0';
            if (tables[t].udp ? !(state == 0x07 && rport == 0 && remote_zero) : state != 0x0A) continue;
            if (!format_addr(local, tables[t].v6, addr, sizeof(addr), &loop)) continue;
            snprintf(key, sizeof(key), "%s %s:%u", tables[t].proto, addr, lport);
            snprintf(detail, sizeof(detail), "processo=%s uid=%lu exposta=%s", owner_of(&owners, inode), uid,
                     loop ? "nao" : "sim");
            inv_add(c->out, "porta", key, detail);
        }
        fclose(f);
    }
    free(owners.v);
    if (!any) note(c->notes, "portas: tabelas de /proc/net ilegíveis");
}

/* ---- filesystem walks ---- */

/* SUID/SGID binaries and world-writable paths (sticky directories like /tmp excepted). */
static void scan_perms(Ctx *c, const char *shown, int depth, dev_t dev) {
    char p[INV_PATH], detail[128], hex[2 * HASH_BYTES + 1];
    struct stat st;
    DIR *d;
    struct dirent *e;
    if (!full_path(c, shown, p) || lstat(p, &st)) return;
    if (S_ISLNK(st.st_mode)) return;
    if (dev != (dev_t)-1 && st.st_dev != dev) return; /* another file system */
    if (S_ISREG(st.st_mode)) {
        if (st.st_mode & (S_ISUID | S_ISGID)) {
            if (!hash_file(p, hex)) snprintf(hex, sizeof(hex), "?");
            snprintf(detail, sizeof(detail), "modo=%04o dono=%lu sha256=%s", (unsigned)(st.st_mode & 07777),
                     (unsigned long)st.st_uid, hex);
            inv_add(c->out, "suid", shown, detail);
        }
        if (st.st_mode & S_IWOTH) inv_add(c->out, "gravavel", shown, "arquivo gravável por todos");
        return;
    }
    if (!S_ISDIR(st.st_mode)) return;
    if ((st.st_mode & S_IWOTH) && !(st.st_mode & S_ISVTX))
        inv_add(c->out, "gravavel", shown, "pasta gravável por todos, sem sticky bit");
    if (depth <= 0) return;
    if (!(d = opendir(p))) {
        if (errno == EACCES) note(c->notes, "sem acesso: %s", shown);
        return;
    }
    while ((e = readdir(d)) != NULL) {
        char child[INV_PATH];
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        if (snprintf(child, sizeof(child), "%s/%s", shown, e->d_name) >= (int)sizeof(child)) continue;
        scan_perms(c, child, depth - 1, st.st_dev);
    }
    closedir(d);
}

/* Records every file under shown with its hash (links with their target). */
static void track(Ctx *c, const char *category, const char *shown, int depth) {
    char p[INV_PATH], detail[INV_PATH + 16], hex[2 * HASH_BYTES + 1];
    struct stat st;
    DIR *d;
    struct dirent *e;
    if (!full_path(c, shown, p)) return;
    if (lstat(p, &st)) {
        if (errno == EACCES) note(c->notes, "sem acesso: %s", shown);
        return;
    }
    if (S_ISLNK(st.st_mode)) {
        char target[INV_PATH];
        ssize_t n = readlink(p, target, sizeof(target) - 1);
        if (n < 0) n = 0;
        target[n] = 0;
        snprintf(detail, sizeof(detail), "-> %s", target);
        inv_add(c->out, category, shown, detail);
        return;
    }
    if (S_ISREG(st.st_mode)) {
        if (hash_file(p, hex)) snprintf(detail, sizeof(detail), "sha256=%s", hex);
        else snprintf(detail, sizeof(detail), "sem acesso");
        inv_add(c->out, category, shown, detail);
        return;
    }
    if (!S_ISDIR(st.st_mode) || depth <= 0) return;
    if (!(d = opendir(p))) {
        if (errno == EACCES) note(c->notes, "sem acesso: %s", shown);
        return;
    }
    while ((e = readdir(d)) != NULL) {
        char child[INV_PATH];
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        if (snprintf(child, sizeof(child), "%s/%s", shown, e->d_name) >= (int)sizeof(child)) continue;
        track(c, category, child, depth - 1);
    }
    closedir(d);
}

static void collect_tracked(Ctx *c) {
    static const struct { const char *category, *path; int depth; } paths[] = {
        /* persistence: what runs at boot, login or on a schedule */
        {"inicio", "/etc/crontab", 0}, {"inicio", "/etc/anacrontab", 0}, {"inicio", "/etc/cron.d", 1},
        {"inicio", "/etc/cron.hourly", 1}, {"inicio", "/etc/cron.daily", 1}, {"inicio", "/etc/cron.weekly", 1},
        {"inicio", "/etc/cron.monthly", 1}, {"inicio", "/var/spool/cron", 2}, {"inicio", "/etc/systemd/system", 3},
        {"inicio", "/etc/systemd/user", 3}, {"inicio", "/etc/rc.local", 0}, {"inicio", "/etc/init.d", 1},
        {"inicio", "/etc/xdg/autostart", 1}, {"inicio", "/etc/profile", 0}, {"inicio", "/etc/profile.d", 1},
        {"inicio", "/etc/bash.bashrc", 0}, {"inicio", "/etc/environment", 0}, {"inicio", "/etc/ld.so.preload", 0},
        /* security configuration */
        {"config", "/etc/ssh/sshd_config", 0}, {"config", "/etc/ssh/sshd_config.d", 1}, {"config", "/etc/sudoers", 0},
        {"config", "/etc/sudoers.d", 1}, {"config", "/etc/pam.d", 1}, {"config", "/etc/hosts", 0},
        {"config", "/etc/ld.so.conf", 0}, {"config", "/etc/ld.so.conf.d", 1}};
    static const char *const home_files[] = {"/.bashrc", "/.bash_profile", "/.profile", "/.zshrc"};
    size_t i, u, h;
    char p[INV_PATH];
    struct stat st;
    for (i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i) track(c, paths[i].category, paths[i].path, paths[i].depth);
    if (full_path(c, "/etc/ld.so.preload", p) && !lstat(p, &st))
        inv_add(c->out, "alerta", "ld.so.preload", "biblioteca carregada em todo programa; raro em sistema normal");

    for (u = 0; u < c->nusers; ++u) {
        const User *us = &c->users[u];
        int seen = 0;
        if (!(us->login || us->uid == 0) || us->home[0] != '/' || !us->home[1]) continue;
        for (h = 0; h < u && !seen; ++h)
            seen = (c->users[h].login || c->users[h].uid == 0) && !strcmp(c->users[h].home, us->home);
        if (seen) continue;
        for (h = 0; h < sizeof(home_files) / sizeof(home_files[0]); ++h) {
            snprintf(p, sizeof(p), "%s%s", us->home, home_files[h]);
            track(c, "inicio", p, 0);
        }
        snprintf(p, sizeof(p), "%s/.config/autostart", us->home);
        track(c, "inicio", p, 1);
        snprintf(p, sizeof(p), "%s/.config/systemd/user", us->home);
        track(c, "inicio", p, 3);
    }
}

static int key_type(const char *t) {
    return !strncmp(t, "ssh-", 4) || !strncmp(t, "ecdsa-", 6) || !strncmp(t, "sk-", 3);
}

/* One item per authorized key: who can log in without a password. */
static void collect_ssh_keys(Ctx *c) {
    static const char *const files[] = {"/.ssh/authorized_keys", "/.ssh/authorized_keys2"};
    size_t u, k;
    for (u = 0; u < c->nusers; ++u) {
        const User *us = &c->users[u];
        if (us->home[0] != '/') continue;
        for (k = 0; k < 2; ++k) {
            char shown[INV_PATH], line[INV_LINE];
            FILE *f;
            int r;
            if (snprintf(shown, sizeof(shown), "%s%s", strcmp(us->home, "/") ? us->home : "", files[k]) >=
                (int)sizeof(shown))
                continue;
            if (!(f = open_shown(c, shown))) continue;
            while ((r = read_line(f, line, sizeof(line))) != 0) {
                char *tok[64], *s = line, key[256], detail[INV_PATH + 640], hex[2 * HASH_BYTES + 1], comment[256] = "";
                int n = 0, i;
                if (r < 0) { note(c->notes, "linha longa ou inválida ignorada: %s", shown); continue; }
                /* Fields split on blanks, except inside "..." in the options (sshd's rule):
                 * command="echo ssh-rsa x" is one field, not a key. */
                while (n < 64) {
                    char *w;
                    int quoted = 0;
                    while (*s == ' ' || *s == '\t') ++s;
                    if (!*s || (*s == '#' && n == 0)) break;
                    tok[n++] = w = s;
                    for (; *s && (quoted || (*s != ' ' && *s != '\t')); ++s) {
                        if (*s == '\\' && quoted && s[1] == '"') *w++ = *s++;
                        else if (*s == '"') quoted = !quoted;
                        *w++ = *s;
                    }
                    if (quoted) { n = 0; break; } /* unterminated quote: sshd rejects the line */
                    if (*s) ++s;
                    *w = 0;
                }
                for (i = 0; i + 1 < n && !key_type(tok[i]); ++i) {}
                if (i + 1 >= n) continue;
                sha256_hex(tok[i + 1], strlen(tok[i + 1]), HASH_BYTES, hex);
                if (i + 2 < n) snprintf(comment, sizeof(comment), "%s", tok[i + 2]);
                snprintf(key, sizeof(key), "%s:%s", us->name, hex);
                snprintf(detail, sizeof(detail), "tipo=%s comentario=%s opcoes=%s arquivo=%s", tok[i],
                         comment[0] ? comment : "-", i ? "sim" : "nao", shown);
                inv_add(c->out, "chave_ssh", key, detail);
            }
            fclose(f);
        }
    }
}

void inventory_collect(const char *root, InvList *out, InvList *notes) {
    static const char *const perm_roots[] = {"/bin", "/sbin", "/usr", "/opt", "/etc"};
    Ctx c;
    size_t i, n;
    memset(&c, 0, sizeof(c));
    snprintf(c.root, sizeof(c.root), "%s", root ? root : "");
    n = strlen(c.root);
    while (n && c.root[n - 1] == '/') c.root[--n] = 0;
    c.out = out;
    c.notes = notes;

    collect_users(&c);
    collect_admins(&c);
    collect_ports(&c);
    for (i = 0; i < sizeof(perm_roots) / sizeof(perm_roots[0]); ++i) scan_perms(&c, perm_roots[i], 24, (dev_t)-1);
    collect_tracked(&c);
    collect_ssh_keys(&c);

    free(c.users);
    inv_finish(out);
    inv_finish(notes);
}

#endif
