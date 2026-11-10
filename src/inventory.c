#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#  define _DARWIN_C_SOURCE 1
#  define _FILE_OFFSET_BITS 64
#endif

#include "inventory.h"
#include "sha256.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#  include <dirent.h>
#  include <fcntl.h>
#  include <grp.h> /* setgroups */
#  include <sys/stat.h>
#  include <unistd.h>
#endif

#define INV_PATH 4096
#define INV_LINE 8192 /* longest line kept from a system file; a longer one keeps its start */
#define HASH_BYTES 32 /* the whole SHA-256: a cut digest can be collided on purpose */
/* Bytes read from one system file, and hashed from one tracked file: a user's sparse
 * 1 TB .bashrc or authorized_keys must not keep a root scan reading for hours. */
#define INV_TEXT_MAX (64ull << 20)
#define INV_HASH_MAX (256ull << 20)

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

#define LINE_LONG (-1) /* longer than cap-1 bytes, or cut by the byte limit: buf keeps its start */
#define LINE_NUL (-2)  /* holds a NUL: buf keeps what came before it, all glibc's parsers see */

/* One line of a text file into buf, without the line ending. Returns 1, 0 at the
 * end, LINE_LONG or LINE_NUL. With left, a bad line is consumed whole, so its tail
 * is never read as a line of its own (a long gecos field must not become a fake
 * account), and at most *left bytes are read from the file. Without it the caller
 * gives up at the first bad line, so reading stops at the first bad byte. */
static int read_line(FILE *f, char *buf, size_t cap, unsigned long long *left) {
    size_t n = 0;
    int ch = EOF, bad = 0;
    for (;;) {
        if (left && !*left) {
            if (n || bad) bad = bad ? bad : LINE_LONG; /* the limit cut this line */
            break;
        }
        if ((ch = getc(f)) == EOF) break;
        if (left) --*left;
        if (ch == '\n') break;
        if (bad) continue;
        if (ch && n + 1 < cap) {
            buf[n++] = (char)ch;
            continue;
        }
        bad = ch ? LINE_LONG : LINE_NUL;
        if (!left) break;
    }
    buf[n] = 0;
    if (ch == EOF && n == 0 && !bad) return 0;
    return bad ? bad : 1;
}

/* Drops a CR before the line ending (a file edited on Windows). Not for glibc's databases:
 * glibc keeps the CR as part of the last field, and so must their reading here. */
static void strip_cr(char *s) {
    size_t n = strlen(s);
    if (n && s[n - 1] == '\r') s[n - 1] = 0;
}

/* ---- report files --------------------------------------------------------- */

/* Report metadata comes from outside (a --raiz path, the host name): never let it break a line. */
static void put_meta(FILE *f, const char *label, const char *s) {
    fprintf(f, "# %s: ", label);
    for (; *s; ++s) fputc((unsigned char)*s < 32 || *s == 127 ? '?' : *s, f);
    fputc('\n', f);
}

void inventory_write(FILE *f, const InvList *items, const InvList *notes, const char *host, const char *when,
                     const char *root) {
    size_t i;
    fprintf(f, "# %s\n", INV_FORMAT);
    put_meta(f, "host", host);
    put_meta(f, "data", when);
    put_meta(f, "raiz", root && *root ? root : "/");
    fprintf(f, "# itens: %zu\n", items->count);
    for (i = 0; notes && i < notes->count; ++i) put_meta(f, "nota", notes->items[i]);
    for (i = 0; i < items->count; ++i) fprintf(f, "%s\n", items->items[i]);
}

/* A report to read. On POSIX a regular file or a pipe (`--comparar base.txt <(ssh host ...)`),
 * checked before the open: opening a device is already an act. */
static FILE *open_report(const char *path, char *err, size_t cap) {
#if defined(_WIN32)
    FILE *f = fopen(path, "rb");
    if (!f) snprintf(err, cap, "%s: %s", path, strerror(errno));
    return f;
#else
    struct stat st;
    FILE *f;
    int fd;
    if (stat(path, &st)) {
        snprintf(err, cap, "%s: %s", path, strerror(errno));
        return NULL;
    }
    if (!S_ISREG(st.st_mode) && !S_ISFIFO(st.st_mode)) {
        snprintf(err, cap, "%s: não é um arquivo comum", path);
        return NULL;
    }
    /* A pipe the operator named waits for its writer; a file opens with O_NONBLOCK, so a FIFO swapped
     * in for it cannot hold the open, and is then refused below. */
    if ((fd = open(path, O_RDONLY | O_NOCTTY | (S_ISFIFO(st.st_mode) ? 0 : O_NONBLOCK))) < 0 || fstat(fd, &st)) {
        snprintf(err, cap, "%s: %s", path, strerror(errno));
        if (fd >= 0) close(fd);
        return NULL;
    }
    if (!S_ISREG(st.st_mode) && !S_ISFIFO(st.st_mode)) {
        snprintf(err, cap, "%s: não é um arquivo comum", path);
        close(fd);
        return NULL;
    }
    if (S_ISFIFO(st.st_mode) && fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) & ~O_NONBLOCK) == -1) {
        snprintf(err, cap, "%s: %s", path, strerror(errno));
        close(fd);
        return NULL;
    }
    if (!(f = fdopen(fd, "rb"))) {
        snprintf(err, cap, "%s: %s", path, strerror(errno));
        close(fd);
    }
    return f;
#endif
}

int inventory_read(const char *path, InvList *out, InvList *notes, char *err, size_t cap) {
    char line[INV_ITEM_MAX + 1];
    unsigned long number = 0, declared = 0, seen = 0;
    int header = 0, counted = 0, r;
    FILE *f = open_report(path, err, cap);
    if (!f) return 0;
    while ((r = read_line(f, line, sizeof(line), NULL)) != 0) {
        size_t n, i;
        const char *t1, *t2;
        char *s;
        strip_cr(line);
        n = strlen(line);
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
        if (!strncmp(line, "# itens: ", 9) && !counted) {
            char *end;
            declared = strtoul(line + 9, &end, 10);
            counted = !*end;
            continue;
        }
        if (!strncmp(line, "# nota: ", 8)) {
            if (notes) inv_add(notes, "nota", line + 8, "");
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
        ++seen;
    }
    if (ferror(f)) {
        snprintf(err, cap, "%s: erro de leitura", path);
        fclose(f);
        return 0;
    }
    fclose(f);
    if (!header) {
        snprintf(err, cap, "%s: arquivo vazio", path);
        return 0;
    }
    /* A report cut after a whole line still parses: the declared count tells it is incomplete. */
    if (!counted || declared != seen) {
        snprintf(err, cap, "%s: relatório incompleto (%lu itens declarados, %lu lidos)", path, declared, seen);
        return 0;
    }
    inv_finish(out);
    if (notes) inv_finish(notes);
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

int inventory_collect(const char *root, InvList *out, InvList *notes) {
    (void)root;
    (void)out;
    note(notes, "coleta disponível só em Linux (e parcial em outros sistemas POSIX)");
    return 0;
}

#else

typedef struct User {
    char name[64], home[512], shell[256];
    unsigned long uid, gid;
    int login;
} User;

typedef struct Ctx {
    char root[INV_PATH];      /* "" for this machine */
    char root_real[INV_PATH]; /* realpath(root) with --raiz */
    InvList *out, *notes;
    User *users;
    size_t nusers;
    int as_user;    /* reading inside a user's home: open as them (see open_as) */
    uid_t as_uid;
    gid_t as_gid;
} Ctx;

static int full_path(const Ctx *c, const char *shown, char *out) {
    int n = snprintf(out, INV_PATH, "%s%s", c->root, shown);
    return n > 0 && n < INV_PATH;
}

/* With --raiz, whether a resolved path stays inside the mounted system. */
static int under_root(const Ctx *c, const char *resolved) {
    size_t n = strlen(c->root_real);
    if (!c->root[0]) return 1;
    if (n == 1 && c->root_real[0] == '/') return 1;
    return !strncmp(resolved, c->root_real, n) && (resolved[n] == 0 || resolved[n] == '/');
}

/* With --raiz, a symlink in the image (/etc -> /etc, /home -> /mnt/x) must not
 * lead out of it: the report would then describe the machine doing the scan.
 * follow_last: the last component is opened too (fopen, opendir), not only lstat'ed;
 * then, if out is given, it receives the path realpath checked ("" when it could not
 * be resolved), so the file opened is the one found inside the image.
 * Returns 1 when safe; 0 when the path is missing or escapes (escapes are noted). */
static int confined(Ctx *c, const char *p, const char *shown, int follow_last, char out[INV_PATH * 2]) {
    char tmp[INV_PATH], resolved[INV_PATH * 2];
    char *slash;
    if (out) out[0] = 0;
    if (!c->root[0]) return 1;
    if (follow_last) {
        if (!realpath(p, resolved)) return errno != ENOENT && errno != ENOTDIR;
        if (out && under_root(c, resolved)) memcpy(out, resolved, sizeof(resolved));
    } else {
        snprintf(tmp, sizeof(tmp), "%s", p);
        slash = strrchr(tmp, '/');
        if (!slash) return 1;
        if (slash == tmp) slash[1] = 0;
        else *slash = 0;
        if (!realpath(tmp, resolved)) return errno != ENOENT && errno != ENOTDIR;
    }
    if (under_root(c, resolved)) return 1;
    note(c->notes, "fora da raiz, não lido: %s", shown);
    return 0;
}

/* An open that failed: a missing file says nothing, any other reason leaves a note. */
static void open_failed(Ctx *c, const char *shown, int err) {
    if (err == ENOENT || err == ENOTDIR) return;
    if (err == EACCES) note(c->notes, "sem acesso: %s", shown);
    else note(c->notes, "não lido (%s): %s", strerror(err), shown);
}

/* Something that is not a regular file where one is expected: an alert, never a silence. */
static void not_regular(Ctx *c, const char *shown, mode_t mode) {
    char key[INV_PATH + 16], detail[64];
    snprintf(key, sizeof(key), "nao_regular:%s", shown);
    snprintf(detail, sizeof(detail), "%s no lugar de um arquivo",
             S_ISFIFO(mode) ? "fifo" : S_ISSOCK(mode) ? "socket" : S_ISCHR(mode) || S_ISBLK(mode) ? "dispositivo"
                 : S_ISDIR(mode) ? "pasta" : "outro tipo");
    inv_add(c->out, "alerta", key, detail);
}

#define ROOT_GROUPS 1024 /* supplementary groups of the scan saved around open_as; more fails closed */

/* Back to the identity open_as started from; a scan that cannot get back stops. */
static void back_to_root(uid_t uid, gid_t gid, const gid_t *groups, int n) {
    if (seteuid(uid) || setegid(gid) || (n >= 0 && setgroups((size_t)n, groups))) {
        fputs("smolmoe-inventario: não consegui voltar a root depois de ler como usuário\n", stderr);
        exit(2);
    }
}

/* Opens path read-only, without waiting on a FIFO. Inside a user's home on this machine,
 * run as root, the open runs as the owner (groups, egid, then euid, as sshd's
 * temporarily_use_uid reads authorized_keys): a link the user swaps in after the checks
 * cannot make root open what only root may (opening /dev/watchdog arms it). If the switch
 * does not take (seteuid(-1) changes nothing), nothing is opened. An image read with
 * --raiz has no one swapping files under it. */
static int open_as(Ctx *c, const char *path, int flags) {
    static gid_t groups[ROOT_GROUPS];
    uid_t was = geteuid();
    gid_t was_g = getegid(), g = c->as_gid;
    int fd, err, n = -1, drop = c->as_user && !c->root[0] && was == 0 && c->as_uid != 0;
    if (drop) {
        errno = 0;
        if ((n = getgroups(ROOT_GROUPS, groups)) < 0 || setgroups(1, &g) || setegid(g) || seteuid(c->as_uid) ||
            geteuid() != c->as_uid || getegid() != g) {
            err = errno ? errno : EPERM;
            back_to_root(was, was_g, groups, n);
            errno = err;
            return -1;
        }
    }
    fd = open(path, O_RDONLY | O_NONBLOCK | O_NOCTTY | flags);
    err = errno;
    if (drop) back_to_root(was, was_g, groups, n);
    errno = err;
    return fd;
}

/* Opens a file to read, only when it is a regular file. It is checked before the open,
 * because opening is already an act (a FIFO blocks, /dev/watchdog arms, a tape rewinds),
 * and again on the open descriptor, the same file, because it can be swapped in between.
 * sshd refuses anything but a regular file as authorized_keys too. With --raiz the path
 * opened is the one realpath found inside the image, with no link left in it. */
static FILE *open_shown(Ctx *c, const char *shown) {
    char p[INV_PATH], resolved[INV_PATH * 2];
    const char *path;
    struct stat before, st;
    FILE *f;
    int fd;
    if (!full_path(c, shown, p) || !confined(c, p, shown, 1, resolved)) return NULL;
    path = resolved[0] ? resolved : p;
    if (stat(path, &before)) {
        open_failed(c, shown, errno);
        return NULL;
    }
    if (!S_ISREG(before.st_mode)) {
        not_regular(c, shown, before.st_mode);
        return NULL;
    }
    if ((fd = open_as(c, path, resolved[0] ? O_NOFOLLOW : 0)) < 0) {
        if (errno == ENOENT) note(c->notes, "sumiu durante a leitura: %s", shown); /* stat had seen it */
        else open_failed(c, shown, errno);
        return NULL;
    }
    if (fstat(fd, &st)) {
        note(c->notes, "não lido (%s): %s", strerror(errno), shown);
        close(fd);
        return NULL;
    }
    if (!S_ISREG(st.st_mode) || st.st_dev != before.st_dev || st.st_ino != before.st_ino) {
        if (!S_ISREG(st.st_mode)) not_regular(c, shown, st.st_mode);
        else note(c->notes, "trocado durante a leitura, não lido: %s", shown);
        close(fd);
        return NULL;
    }
    if (!(f = fdopen(fd, "rb"))) {
        note(c->notes, "não lido (%s): %s", strerror(errno), shown);
        close(fd);
    }
    return f;
}

/* Closes a file read with read_line under a byte limit; a read error or a file cut by
 * the limit leaves a note (the data may be partial). */
static void close_read(Ctx *c, FILE *f, const char *shown, unsigned long long left) {
    if (ferror(f)) note(c->notes, "erro de leitura, dados talvez incompletos: %s", shown);
    else if (!left && getc(f) != EOF)
        note(c->notes, "maior que o limite de %llu MiB, lido só o início: %s", INV_TEXT_MAX >> 20, shown);
    fclose(f);
}

enum { HASH_OK, HASH_ERROR, HASH_NOT_REGULAR, HASH_TOO_BIG };

/* The file's full SHA-256 in hex. Every caller has seen a regular file at path whose last
 * component is no link (lstat, or realpath's result), so O_NOFOLLOW: a link swapped in
 * afterwards fails instead of being followed. The file is checked again on the open
 * descriptor, since its owner can swap it for a FIFO in between, and inside a home it is
 * opened with the owner's euid (open_as). HASH_ERROR leaves errno; HASH_NOT_REGULAR the type in *mode. */
static int hash_file(Ctx *c, const char *path, char hex[2 * HASH_BYTES + 1], mode_t *mode) {
    static const char digits[] = "0123456789abcdef";
    unsigned char buf[65536], d[32];
    unsigned long long total = 0;
    struct stat st;
    Sha256 s;
    ssize_t n;
    int i, err, fd = open_as(c, path, O_NOFOLLOW);
    if (fd < 0) return HASH_ERROR;
    if (fstat(fd, &st)) goto fail;
    if (!S_ISREG(st.st_mode)) {
        *mode = st.st_mode;
        close(fd);
        return HASH_NOT_REGULAR;
    }
    if ((unsigned long long)st.st_size > INV_HASH_MAX) {
        close(fd);
        return HASH_TOO_BIG;
    }
    sha256_init(&s);
    while ((n = read(fd, buf, sizeof(buf))) != 0) {
        if (n < 0) {
            if (errno == EINTR) continue;
            goto fail;
        }
        if ((total += (unsigned long long)n) > INV_HASH_MAX) { /* grew while being read */
            close(fd);
            return HASH_TOO_BIG;
        }
        sha256_update(&s, buf, (size_t)n);
    }
    close(fd);
    sha256_final(&s, d);
    for (i = 0; i < HASH_BYTES; ++i) hex[2 * i] = digits[d[i] >> 4], hex[2 * i + 1] = digits[d[i] & 15];
    hex[2 * HASH_BYTES] = 0;
    return HASH_OK;
fail:
    err = errno;
    close(fd);
    errno = err;
    return HASH_ERROR;
}

/* What a hash that failed means for a file the report expected to describe. */
static void hash_failed(Ctx *c, const char *shown, int r, mode_t mode) {
    if (r == HASH_NOT_REGULAR) not_regular(c, shown, mode);
    else if (r == HASH_TOO_BIG) note(c->notes, "maior que %llu MiB, sem hash: %s", INV_HASH_MAX >> 20, shown);
    else if (errno == ENOENT) note(c->notes, "sumiu durante a leitura: %s", shown); /* lstat had seen it */
    else open_failed(c, shown, errno);
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

/* isspace in the C locale: what glibc skips before a line and strtoul before a number. */
static int c_space(int ch) { return ch == ' ' || (ch >= '\t' && ch <= '\r'); }

/* A line of a glibc database (passwd, shadow, group). nss_files skips the blanks before
 * a line, however many: they are skipped here too, before the line is kept, so no run
 * of blanks pushes an entry out of buf. *blanks: there were some. tail: read the last
 * line as __nss_readline does (getpwnam, getspnam, getgrnam), not as getline does
 * (initgroups, which reads comments too). */
static int read_entry(FILE *f, char *buf, size_t cap, unsigned long long *left, int *blanks, int tail) {
    unsigned long long skipped = 0, i;
    char lead[INV_LINE]; /* the blanks, for the copy below */
    size_t n;
    int ch, r;
    while (*left && (ch = getc(f)) != EOF) {
        if (ch == '\n' || !c_space(ch)) {
            ungetc(ch, f);
            break;
        }
        if (skipped < sizeof(lead)) lead[skipped] = (char)ch;
        --*left;
        ++skipped;
    }
    *blanks = skipped > 0;
    r = read_line(f, buf, cap, left);
    /* glibc (2.30 on, __nss_readline) moves a line over its blanks with memmove(buf, p, strlen(p)),
     * leaving the terminator behind, so the line keeps its last `skipped` bytes twice ("\tb::1::" is
     * read as "b::1:::") wherever no newline was copied along to cut it: the last line, with no
     * newline, and any line with a NUL (strlen stops there). PAM's getspnam reads it so; so does this. */
    if (!tail || !skipped || !(r == LINE_NUL || (r == 1 && feof(f)))) return r;
    n = strlen(buf);
    for (i = 0; i < skipped; ++i) {
        if (n + i + 1 >= cap || n + i >= sizeof(lead)) {
            buf[n + i] = 0;
            return LINE_LONG;
        }
        buf[n + i] = n + i < skipped ? lead[n + i] : buf[n + i - skipped];
    }
    buf[n + skipped] = 0;
    return r;
}

/* A uid or gid field read the way glibc's nss_files reads it (INT_FIELD over
 * strtou32): blanks, a sign and decimal digits, then nothing. "-1" wraps to
 * 4294967295 and anything above that saturates, as strtoul then the 32-bit clamp do.
 * Returns 0 when glibc rejects the line ("", "x", "1x": strtoul alone would say 0).
 * *canonical: the field is the number as printed, not "-0", "+0", " 0" or "007". */
static int glibc_id(const char *s, unsigned long *out, int *canonical) {
    const char *p = s;
    unsigned long long v = 0;
    int neg = 0, over = 0, digits = 0;
    char printed[24];
    while (c_space((unsigned char)*p)) ++p;
    if (*p == '+' || *p == '-') neg = *p++ == '-';
    for (; *p >= '0' && *p <= '9'; ++p, ++digits) {
        unsigned d = (unsigned)(*p - '0');
        if (v > (ULLONG_MAX - d) / 10) over = 1;
        else v = v * 10 + d;
    }
    if (!digits || *p) return 0;
    if (over) v = ULLONG_MAX; /* strtoul's ERANGE value, whatever the sign */
    else if (neg) v = 0 - v;
    if (v > 0xFFFFFFFFull) v = 0xFFFFFFFFull;
    *out = (unsigned long)v;
    snprintf(printed, sizeof(printed), "%lu", *out);
    *canonical = !strcmp(printed, s);
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

#define NAME_SHOWN 63 /* bytes of an account's name in an alert key; longer names are cut, and the alert says so */

/* A name for an alert key: at most NAME_SHOWN bytes, never splitting a UTF-8 character. */
static void short_name(char out[NAME_SHOWN + 2], const char *s) {
    size_t n = strlen(s);
    if (n > NAME_SHOWN + 1) n = NAME_SHOWN + 1;
    memcpy(out, s, n);
    out[n] = 0;
    cut_utf8(out, NAME_SHOWN);
}

static void add_why(char *why, size_t cap, const char *reason) {
    size_t n = strlen(why);
    snprintf(why + n, cap - n, "%s%s", n ? ", " : "", reason);
}

/* One /etc/passwd line (after the blanks read_entry skipped), read the way glibc's
 * nss_files reads it: comments skipped, missing fields empty, the line ending at a NUL. An account glibc
 * accepts is never dropped: one that does not fit the report or is not written the
 * plain way becomes an alert with its name and line number. A line glibc rejects
 * leaves a note with its line number, so two such lines never merge into one note. */
static void passwd_line(Ctx *c, char *line, int r, int blanks, unsigned long no) {
    static char empty[1];
    char *fl[7], why[256] = "", uid_text[24] = "?", name[NAME_SHOWN + 2], key[NAME_SHOWN + 32], detail[512];
    User u;
    int n, i, uid_ok = 0, gid_ok = 0, uid_canon = 1, gid_canon = 1, cut = r == LINE_LONG, fits;
    int uid_known, gid_known;
    if (!*line || *line == '#') return;
    n = split_colons(line, fl, 7);
    for (i = n; i < 7; ++i) fl[i] = empty;
    memset(&u, 0, sizeof(u));
    /* In a cut line the last field read may be cut too: a number is known once a ':' follows it. */
    uid_known = n > 3 || (!cut && n == 3);
    gid_known = n > 4 || (!cut && n == 4);
    if (uid_known) uid_ok = glibc_id(fl[2], &u.uid, &uid_canon);
    if (gid_known) gid_ok = glibc_id(fl[3], &u.gid, &gid_canon);
    if ((uid_known && !uid_ok) || (gid_known && !gid_ok) || (!cut && n < 4)) {
        note(c->notes, "/etc/passwd linha %lu: não é conta para a glibc, ignorada", no);
        return;
    }
    fits = !cut && fl[0][0] && strlen(fl[0]) < sizeof(u.name) && strlen(fl[5]) < sizeof(u.home) &&
           strlen(fl[6]) < sizeof(u.shell);
    if (uid_known && !uid_canon) add_why(why, sizeof(why), "uid escrito fora do padrão");
    if (gid_known && !gid_canon) add_why(why, sizeof(why), "gid escrito fora do padrão");
    if (!cut && n < 7) add_why(why, sizeof(why), "campos faltando");
    if (r == LINE_NUL) add_why(why, sizeof(why), "byte nulo na linha");
    if (blanks) add_why(why, sizeof(why), "espaço antes do nome");
    if (cut) add_why(why, sizeof(why), "linha longa demais");
    if (!fl[0][0]) add_why(why, sizeof(why), "sem nome");
    if (strlen(fl[0]) > NAME_SHOWN) add_why(why, sizeof(why), "nome cortado");
    else if (strlen(fl[0]) >= sizeof(u.name)) add_why(why, sizeof(why), "nome grande demais");
    if (strlen(fl[5]) >= sizeof(u.home)) add_why(why, sizeof(why), "home grande demais");
    if (strlen(fl[6]) >= sizeof(u.shell)) add_why(why, sizeof(why), "shell grande demais");
    if (why[0]) {
        short_name(name, fl[0]);
        if (uid_ok) snprintf(uid_text, sizeof(uid_text), "%lu", u.uid);
        snprintf(key, sizeof(key), "conta_anormal:%s", name);
        snprintf(detail, sizeof(detail), "uid=%s linha=%lu: %s%s", uid_text, no, why,
                 fits ? "" : "; fora da lista de usuários, arquivos da conta não lidos");
        inv_add(c->out, "alerta", key, detail);
    }
    if (!fits) return;
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

/* Empty, or a number glibc's INT_FIELD_MAYBE_NULL reads. */
static int num_or_empty(const char *s) {
    unsigned long v;
    int canon;
    return !s[0] || glibc_id(s, &v, &canon);
}

static int blank_only(const char *s) {
    for (; *s; ++s)
        if (!c_space((unsigned char)*s)) return 0;
    return 1;
}

/* Whether glibc's shadow parser accepts a whole line split in n fields (fl holds 10, the
 * 10th taking the rest), as fgetspent does on glibc 2.39: lstchg, min and max empty or a
 * number; then glibc skips blanks and takes either the old form (max non-empty and last,
 * or only blanks after it) or warn, inact, a non-empty expire and an optional flag, never
 * more than 9 fields. */
static int shadow_accepted(char **fl, int n) {
    int i;
    if (n < 5 || n > 9) return 0;
    for (i = 2; i < n; ++i)
        if (!num_or_empty(fl[i]) && !(i == 5 && blank_only(fl[5]))) return 0;
    if (n == 5) return fl[4][0] != 0;
    if (n == 6) return blank_only(fl[5]);
    return n == 9 || (n == 8 && fl[7][0]);
}

/* 0 when /etc/passwd cannot be read: without it there is no system to describe. */
static int collect_users(Ctx *c) {
    char line[INV_LINE], key[INV_LINE + 32];
    unsigned long long left = INV_TEXT_MAX;
    unsigned long no = 0;
    int r, blanks;
    FILE *f = open_shown(c, "/etc/passwd");
    if (!f) return 0;
    while ((r = read_entry(f, line, sizeof(line), &left, &blanks, 1)) != 0) passwd_line(c, line, r, blanks, ++no);
    close_read(c, f, "/etc/passwd", left);

    f = open_shown(c, "/etc/shadow");
    if (!f) return 1;
    left = INV_TEXT_MAX;
    no = 0;
    while ((r = read_entry(f, line, sizeof(line), &left, &blanks, 1)) != 0) {
        char *fl[10];
        int n, i, ok, cut = r == LINE_LONG;
        /* A NUL-cut line is what glibc parses. A long line keeps its start: a password cut there is long,
         * so not empty, and fields it does not show whole are taken as glibc may read them. */
        ++no;
        if (!line[0] || line[0] == '#') continue;
        if ((n = split_colons(line, fl, 10)) == 1 && cut) { /* a name longer than the line kept: its password unseen */
            char name[NAME_SHOWN + 2], skey[NAME_SHOWN + 32], detail[96];
            short_name(name, fl[0]);
            snprintf(skey, sizeof(skey), "shadow_anormal:%s", name);
            snprintf(detail, sizeof(detail), "linha=%lu: nome longo demais; a senha não foi lida", no);
            inv_add(c->out, "alerta", skey, detail);
            continue;
        }
        if (n < 2 || fl[1][0]) continue;
        if (!cut) ok = shadow_accepted(fl, n);
        else
            for (i = 2, ok = 1; i < n - 1 && i < 9 && ok; ++i) ok = num_or_empty(fl[i]);
        if (!ok) continue;
        snprintf(key, sizeof(key), "senha_vazia:%s", fl[0]);
        inv_add(c->out, "alerta", key, "entra sem senha");
    }
    close_read(c, f, "/etc/shadow", left);
    return 1;
}

/* Groups that amount to root (or read every disk). */
static const char *const admin_groups[] = {"root", "sudo", "wheel", "admin", "docker", "lxd", "libvirt", "disk"};
#define N_ADMIN (sizeof(admin_groups) / sizeof(admin_groups[0]))

typedef struct AdminGid {
    unsigned long gid;
    size_t group; /* index in admin_groups */
} AdminGid;

static size_t admin_index(const char *name) {
    size_t g;
    for (g = 0; g < N_ADMIN && strcmp(name, admin_groups[g]); ++g) {}
    return g;
}

/* The admin group a gid belongs to (the first line that gave it), or N_ADMIN. */
static size_t admin_of_gid(const AdminGid *ag, size_t n, unsigned long gid) {
    size_t i;
    for (i = 0; i < n; ++i)
        if (ag[i].gid == gid) return ag[i].group;
    return N_ADMIN;
}

/* An /etc/group line split as glibc splits it: name, password, gid, members (fl[3], or
 * NULL). 1 when the line is a group for glibc; *gid_ok when its gid was read (in a cut
 * line, only once a ':' followed it). comments: read "#" lines too, as initgroups does. */
static int group_fields(char *line, int r, char *fl[4], unsigned long *gid, int *gid_ok, int comments) {
    int nf, canon, known;
    *gid_ok = 0;
    if (!line[0] || (line[0] == '#' && !comments)) return 0;
    nf = split_colons(line, fl, 4);
    if (nf < 4) fl[3] = NULL;
    known = nf > 3 || (r != LINE_LONG && nf == 3);
    if (r != LINE_LONG && nf < 3) return 0;
    if (!known) return 1;
    *gid_ok = glibc_id(fl[2], gid, &canon);
    return *gid_ok;
}

/* Members of the admin groups: listed members and accounts whose primary group it is.
 * A group grants what its gid grants, whatever the line is called: glibc's initgroups
 * collects every gid a user is listed under, and sudo matches %sudo by getgrnam's gid
 * and by getgrgid's name. So an admin group is any line named like one, or holding the
 * gid of a line named like one; any other name or gid for it is an alert. The names come
 * from the file as getgrnam reads it; the members as initgroups reads it (login, sshd and
 * sudo get those): with getline, comments too, no repeated tail, blanks before each member
 * skipped. Both checked with getent and id on glibc 2.39. */
static void collect_admins(Ctx *c) {
    char line[INV_LINE], key[INV_LINE + 128], name[NAME_SHOWN + 2], why[256], detail[384], gid_text[24];
    unsigned long long left = INV_TEXT_MAX;
    unsigned long no = 0, gid;
    AdminGid *ag = NULL;
    size_t nag = 0, i, u;
    char *fl[4];
    int r, blanks, gid_ok, ch;
    FILE *f = open_shown(c, "/etc/group");
    if (!f) return;
    /* First pass: every gid an admin name has, in file order (the first is getgrnam's). */
    while ((r = read_entry(f, line, sizeof(line), &left, &blanks, 1)) != 0) {
        size_t g;
        if (!group_fields(line, r, fl, &gid, &gid_ok, 0) || !gid_ok || (g = admin_index(fl[0])) == N_ADMIN) continue;
        if (admin_of_gid(ag, nag, gid) != N_ADMIN) continue;
        ag = (AdminGid *)xrealloc(ag, (nag + 1) * sizeof(*ag));
        ag[nag].gid = gid;
        ag[nag++].group = g;
    }
    rewind(f);
    left = INV_TEXT_MAX;
    /* Second pass: the members, as initgroups reads them. */
    while ((r = read_entry(f, line, sizeof(line), &left, &blanks, 0)) != 0) {
        size_t g, by_name;
        char *m, *next;
        int last = feof(f);
        ++no;
        if (!group_fields(line, r, fl, &gid, &gid_ok, 1)) {
            if (r < 0 && line[0] && line[0] != '#')
                note(c->notes, "/etc/group linha %lu: longa demais ou com byte nulo, ignorada", no);
            continue;
        }
        by_name = admin_index(fl[0]);
        g = by_name != N_ADMIN ? by_name : gid_ok ? admin_of_gid(ag, nag, gid) : N_ADMIN;
        if (g == N_ADMIN && gid_ok) continue; /* an ordinary group */
        why[0] = 0;
        if (fl[0][0] == '#') add_why(why, sizeof(why), "comentário para o getgrnam, mas o initgroups lê a linha");
        if (blanks) add_why(why, sizeof(why), "espaço antes do nome");
        if (blanks && last && r == 1)
            add_why(why, sizeof(why), "última linha recuada e sem quebra: o getgrnam a lê com a cauda repetida");
        if (!gid_ok) add_why(why, sizeof(why), "gid não lido; pode ser o de um grupo de admin");
        else if (by_name == N_ADMIN) add_why(why, sizeof(why), "outro nome para o gid de um grupo de admin");
        else {
            for (i = 0; i < nag && ag[i].group != by_name; ++i) {}
            if (i < nag && ag[i].gid != gid) add_why(why, sizeof(why), "outro gid para o nome de um grupo de admin");
        }
        if (r == LINE_LONG) add_why(why, sizeof(why), "linha longa demais; o último membro lido pode estar cortado e fica de fora");
        if (r == LINE_NUL) add_why(why, sizeof(why), "byte nulo na linha");
        if (why[0]) {
            short_name(name, fl[0]);
            if (gid_ok) snprintf(gid_text, sizeof(gid_text), "%lu", gid);
            else snprintf(gid_text, sizeof(gid_text), "?");
            snprintf(key, sizeof(key), "grupo_anormal:%s", name);
            snprintf(detail, sizeof(detail), "gid=%s linha=%lu: %s", gid_text, no, why);
            inv_add(c->out, "alerta", key, detail);
        }
        if (g == N_ADMIN) continue;
        for (m = fl[3]; m && *m; m = next) {
            next = strchr(m, ',');
            if (next) *next++ = 0;
            else if (r == LINE_LONG) break; /* the last member of a cut line may be cut itself */
            while (c_space((unsigned char)*m)) ++m; /* glibc's list parser skips them */
            if (!*m) continue;
            snprintf(key, sizeof(key), "%s:%s", admin_groups[g], m);
            /* glibc keeps what ends the name: "ana " or "ana\r" is not ana, though it looks like her on screen. */
            ch = (unsigned char)m[strlen(m) - 1];
            inv_add(c->out, "admin", key, c_space(ch) || ch < 32 || ch == 127
                                              ? "membro; o nome termina em branco ou byte de controle e não é o nome sem ele"
                                              : "membro");
        }
    }
    close_read(c, f, "/etc/group", left);
    for (i = 0; i < nag; ++i)
        for (u = 0; u < c->nusers; ++u)
            if (c->users[u].gid == ag[i].gid) {
                snprintf(key, sizeof(key), "%s:%s", admin_groups[ag[i].group], c->users[u].name);
                inv_add(c->out, "admin", key, "grupo primário");
            }
    free(ag);
}

/* ---- ports from /proc/net ---- */

typedef struct Sock {
    unsigned long inode;
    char comm[32];
} Sock;

typedef struct SockMap {
    Sock *v;
    size_t n;
} SockMap;

typedef struct Listen {
    char key[96];
    unsigned long uid, inode;
    int loopback;
} Listen;

static int is_digits(const char *s) {
    if (!*s) return 0;
    for (; *s; ++s)
        if (*s < '0' || *s > '9') return 0;
    return 1;
}

static void read_comm(Ctx *c, const char *pid, char *out, size_t cap) {
    char shown[320], *nl;
    FILE *f;
    snprintf(out, cap, "?");
    snprintf(shown, sizeof(shown), "/proc/%s/comm", pid);
    if (!(f = open_shown(c, shown))) return;
    if (fgets(out, (int)cap, f)) {
        nl = strchr(out, '\n');
        if (nl) *nl = 0;
        if (!*out) snprintf(out, cap, "?");
    }
    fclose(f);
}

/* Socket inode -> process names, from /proc/<pid>/fd. Other users' processes need root. */
static SockMap socket_owners(Ctx *c) {
    SockMap m = {NULL, 0};
    char p[INV_PATH], link[128];
    DIR *proc, *fds;
    struct dirent *e, *fe;
    int denied = 0;
    if (snprintf(p, sizeof(p), "%s/proc", c->root) >= (int)sizeof(p) || !confined(c, p, "/proc", 1, NULL) ||
        !(proc = opendir(p)))
        return m;
    while ((e = readdir(proc)) != NULL) {
        char comm[32] = "";
        if (!is_digits(e->d_name)) continue;
        if (snprintf(p, sizeof(p), "%s/proc/%s/fd", c->root, e->d_name) >= (int)sizeof(p)) continue;
        if (!(fds = opendir(p))) {
            if (errno == EACCES) denied = 1;
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
            snprintf(m.v[m.n].comm, sizeof(m.v[m.n].comm), "%s", comm);
            ++m.n;
        }
        closedir(fds);
    }
    closedir(proc);
    /* No count in the text: notes are compared between reports, and the number of processes always varies. */
    if (denied) note(c->notes, "portas: dono de alguns sockets desconhecido (rode como root)");
    return m;
}

static int comm_cmp(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* Every distinct process name holding the socket, sorted and joined by '+' (a socket
 * shared by systemd and sshd, or by a server's workers): the same text whatever
 * order /proc lists the processes in. "?" when none is known. */
static void owners_of(const SockMap *m, unsigned long inode, char *out, size_t cap) {
    const char *names[64];
    size_t i, k, n = 0, len = 0;
    for (i = 0; i < m->n && n < 64; ++i)
        if (m->v[i].inode == inode) {
            for (k = 0; k < n && strcmp(names[k], m->v[i].comm); ++k) {}
            if (k == n) names[n++] = m->v[i].comm;
        }
    if (!n) {
        snprintf(out, cap, "?");
        return;
    }
    qsort(names, n, sizeof(names[0]), comm_cmp);
    out[0] = 0;
    for (k = 0; k < n; ++k) {
        int w = snprintf(out + len, cap - len, "%s%s", k ? "+" : "", names[k]);
        if (w < 0 || (size_t)w >= cap - len) break;
        len += (size_t)w;
    }
}

static int host_little_endian(void) {
    const unsigned one = 1;
    return *(const unsigned char *)&one == 1;
}

/* /proc/net address to text. The kernel prints each 32-bit word of the address as
 * the host reads it, so the byte order of the hex follows the host's endianness. */
static int format_addr(const char *hex, int v6, char *out, size_t cap, int *loopback) {
    unsigned char b[16];
    int words = v6 ? 4 : 1, i, allzero = 1, little = host_little_endian();
    if (strlen(hex) != (size_t)(8 * words)) return 0;
    for (i = 0; i < words; ++i) {
        char w[9];
        unsigned long x;
        int k;
        memcpy(w, hex + 8 * i, 8);
        w[8] = 0;
        x = strtoul(w, NULL, 16);
        for (k = 0; k < 4; ++k) b[4 * i + k] = (unsigned char)(x >> (little ? 8 * k : 8 * (3 - k)));
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
    Listen *ls = NULL;
    size_t t, nls = 0, i;
    int any = 0, r;
    char probe[INV_PATH];
    struct stat st;
    if (!full_path(c, "/proc/net/tcp", probe) || stat(probe, &st)) {
        note(c->notes, "portas: /proc/net indisponível");
        return;
    }
    /* Tables first, owners after: a listener opened during the scan still finds its process. */
    for (t = 0; t < sizeof(tables) / sizeof(tables[0]); ++t) {
        char line[1024];
        unsigned long long left = INV_TEXT_MAX;
        FILE *f = open_shown(c, tables[t].file);
        if (!f) continue;
        any = 1;
        if (read_line(f, line, sizeof(line), &left) == 0) { /* header */
            fclose(f);
            continue;
        }
        while ((r = read_line(f, line, sizeof(line), &left)) != 0) {
            char local[40], remote[40], lhex[8], rhex[8], shex[8], addr[64];
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
            ls = (Listen *)xrealloc(ls, (nls + 1) * sizeof(*ls));
            snprintf(ls[nls].key, sizeof(ls[nls].key), "%s %s:%u", tables[t].proto, addr, lport);
            ls[nls].uid = uid;
            ls[nls].inode = inode;
            ls[nls].loopback = loop;
            ++nls;
        }
        close_read(c, f, tables[t].file, left);
    }
    owners = socket_owners(c);
    for (i = 0; i < nls; ++i) {
        char who[256], detail[384];
        owners_of(&owners, ls[i].inode, who, sizeof(who));
        snprintf(detail, sizeof(detail), "processo=%s uid=%lu exposta=%s", who, ls[i].uid,
                 ls[i].loopback ? "nao" : "sim");
        inv_add(c->out, "porta", ls[i].key, detail);
    }
    free(ls);
    free(owners.v);
    if (!any) note(c->notes, "portas: tabelas de /proc/net ilegíveis");
}

/* ---- filesystem walks ---- */

/* SUID/SGID binaries and world-writable paths (sticky directories like /tmp excepted). */
static void scan_perms(Ctx *c, const char *shown, int depth, dev_t dev) {
    char p[INV_PATH], detail[256], hex[2 * HASH_BYTES + 1];
    struct stat st;
    DIR *d;
    struct dirent *e;
    if (!full_path(c, shown, p)) return;
    if (dev == (dev_t)-1 && !confined(c, p, shown, 0, NULL)) return; /* the top of each walk; below it no link is followed */
    if (lstat(p, &st)) {
        if (errno == EACCES) note(c->notes, "sem acesso: %s", shown);
        return;
    }
    if (S_ISLNK(st.st_mode)) return;
    if (dev != (dev_t)-1 && st.st_dev != dev) {
        if (S_ISDIR(st.st_mode)) note(c->notes, "outro sistema de arquivos, não verificado: %s", shown);
        return;
    }
    if (S_ISREG(st.st_mode)) {
        if (st.st_mode & (S_ISUID | S_ISGID)) {
            mode_t mode = 0;
            int r = hash_file(c, p, hex, &mode);
            /* An unknown hash is not a value to compare: the binary becomes a note (or an alert) instead. */
            if (r != HASH_OK) hash_failed(c, shown, r, mode);
            else {
                snprintf(detail, sizeof(detail), "modo=%04o dono=%lu sha256=%s", (unsigned)(st.st_mode & 07777),
                         (unsigned long)st.st_uid, hex);
                inv_add(c->out, "suid", shown, detail);
            }
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

/* The hash of what a link points to, when it is a regular file inside the system
 * read: a unit enabled by a link changes when its target changes, not the link. */
static void link_target_hash(Ctx *c, const char *p, const char *target, char *out, size_t cap) {
    char path[INV_PATH * 2], resolved[INV_PATH * 2], hex[2 * HASH_BYTES + 1];
    struct stat st;
    mode_t mode;
    out[0] = 0;
    if (target[0] == '/') snprintf(path, sizeof(path), "%s%s", c->root, target);
    else {
        const char *slash = strrchr(p, '/');
        snprintf(path, sizeof(path), "%.*s/%s", slash ? (int)(slash - p) : 0, p, target);
    }
    if (!realpath(path, resolved) || !under_root(c, resolved)) return;
    if (stat(resolved, &st) || !S_ISREG(st.st_mode) || hash_file(c, resolved, hex, &mode) != HASH_OK) return;
    snprintf(out, cap, " sha256=%s", hex);
}

/* Records every file under shown: mode, owner and hash (links: target and the target's hash). */
static void track(Ctx *c, const char *category, const char *shown, int depth, int top) {
    char p[INV_PATH], detail[INV_PATH + 200], hex[2 * HASH_BYTES + 1];
    struct stat st;
    DIR *d;
    struct dirent *e;
    if (!full_path(c, shown, p)) return;
    if (top && !confined(c, p, shown, 0, NULL)) return; /* below the top, directories come from lstat: never links */
    if (lstat(p, &st)) {
        if (errno == EACCES) note(c->notes, "sem acesso: %s", shown);
        return;
    }
    if (S_ISLNK(st.st_mode)) {
        char target[INV_PATH], th[2 * HASH_BYTES + 16];
        ssize_t n = readlink(p, target, sizeof(target) - 1);
        if (n < 0) n = 0;
        target[n] = 0;
        link_target_hash(c, p, target, th, sizeof(th));
        snprintf(detail, sizeof(detail), "-> %s%s", target, th);
        inv_add(c->out, category, shown, detail);
        return;
    }
    if (S_ISREG(st.st_mode)) {
        mode_t mode = 0;
        int r = hash_file(c, p, hex, &mode);
        if (r != HASH_OK) {
            hash_failed(c, shown, r, mode);
            return;
        }
        snprintf(detail, sizeof(detail), "modo=%04o dono=%lu sha256=%s", (unsigned)(st.st_mode & 07777),
                 (unsigned long)st.st_uid, hex);
        inv_add(c->out, category, shown, detail);
        return;
    }
    if (!S_ISDIR(st.st_mode)) { /* a FIFO as .bashrc used to leave the report without a word */
        not_regular(c, shown, st.st_mode);
        return;
    }
    if (depth <= 0) return;
    if (!(d = opendir(p))) {
        if (errno == EACCES) note(c->notes, "sem acesso: %s", shown);
        return;
    }
    while ((e = readdir(d)) != NULL) {
        char child[INV_PATH];
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        if (snprintf(child, sizeof(child), "%s/%s", shown, e->d_name) >= (int)sizeof(child)) continue;
        track(c, category, child, depth - 1, 0);
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
        {"inicio", "/etc/rcS.d", 1}, {"inicio", "/etc/rc0.d", 1}, {"inicio", "/etc/rc1.d", 1},
        {"inicio", "/etc/rc2.d", 1}, {"inicio", "/etc/rc3.d", 1}, {"inicio", "/etc/rc4.d", 1},
        {"inicio", "/etc/rc5.d", 1}, {"inicio", "/etc/rc6.d", 1},
        {"inicio", "/etc/xdg/autostart", 1}, {"inicio", "/etc/profile", 0}, {"inicio", "/etc/profile.d", 1},
        {"inicio", "/etc/bash.bashrc", 0}, {"inicio", "/etc/environment", 0}, {"inicio", "/etc/ld.so.preload", 0},
        /* security configuration */
        {"config", "/etc/ssh/sshd_config", 0}, {"config", "/etc/ssh/sshd_config.d", 1}, {"config", "/etc/sudoers", 0},
        {"config", "/etc/sudoers.d", 1}, {"config", "/etc/pam.d", 1}, {"config", "/etc/hosts", 0},
        {"config", "/etc/ld.so.conf", 0}, {"config", "/etc/ld.so.conf.d", 1},
        {"config", "/etc/nsswitch.conf", 0}}; /* a new NSS module makes accounts /etc/passwd never shows */
    static const struct { const char *path; int depth; } home[] = {
        {"/.bashrc", 0}, {"/.bash_profile", 0}, {"/.bash_login", 0}, {"/.profile", 0}, {"/.zshrc", 0},
        {"/.config/autostart", 1}, {"/.config/systemd/user", 3}, {"/.local/share/systemd/user", 3}};
    size_t i, u, h;
    char p[INV_PATH];
    struct stat st;
    for (i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i)
        track(c, paths[i].category, paths[i].path, paths[i].depth, 1);
    if (full_path(c, "/etc/ld.so.preload", p) && confined(c, p, "/etc/ld.so.preload", 0, NULL) && !lstat(p, &st))
        inv_add(c->out, "alerta", "ld.so.preload", "biblioteca carregada em todo programa; raro em sistema normal");

    for (u = 0; u < c->nusers; ++u) {
        const User *us = &c->users[u];
        int seen = 0;
        if (!(us->login || us->uid == 0) || us->home[0] != '/' || !us->home[1]) continue;
        for (h = 0; h < u && !seen; ++h)
            seen = (c->users[h].login || c->users[h].uid == 0) && !strcmp(c->users[h].home, us->home);
        if (seen) continue;
        c->as_user = 1, c->as_uid = (uid_t)us->uid, c->as_gid = (gid_t)us->gid;
        for (h = 0; h < sizeof(home) / sizeof(home[0]); ++h) {
            snprintf(p, sizeof(p), "%s%s", us->home, home[h].path);
            track(c, "inicio", p, home[h].depth, 1);
        }
        c->as_user = 0;
    }
}

static int key_type(const char *t) {
    return !strncmp(t, "ssh-", 4) || !strncmp(t, "ecdsa-", 6) || !strncmp(t, "sk-", 3);
}

/* One item per authorized key: who can log in without a password, and under which options. */
static void collect_ssh_keys(Ctx *c) {
    static const char *const files[] = {"/.ssh/authorized_keys", "/.ssh/authorized_keys2"};
    size_t u, k;
    for (u = 0; u < c->nusers; ++u) {
        const User *us = &c->users[u];
        if (us->home[0] != '/') continue;
        for (k = 0; k < 2; ++k) {
            char shown[INV_PATH], line[INV_LINE];
            unsigned long long left = INV_TEXT_MAX;
            unsigned long no = 0;
            FILE *f;
            int r;
            if (snprintf(shown, sizeof(shown), "%s%s", strcmp(us->home, "/") ? us->home : "", files[k]) >=
                (int)sizeof(shown))
                continue;
            c->as_user = 1, c->as_uid = (uid_t)us->uid, c->as_gid = (gid_t)us->gid; /* sshd reads it as the user too */
            f = open_shown(c, shown);
            c->as_user = 0;
            if (!f) continue;
            while ((r = read_line(f, line, sizeof(line), &left)) != 0) {
                char *tok[64], *s = line, key[256], hex[2 * HASH_BYTES + 1], comment[256] = "";
                char detail[INV_LINE + INV_PATH + 512], options[INV_LINE] = "";
                int n = 0, i, j;
                size_t ol = 0;
                ++no;
                strip_cr(line);
                if (r < 0) {
                    /* sshd reads lines of any length: a key behind long options must not vanish. */
                    char bad[INV_PATH + 128];
                    snprintf(bad, sizeof(bad), "chave_anormal:%s:%s", us->name, shown);
                    snprintf(line, sizeof(line), "linha=%lu: linha longa demais ou com byte nulo; não lida", no);
                    inv_add(c->out, "alerta", bad, line);
                    continue;
                }
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
                for (j = 0; j < i; ++j) {
                    int w = snprintf(options + ol, sizeof(options) - ol, "%s%s", j ? " " : "", tok[j]);
                    if (w < 0 || (size_t)w >= sizeof(options) - ol) break;
                    ol += (size_t)w;
                }
                sha256_hex(tok[i + 1], strlen(tok[i + 1]), HASH_BYTES, hex);
                if (i + 2 < n) snprintf(comment, sizeof(comment), "%s", tok[i + 2]);
                snprintf(key, sizeof(key), "%s:%s", us->name, hex);
                /* The options go last and whole: command="...", from="..." change what the key can do. */
                snprintf(detail, sizeof(detail), "tipo=%s comentario=%s arquivo=%s opcoes=%s", tok[i],
                         comment[0] ? comment : "-", shown, options[0] ? options : "-");
                inv_add(c->out, "chave_ssh", key, detail);
            }
            close_read(c, f, shown, left);
        }
    }
}

int inventory_collect(const char *root, InvList *out, InvList *notes) {
    static const char *const perm_roots[] = {"/bin", "/sbin", "/usr", "/opt", "/etc"};
    Ctx c;
    size_t i, n;
    int ok;
    memset(&c, 0, sizeof(c));
    snprintf(c.root, sizeof(c.root), "%s", root ? root : "");
    n = strlen(c.root);
    while (n && c.root[n - 1] == '/') c.root[--n] = 0;
    c.out = out;
    c.notes = notes;
    if (c.root[0] && !realpath(c.root, c.root_real)) {
        note(notes, "raiz inexistente: %s", c.root);
        return 0;
    }

    ok = collect_users(&c);
    if (ok) {
        collect_admins(&c);
        collect_ports(&c);
        for (i = 0; i < sizeof(perm_roots) / sizeof(perm_roots[0]); ++i)
            scan_perms(&c, perm_roots[i], 24, (dev_t)-1);
        collect_tracked(&c);
        collect_ssh_keys(&c);
    } else {
        note(notes, "sem /etc/passwd legível: não parece um sistema Linux");
    }

    free(c.users);
    inv_finish(out);
    inv_finish(notes);
    return ok;
}

#endif
