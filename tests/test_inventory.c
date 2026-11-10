/* Unit tests for smolmoe-inventario.
 *
 * Portable: SHA-256 vectors, merging, size bounds, report round trip, and
 * seeded fuzzing of the report reader and of the diff.
 * POSIX: collection from fake system trees, including
 *  - an oracle test: random systems whose expected report is computed here,
 *    independently, and must match exactly (nothing invented, nothing missed);
 *  - fuzzed system files (passwd, group, shadow, /proc/net, authorized_keys):
 *    no crash, every item well formed and traceable to the input;
 *  - regressions for inputs that used to produce items that do not exist.
 *
 * SM_FUZZ_ITERS raises the fuzz iteration counts (default: fast enough for CI). */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#  define _DARWIN_C_SOURCE 1
#  define _FILE_OFFSET_BITS 64
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inventory.h"
#include "sha256.h"

#if !defined(_WIN32)
#  include <dirent.h>
#  include <fcntl.h>
#  include <signal.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  include <time.h>
#  include <unistd.h>
#  if defined(__GLIBC__)
#    include <pwd.h>
#    include <shadow.h>
#  endif
#endif

static int failures;

#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fputc('\n', stderr);                                 \
            ++failures;                                          \
        }                                                        \
    } while (0)

/* ---- helpers ---------------------------------------------------------------- */

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint32_t rnd(uint32_t n) { /* uniform enough in [0, n) */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return n ? (uint32_t)(rng_state % n) : 0;
}

static unsigned iters(unsigned fallback) {
    const char *s = getenv("SM_FUZZ_ITERS");
    unsigned long v = s ? strtoul(s, NULL, 10) : 0;
    return v ? (unsigned)v : fallback;
}

/* Exactly two tabs, no other control byte, within INV_ITEM_MAX - 1, valid UTF-8 lead/continuation structure. */
static int well_formed(const char *s) {
    size_t n = strlen(s), i, tabs = 0;
    if (n > INV_ITEM_MAX - 1) return 0;
    for (i = 0; i < n; ++i) {
        unsigned char c = (unsigned char)s[i];
        if (c == '\t') ++tabs;
        else if (c < 32 || c == 127) return 0;
    }
    return tabs == 2;
}

/* No UTF-8 character cut at the end (a lead byte without all its continuation bytes). */
static int utf8_tail_ok(const char *s) {
    size_t n = strlen(s), i = n, cont = 0, need;
    unsigned char c;
    while (i && ((unsigned char)s[i - 1] & 0xC0) == 0x80 && cont < 4) --i, ++cont;
    if (!i) return 1;
    c = (unsigned char)s[i - 1];
    if (c < 0x80) return cont == 0;
    need = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0;
    return need == cont;
}

static size_t diff_text(const InvList *a, const InvList *b, char *out, size_t cap) {
    size_t n, changes;
    FILE *f = tmpfile();
    if (!f) return (size_t)-1;
    changes = inventory_diff(a, b, f);
    rewind(f);
    n = fread(out, 1, cap - 1, f);
    out[n] = 0;
    fclose(f);
    return changes;
}

static int same_list(const InvList *a, const InvList *b) {
    size_t i;
    if (a->count != b->count) return 0;
    for (i = 0; i < a->count; ++i)
        if (strcmp(a->items[i], b->items[i])) return 0;
    return 1;
}

/* Adds a report line "categoria\tchave\tdetalhe" as an item. */
static void add_line(InvList *l, const char *line) {
    char buf[INV_ITEM_MAX], *t1, *t2;
    snprintf(buf, sizeof(buf), "%s", line);
    t1 = strchr(buf, '\t');
    t2 = t1 ? strchr(t1 + 1, '\t') : NULL;
    if (!t2) {
        fprintf(stderr, "FAIL add_line: [%s]\n", line);
        ++failures;
        return;
    }
    *t1 = *t2 = 0;
    inv_add(l, buf, t1 + 1, t2 + 1);
}

static void write_report(const char *path, const InvList *items) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "FAIL cannot write %s\n", path);
        ++failures;
        return;
    }
    inventory_write(f, items, NULL, "h", "t", "");
    fclose(f);
}

/* ---- portable tests --------------------------------------------------------- */

static void test_sha256(void) {
    char hex[65];
    char *big = (char *)malloc(1000000);
    sha256_hex("", 0, 32, hex);
    CHECK(!strcmp(hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"), "empty: %s", hex);
    sha256_hex("abc", 3, 32, hex);
    CHECK(!strcmp(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "abc: %s", hex);
    sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56, 32, hex);
    CHECK(!strcmp(hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"), "56 bytes: %s", hex);
    if (big) {
        memset(big, 'a', 1000000);
        sha256_hex(big, 1000000, 32, hex);
        CHECK(!strcmp(hex, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"), "million a: %s", hex);
        free(big);
    }
}

static void test_list(void) {
    InvList l;
    inv_init(&l);
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=b");
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=a");
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=b");
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=a");
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=b");
    inv_add(&l, "porta", "tcp 0.0.0.0:8", "x\ty\nz");
    inv_add(&l, "admin", "sudo:ana", "membro");
    inv_finish(&l);
    CHECK(l.count == 3, "count %zu", l.count);
    CHECK(!strcmp(l.items[0], "admin\tsudo:ana\tmembro"), "%s", l.items[0]);
    CHECK(!strcmp(l.items[1], "porta\ttcp 0.0.0.0:8\tx?y?z"), "%s", l.items[1]);
    /* Repeated details are listed once, however many times and in whatever order they come. */
    CHECK(!strcmp(l.items[2], "porta\ttcp 0.0.0.0:80\tprocesso=a, processo=b"), "%s", l.items[2]);
    inv_free(&l);
}

/* No item outgrows a report line: huge fields are cut (on a character boundary) and marked. */
static void test_bounds(void) {
    InvList l, back;
    char err[256], path[] = "inv-test-bounds.txt";
    size_t i, big = 20000;
    char *key = (char *)malloc(big + 1), *detail = (char *)malloc(big + 1);
    if (!key || !detail) return;
    inv_init(&l);
    inv_init(&back);
    for (i = 0; i < big; i += 2) key[i] = (char)0xC3, key[i + 1] = (char)0xA7;      /* "ç" repeated */
    for (i = 0; i < big; i += 3) detail[i] = (char)0xE2, detail[i + 1] = (char)0x82, detail[i + 2] = (char)0xAC; /* "€" */
    key[big] = 0;
    detail[big - big % 3] = 0;
    inv_add(&l, "inicio", key, detail);
    CHECK(well_formed(l.items[0]), "huge item not bounded (%zu bytes)", strlen(l.items[0]));
    CHECK(strstr(l.items[0], " [cortado]") != NULL, "cut not marked");
    {
        /* The key is cut to 4096 bytes without splitting "ç". */
        const char *k = strchr(l.items[0], '\t') + 1;
        size_t kl = (size_t)(strchr(k, '\t') - k);
        CHECK(kl <= 4096 && kl % 2 == 0, "key length %zu", kl);
    }
    {
        char *copy = (char *)malloc(strlen(l.items[0]) + 1);
        if (copy) {
            strcpy(copy, l.items[0]);
            copy[strlen(copy) - strlen(" [cortado]")] = 0;
            CHECK(utf8_tail_ok(copy), "cut split a UTF-8 character");
            free(copy);
        }
    }
    /* 3000 details under one key (e.g. many processes on one port): the merged line stays bounded. */
    for (i = 0; i < 3000; ++i) {
        char d[64];
        snprintf(d, sizeof(d), "processo=p%04zu uid=%zu", i, i);
        inv_add(&l, "porta", "tcp 0.0.0.0:443", d);
    }
    inv_finish(&l);
    CHECK(l.count == 2, "count %zu", l.count);
    for (i = 0; i < l.count; ++i) CHECK(well_formed(l.items[i]), "item %zu: %zu bytes", i, strlen(l.items[i]));
    CHECK(strlen(l.items[1]) > 5 && !strcmp(l.items[1] + strlen(l.items[1]) - 5, ", ..."), "merge overflow not marked");
    /* The program can always read back what it wrote. */
    write_report(path, &l);
    CHECK(inventory_read(path, &back, NULL, err, sizeof(err)), "own report rejected: %s", err);
    CHECK(same_list(&l, &back), "round trip changed the items");
    remove(path);
    inv_free(&l);
    inv_free(&back);
    free(key);
    free(detail);
}

static void test_diff_and_roundtrip(void) {
    InvList a, b, back, notes;
    char text[1024], err[256], path[] = "inv-test-report.txt";
    FILE *f;
    inv_init(&a);
    inv_init(&b);
    inv_init(&back);
    inv_init(&notes);
    inv_add(&a, "porta", "tcp 0.0.0.0:22", "processo=sshd uid=0 exposta=sim");
    inv_add(&a, "usuario", "bob", "uid=1001 shell=/bin/bash");
    inv_add(&a, "config", "/etc/sudoers", "sha256=aaaa");
    inv_add(&b, "porta", "tcp 0.0.0.0:22", "processo=sshd uid=0 exposta=sim");
    inv_add(&b, "porta", "tcp 0.0.0.0:4444", "processo=nc uid=1000 exposta=sim");
    inv_add(&b, "config", "/etc/sudoers", "sha256=bbbb");
    inv_finish(&a);
    inv_finish(&b);
    CHECK(diff_text(&a, &b, text, sizeof(text)) == 3, "changes");
    CHECK(!strcmp(text, "~\tconfig\t/etc/sudoers\tsha256=aaaa -> sha256=bbbb\n"
                        "+\tporta\ttcp 0.0.0.0:4444\tprocesso=nc uid=1000 exposta=sim\n"
                        "-\tusuario\tbob\tuid=1001 shell=/bin/bash\n"),
          "diff:\n%s", text);
    CHECK(diff_text(&a, &a, text, sizeof(text)) == 0 && !text[0], "self diff: %s", text);

    f = fopen(path, "wb");
    CHECK(f != NULL, "cannot write %s", path);
    if (f) {
        inventory_write(f, &b, &notes, "host", "2026-11-09T10:00", "");
        fclose(f);
        CHECK(inventory_read(path, &back, NULL, err, sizeof(err)), "read: %s", err);
        CHECK(same_list(&b, &back), "round trip");
    }
    f = fopen(path, "wb");
    if (f) {
        fputs("# smolmoe-inventario 1\nporta\tsem detalhe\n", f);
        fclose(f);
        inv_free(&back);
        CHECK(!inventory_read(path, &back, NULL, err, sizeof(err)) && strstr(err, ":2:"), "bad line accepted: %s", err);
    }
    f = fopen(path, "wb");
    if (f) {
        fputs("qualquer coisa\n", f);
        fclose(f);
        inv_free(&back);
        CHECK(!inventory_read(path, &back, NULL, err, sizeof(err)), "foreign file accepted");
    }
    remove(path);
    inv_free(&a);
    inv_free(&b);
    inv_free(&back);
    inv_free(&notes);
}

/* Random field text: printable ASCII, UTF-8, separators and control bytes. */
static void random_field(char *out, size_t max) {
    static const char *const pieces[] = {"a", "Z", "0", ":", ",", " ", "/", "\t", "\n", "\r", "\x01", "\x7f",
                                         "#", "\"", "\\", "ç", "€", "->", "~", "+", "-"};
    size_t n = 0, want = rnd((uint32_t)max);
    while (n < want) {
        const char *p = pieces[rnd(sizeof(pieces) / sizeof(pieces[0]))];
        size_t pl = strlen(p);
        if (n + pl >= max) break;
        memcpy(out + n, p, pl);
        n += pl;
    }
    out[n] = 0;
}

/* Mutated report files: never crash; whatever is accepted is well formed and stable. */
static void fuzz_report_reader(void) {
    static const char *const cats[] = {"porta", "usuario", "admin", "inicio"};
    char path[] = "inv-fuzz-report.txt", err[256];
    unsigned it, n = iters(300);
    for (it = 0; it < n; ++it) {
        InvList items, back, again;
        FILE *f;
        char *buf;
        long size;
        size_t k, count = rnd(12);
        inv_init(&items);
        inv_init(&back);
        inv_init(&again);
        for (k = 0; k < count; ++k) {
            char key[64], detail[64];
            random_field(key, sizeof(key));
            random_field(detail, sizeof(detail));
            inv_add(&items, cats[rnd(4)], key, detail);
        }
        inv_finish(&items);
        write_report(path, &items);
        /* Mutate the bytes on disk. */
        f = fopen(path, "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END);
        size = ftell(f);
        rewind(f);
        buf = (char *)malloc((size_t)size + 64);
        if (buf && fread(buf, 1, (size_t)size, f) == (size_t)size) {
            size_t len = (size_t)size, m, muts = rnd(6);
            static const char hot[] = "\t\n\r\0#:a\xff";
            fclose(f);
            for (m = 0; m < muts && len; ++m) {
                size_t at = rnd((uint32_t)len);
                switch (rnd(4)) {
                case 0: buf[at] = hot[rnd(sizeof(hot) - 1)]; break;
                case 1: memmove(buf + at + 1, buf + at, len - at), buf[at] = '\t', ++len; break;
                case 2: memmove(buf + at, buf + at + 1, len - at - 1), --len; break;
                default: len = at; break; /* truncation */
                }
                if (len + 2 > (size_t)size + 60) break;
            }
            f = fopen(path, "wb");
            if (f) {
                fwrite(buf, 1, len, f);
                fclose(f);
            }
            if (inventory_read(path, &back, NULL, err, sizeof(err))) {
                for (k = 0; k < back.count; ++k) CHECK(well_formed(back.items[k]), "accepted: [%s]", back.items[k]);
                write_report(path, &back);
                CHECK(inventory_read(path, &again, NULL, err, sizeof(err)) && same_list(&back, &again),
                      "accepted report not stable: %s", err);
            }
        } else if (f) {
            fclose(f);
        }
        free(buf);
        inv_free(&items);
        inv_free(&back);
        inv_free(&again);
    }
    remove(path);
}

/* Applies a diff to `before` and checks the result is `after`: the diff reports
 * every change and nothing else. Keys come from a small alphabet so that
 * additions, removals and changes all happen. */
static void fuzz_diff(void) {
    unsigned it, n = iters(400);
    for (it = 0; it < n; ++it) {
        InvList a, b, applied;
        char text[65536], *line, *next;
        size_t k, changes, lines = 0;
        inv_init(&a);
        inv_init(&b);
        inv_init(&applied);
        for (k = 0; k < 20; ++k) {
            char key[8], d[8];
            snprintf(key, sizeof(key), "k%u", rnd(15));
            snprintf(d, sizeof(d), "d%u", rnd(3));
            if (rnd(2)) inv_add(&a, rnd(2) ? "porta" : "suid", key, d);
            snprintf(d, sizeof(d), "d%u", rnd(3));
            if (rnd(2)) inv_add(&b, rnd(2) ? "porta" : "suid", key, d);
        }
        inv_finish(&a);
        inv_finish(&b);
        changes = diff_text(&a, &b, text, sizeof(text));
        for (k = 0; k < a.count; ++k) add_line(&applied, a.items[k]);
        for (line = text; *line; line = next) {
            next = strchr(line, '\n');
            if (!next) break;
            *next++ = 0;
            ++lines;
            if (line[0] == '+') {
                add_line(&applied, line + 2);
            } else {
                /* '-' or '~': drop the old item with that key ('~' then adds the new detail). */
                char *body = line + 2, *arrow = strstr(body, " -> ");
                size_t i, kl;
                int found = 0;
                const char *t = strchr(body, '\t');
                t = t ? strchr(t + 1, '\t') : NULL;
                kl = t ? (size_t)(t - body) : strlen(body);
                for (i = 0; i < applied.count; ++i)
                    if (!strncmp(applied.items[i], body, kl) && applied.items[i][kl] == '\t') {
                        free(applied.items[i]);
                        applied.items[i] = applied.items[--applied.count];
                        found = 1;
                        break;
                    }
                CHECK(found, "diff removed a key that was not there: %s", body);
                CHECK(line[0] == '-' || arrow, "changed line without ' -> ': %s", body);
                if (line[0] == '~' && arrow) {
                    char nl[INV_ITEM_MAX];
                    snprintf(nl, sizeof(nl), "%.*s\t%s", (int)kl, body, arrow + 4);
                    add_line(&applied, nl);
                }
            }
        }
        inv_finish(&applied);
        CHECK(lines == changes, "reported %zu changes, printed %zu lines", changes, lines);
        CHECK(same_list(&applied, &b), "diff does not turn before into after (iteration %u)", it);
        CHECK(diff_text(&b, &b, text, sizeof(text)) == 0, "self diff not empty");
        inv_free(&a);
        inv_free(&b);
        inv_free(&applied);
    }
}

/* ---- POSIX: collection ------------------------------------------------------ */

#if !defined(_WIN32)

static char root[256];

static int has(const InvList *l, const char *line) {
    size_t i;
    for (i = 0; i < l->count; ++i)
        if (!strcmp(l->items[i], line)) return 1;
    return 0;
}

static int has_prefix(const InvList *l, const char *prefix) {
    size_t i;
    for (i = 0; i < l->count; ++i)
        if (!strncmp(l->items[i], prefix, strlen(prefix))) return 1;
    return 0;
}

static int any_contains(const InvList *l, const char *needle) {
    size_t i;
    for (i = 0; i < l->count; ++i)
        if (strstr(l->items[i], needle)) return 1;
    return 0;
}

static void put_bytes(const char *rel, const char *data, size_t n, unsigned mode) {
    char p[512];
    FILE *f;
    snprintf(p, sizeof(p), "%s%s", root, rel);
    f = fopen(p, "wb");
    if (!f) {
        fprintf(stderr, "FAIL cannot create %s\n", p);
        ++failures;
        return;
    }
    fwrite(data, 1, n, f);
    fclose(f);
    chmod(p, mode);
}

static void put(const char *rel, const char *text, unsigned mode) { put_bytes(rel, text, strlen(text), mode); }

static void dir(const char *rel, unsigned mode) {
    char p[512];
    snprintf(p, sizeof(p), "%s%s", root, rel);
    mkdir(p, 0755);
    chmod(p, mode);
}

static void link_to(const char *target, const char *rel) {
    char p[512];
    snprintf(p, sizeof(p), "%s%s", root, rel);
    if (symlink(target, p)) {
        fprintf(stderr, "FAIL symlink %s\n", p);
        ++failures;
    }
}

static void new_root(void) {
    snprintf(root, sizeof(root), "inv-root-XXXXXX");
    if (!mkdtemp(root)) {
        fprintf(stderr, "FAIL mkdtemp\n");
        ++failures;
        root[0] = 0;
        return;
    }
    dir("/etc", 0755);
    dir("/proc", 0755);
    dir("/proc/net", 0755);
}

/* Deletes a tree in-process: no `rm -rf` child per test, which made this test
 * slow enough to disturb the timing-sensitive tests `make -j test` runs beside it. */
static void remove_tree(const char *path) {
    struct stat st;
    DIR *d;
    struct dirent *e;
    if (lstat(path, &st)) return;
    if (S_ISDIR(st.st_mode) && (d = opendir(path)) != NULL) {
        while ((e = readdir(d)) != NULL) {
            char child[8192];
            if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
            if (snprintf(child, sizeof(child), "%s/%s", path, e->d_name) < (int)sizeof(child)) remove_tree(child);
        }
        closedir(d);
        rmdir(path);
    } else {
        unlink(path);
    }
}

static void remove_root(void) {
    if (root[0]) remove_tree(root);
}

static void collect(InvList *items, InvList *notes) {
    inv_init(items);
    inv_init(notes);
    inventory_collect(root, items, notes);
}

static void build_fake_system(void) {
    dir("/etc/cron.d", 0755);
    dir("/etc/systemd", 0755);
    dir("/etc/systemd/system", 0755);
    dir("/etc/systemd/system/multi-user.target.wants", 0755);
    dir("/usr", 0755);
    dir("/usr/bin", 0755);
    dir("/usr/share", 0755);
    dir("/usr/share/drop", 0777);
    dir("/home", 0755);
    dir("/home/ana", 0755);
    dir("/home/ana/.ssh", 0700);
    dir("/proc/123", 0755);
    dir("/proc/123/fd", 0755);

    put("/etc/passwd",
        "root:x:0:0:root:/root:/bin/bash\n"
        "daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin\n"
        "ana:x:1000:1000:Ana:/home/ana:/bin/zsh\n"
        "toor:x:0:0::/root:/bin/sh\n",
        0644);
    put("/etc/shadow", "root:$6$abc:1::::::\nana::1::::::\n", 0600);
    put("/etc/group", "root:x:0:\nsudo:x:27:ana\ndocker:x:999:ana,bob\nusers:x:100:ana\n", 0644);
    put("/etc/crontab", "* * * * * root true\n", 0644);
    put("/etc/cron.d/backup", "0 3 * * * root /usr/bin/backup\n", 0644);
    put("/etc/ld.so.preload", "/lib/evil.so\n", 0644);
    put("/etc/sudoers", "root ALL=(ALL) ALL\n", 0440);
    put("/etc/badperm", "x\n", 0666);
    link_to("/lib/systemd/system/ssh.service", "/etc/systemd/system/multi-user.target.wants/ssh.service");
    put("/usr/bin/passwd", "binary", 04755);
    put("/usr/bin/ls", "binary", 0755);
    put("/home/ana/.zshrc", "export X=1\n", 0644);
    put("/home/ana/.ssh/authorized_keys",
        "# comment\n"
        "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIabc ana@laptop\n"
        "from=\"10.0.0.0/8\" ssh-rsa AAAAB3NzaC1yc2EAAAADAQAB\n",
        0600);
    put("/proc/net/tcp",
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        "   0: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 5555 1 0\n"
        "   1: 0100007F:0CEA 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 6666 1 0\n"
        "   2: 0100007F:A000 0100007F:0016 01 00000000:00000000 00:00000000 00000000  1000        0 7777 1 0\n",
        0644);
    put("/proc/net/tcp6",
        "  sl  local_address                         remote_address                        st tx_queue rx_queue tr "
        "tm->when retrnsmt   uid  timeout inode\n"
        "   0: 00000000000000000000000000000000:0016 00000000000000000000000000000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000     0        0 5556 1 0\n"
        "   1: 00000000000000000000000001000000:1F90 00000000000000000000000000000000:0000 0A 00000000:00000000 "
        "00:00000000 00000000  1000        0 8888 1 0\n",
        0644);
    put("/proc/net/udp",
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode ref pointer drops\n"
        "   0: 00000000:0044 00000000:0000 07 00000000:00000000 00:00000000 00000000     0        0 9999 2 0 0\n",
        0644);
    put("/proc/123/comm", "sshd\n", 0644);
    link_to("socket:[5555]", "/proc/123/fd/3");
    link_to("/dev/null", "/proc/123/fd/0");
}

static void test_collect(void) {
    InvList items, notes, again, n2;
    char text[4096];
    size_t i, keys = 0, with_opts = 0;
    new_root();
    if (!root[0]) return;
    build_fake_system();
    collect(&items, &notes);

    CHECK(has(&items, "usuario\troot\tuid=0 shell=/bin/bash"), "root user");
    CHECK(has(&items, "usuario\tana\tuid=1000 shell=/bin/zsh"), "ana user");
    CHECK(!has_prefix(&items, "usuario\tdaemon\t"), "nologin user listed");
    CHECK(has(&items, "alerta\tuid0:toor\toutro usuário com uid 0"), "uid0 alert");
    CHECK(has(&items, "alerta\tsenha_vazia:ana\tentra sem senha"), "empty password alert");
    CHECK(!has_prefix(&items, "alerta\tsenha_vazia:root"), "root has a password");
    CHECK(has(&items, "admin\tsudo:ana\tmembro"), "sudo member");
    CHECK(has(&items, "admin\tdocker:bob\tmembro"), "docker member");
    CHECK(!has_prefix(&items, "admin\tusers:"), "ordinary group listed");

    CHECK(has(&items, "porta\ttcp 0.0.0.0:22\tprocesso=sshd uid=0 exposta=sim"), "ssh port");
    CHECK(has(&items, "porta\ttcp 127.0.0.1:3306\tprocesso=? uid=1000 exposta=nao"), "loopback port");
    CHECK(!has_prefix(&items, "porta\ttcp 127.0.0.1:40960"), "established connection listed");
    CHECK(has(&items, "porta\ttcp6 [::]:22\tprocesso=? uid=0 exposta=sim"), "v6 any");
    CHECK(has(&items, "porta\ttcp6 [::1]:8080\tprocesso=? uid=1000 exposta=nao"), "v6 loopback");
    CHECK(has(&items, "porta\tudp 0.0.0.0:68\tprocesso=? uid=0 exposta=sim"), "udp");

    CHECK(has_prefix(&items, "suid\t/usr/bin/passwd\tmodo=4755 "), "suid binary");
    CHECK(!has_prefix(&items, "suid\t/usr/bin/ls"), "plain binary listed as suid");
    CHECK(has(&items, "gravavel\t/usr/share/drop\tpasta gravável por todos, sem sticky bit"), "writable dir");
    CHECK(has(&items, "gravavel\t/etc/badperm\tarquivo gravável por todos"), "writable file");

    CHECK(has_prefix(&items, "inicio\t/etc/crontab\tmodo=0644 dono="), "crontab");
    CHECK(has_prefix(&items, "inicio\t/etc/cron.d/backup\tmodo=0644 "), "cron.d");
    CHECK(has(&items, "inicio\t/etc/systemd/system/multi-user.target.wants/ssh.service\t-> "
                      "/lib/systemd/system/ssh.service"),
          "systemd link");
    CHECK(has_prefix(&items, "inicio\t/home/ana/.zshrc\tmodo=0644 "), "home rc");
    CHECK(has_prefix(&items, "alerta\tld.so.preload\t"), "preload alert");
    CHECK(has_prefix(&items, "config\t/etc/sudoers\tmodo=0440 "), "sudoers");

    for (i = 0; i < items.count; ++i)
        if (!strncmp(items.items[i], "chave_ssh\tana:", 14)) {
            ++keys;
            with_opts += strstr(items.items[i], "tipo=ssh-rsa comentario=- arquivo=/home/ana/.ssh/authorized_keys opcoes=from=\"10.0.0.0/8\"") != NULL;
        }
    CHECK(keys == 2 && with_opts == 1, "keys %zu with options %zu", keys, with_opts);

    /* A second run over the same tree gives the same report. */
    collect(&again, &n2);
    CHECK(diff_text(&items, &again, text, sizeof(text)) == 0, "not deterministic:\n%s", text);
    inv_free(&again);
    inv_free(&n2);

    /* A new cron job shows up as one addition, and only that. */
    put("/etc/cron.d/miner", "* * * * * root /tmp/x\n", 0644);
    collect(&again, &n2);
    CHECK(diff_text(&items, &again, text, sizeof(text)) == 1 && !strncmp(text, "+\tinicio\t/etc/cron.d/miner\t", 27),
          "cron addition:\n%s", text);

    inv_free(&items);
    inv_free(&notes);
    inv_free(&again);
    inv_free(&n2);
    remove_root();
}

/* Inputs that used to make the inventory report things that do not exist. */
static void test_no_invented_items(void) {
    InvList items, notes;
    char *line;
    size_t pre = INV_ITEM_MAX - 1, n;
    const char *tail = "evil:x:0:0::/root:/bin/sh\n";
    new_root();
    if (!root[0]) return;
    dir("/home", 0755);
    dir("/home/ana", 0755);
    dir("/home/ana/.ssh", 0700);

    /* 1. A passwd line longer than the line buffer: its tail looks like a root account.
     *    Read in pieces, the tail became a fake "evil" with uid 0. */
    line = (char *)malloc(pre + strlen(tail) + 1024);
    if (!line) return;
    n = (size_t)sprintf(line, "root:x:0:0:root:/root:/bin/bash\nana:x:1000:1000:");
    memset(line + n, 'A', pre - strlen("ana:x:1000:1000:"));
    n += pre - strlen("ana:x:1000:1000:");
    n += (size_t)sprintf(line + n, "%s", tail);
    /* 2. A NUL inside a line: the rest of the line must not be read as a line. */
    memcpy(line + n, "nul\0x:0:0::/:/bin/sh\n", 21);
    n += 21;
    /* 3. A uid that is empty or not a number: strtoul reads it as 0 (root), glibc rejects the line.
     *    "-0" is different: glibc reads it as uid 0, so it must show up (test_hidden_accounts). */
    n += (size_t)sprintf(line + n, "blank:x::0::/:/bin/sh\nword:x:abc:0::/:/bin/sh\nneg:x:-0:0::/:/bin/sh\n");
    /* 4. A name longer than the account buffer: cutting it could merge two accounts, so it
     *    becomes an alert, never a user item. */
    n += (size_t)sprintf(line + n, "%s:x:1001:0::/:/bin/sh\n",
                         "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    n += (size_t)sprintf(line + n, "ana2:x:1002:1002::/home/ana:/bin/bash\n");
    put_bytes("/etc/passwd", line, n, 0644);
    free(line);

    /* 5. Quoted options holding a fake key: only the key after them is real. */
    put("/home/ana/.ssh/authorized_keys",
        "command=\"echo ssh-rsa FAKEKEY\",no-pty ssh-ed25519 REALKEY ana@pc\n"
        "command=\"unterminated ssh-rsa FAKE2\n"
        "environment=\"A=\\\"ssh-dss FAKE3\\\"\" ssh-ed25519 REALKEY2 x\n",
        0600);
    /* 6. /proc/net lines that are not listening sockets, or not parseable, give no port. */
    put("/proc/net/tcp",
        "header\n"
        "   0: 0100007F:1F90 0100007F:0016 01 00000000:00000000 00:00000000 00000000 0 0 1 1 0\n"
        "   1: GARBAGE:0016 00000000:0000 0A x x x 0 0 1\n"
        "   2: 0100007:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 1 1 0\n",
        0644);

    collect(&items, &notes);
    CHECK(!any_contains(&items, "evil"), "tail of a long passwd line became an account");
    CHECK(!any_contains(&items, "nul") && !any_contains(&items, "x:0:0"), "text after a NUL became an account");
    CHECK(!any_contains(&items, "blank") && !any_contains(&items, "word"), "malformed uid read as root");
    CHECK(has_prefix(&items, "alerta\tuid0:neg\t"), "uid -0, root for glibc, hidden");
    CHECK(!has_prefix(&items, "usuario\tbbbb"), "oversized name cut and kept as a user");
    CHECK(has_prefix(&items, "alerta\tconta_anormal:bbbb"), "oversized name dropped without an alert");
    CHECK(has(&items, "usuario\troot\tuid=0 shell=/bin/bash"), "valid account before the bad lines lost");
    CHECK(has(&items, "usuario\tana2\tuid=1002 shell=/bin/bash"), "valid account after the bad lines lost");
    CHECK(!has_prefix(&items, "alerta\tuid0:evil") && !has_prefix(&items, "alerta\tuid0:nul") &&
              !has_prefix(&items, "alerta\tuid0:x") && !has_prefix(&items, "alerta\tuid0:blank") &&
              !has_prefix(&items, "alerta\tuid0:word"),
          "invented uid 0 alert");
    /* The fake keys may appear inside the recorded options text, never as a key of their own. */
    CHECK(!any_contains(&items, "tipo=ssh-rsa") && !any_contains(&items, "tipo=ssh-dss"), "fake key from quoted options");
    {
        size_t i, keys = 0;
        for (i = 0; i < items.count; ++i) keys += !strncmp(items.items[i], "chave_ssh\t", 10);
        CHECK(keys == 2, "expected the 2 real keys, got %zu", keys);
    }
    CHECK(!has_prefix(&items, "porta\t"), "port invented from a non-listening or broken line");
    {
        size_t i, noted = 0;
        for (i = 0; i < notes.count; ++i) noted += strstr(notes.items[i], "/etc/passwd") != NULL;
        CHECK(noted >= 1, "skipped passwd lines not reported as notes");
    }
    inv_free(&items);
    inv_free(&notes);
    remove_root();
}

/* Oracle: random but valid systems, expected report computed here. */
static void test_oracle(void) {
    static const char *const shells[] = {"/bin/bash", "/bin/sh", "/usr/sbin/nologin", "/bin/false", "/bin/zsh", ""};
    static const char *const groups[] = {"sudo", "docker", "users", "wheel", "audio"};
    unsigned it, n = iters(100);
    new_root(); /* one tree; every iteration rewrites all of its files */
    if (!root[0]) return;
    for (it = 0; it < n; ++it) {
        InvList items, notes, want;
        char passwd[8192], shadow[4096], group[4096], tcp[16384];
        size_t pl = 0, sl = 0, gl = 0, tl = 0, u, g, nusers = 1 + rnd(12), nports = rnd(10);
        char names[16][16];
        unsigned used_ports[16];
        inv_init(&want);
        for (u = 0; u < nusers; ++u) {
            const char *shell = shells[rnd(6)];
            unsigned long uid = rnd(4) ? 1000 + u : (rnd(2) ? 0 : 1 + u);
            int login;
            snprintf(names[u], sizeof(names[u]), "%s%c%zu", u == 0 ? "root" : "u", (char)('a' + rnd(26)), u);
            if (u == 0) snprintf(names[u], sizeof(names[u]), "root"), uid = 0;
            pl += (size_t)snprintf(passwd + pl, sizeof(passwd) - pl, "%s:x:%lu:%lu::/home/%s:%s\n", names[u], uid,
                                   uid, names[u], shell);
            login = !(strstr(shell, "nologin") || strstr(shell, "/false"));
            if (login || uid == 0) {
                char d[320];
                snprintf(d, sizeof(d), "uid=%lu shell=%s", uid, shell[0] ? shell : "/bin/sh");
                inv_add(&want, "usuario", names[u], d);
            }
            if (uid == 0 && strcmp(names[u], "root")) {
                char k[320];
                snprintf(k, sizeof(k), "uid0:%s", names[u]);
                inv_add(&want, "alerta", k, "outro usuário com uid 0");
            }
            {
                int empty = !rnd(5);
                sl += (size_t)snprintf(shadow + sl, sizeof(shadow) - sl, "%s:%s:1::::::\n", names[u],
                                       empty ? "" : "$6$x");
                if (empty) {
                    char k[320];
                    snprintf(k, sizeof(k), "senha_vazia:%s", names[u]);
                    inv_add(&want, "alerta", k, "entra sem senha");
                }
            }
        }
        for (g = 0; g < 5; ++g) {
            int admin = g != 2 && g != 4;
            size_t members = rnd(4), m;
            gl += (size_t)snprintf(group + gl, sizeof(group) - gl, "%s:x:%zu:", groups[g], 100 + g);
            for (m = 0; m < members; ++m) {
                const char *who = names[rnd((uint32_t)nusers)];
                gl += (size_t)snprintf(group + gl, sizeof(group) - gl, "%s%s", m ? "," : "", who);
                if (admin) {
                    char k[48];
                    snprintf(k, sizeof(k), "%s:%s", groups[g], who);
                    inv_add(&want, "admin", k, "membro");
                }
            }
            gl += (size_t)snprintf(group + gl, sizeof(group) - gl, "\n");
        }
        tl += (size_t)snprintf(tcp + tl, sizeof(tcp) - tl, "  sl  local_address rem_address   st ...\n");
        for (u = 0; u < nports; ++u) {
            unsigned a = rnd(4) ? 0x0100007Fu : rnd(0xFFFFFFFFu), port, state = rnd(3) ? 0x0A : 0x01, uid = rnd(2000);
            size_t q;
            int dup;
            do {
                port = 1 + rnd(65535);
                for (dup = 0, q = 0; q < u; ++q) dup |= used_ports[q] == port;
            } while (dup);
            used_ports[u] = port;
            tl += (size_t)snprintf(tcp + tl, sizeof(tcp) - tl,
                                   "%4zu: %08X:%04X 00000000:0000 %02X 00000000:00000000 00:00000000 00000000 %5u "
                                   "0 %u 1 0\n",
                                   u, a, port, state, uid, 1000 + (unsigned)u);
            if (state == 0x0A) {
                char k[64], d[96];
                unsigned b0 = a & 255, b1 = a >> 8 & 255, b2 = a >> 16 & 255, b3 = a >> 24;
                snprintf(k, sizeof(k), "tcp %u.%u.%u.%u:%u", b0, b1, b2, b3, port);
                snprintf(d, sizeof(d), "processo=? uid=%u exposta=%s", uid, b0 == 127 ? "nao" : "sim");
                inv_add(&want, "porta", k, d);
            }
        }
        put_bytes("/etc/passwd", passwd, pl, 0644);
        put_bytes("/etc/shadow", shadow, sl, 0600);
        put_bytes("/etc/group", group, gl, 0644);
        put_bytes("/proc/net/tcp", tcp, tl, 0644);
        inv_finish(&want);
        collect(&items, &notes);
        if (!same_list(&items, &want)) {
            char text[16384];
            diff_text(&want, &items, text, sizeof(text));
            CHECK(0, "oracle mismatch (iteration %u), expected -> got:\n%s", it, text);
            it = n; /* one detailed failure is enough */
        }
        inv_free(&items);
        inv_free(&notes);
        inv_free(&want);
    }
    remove_root();
}

/* The input as the report would show it: control bytes (newline aside) become '?',
 * NULs too, so a search sees the whole file. */
static void shown_as(char *buf, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i)
        if (((unsigned char)buf[i] < 32 && buf[i] != '\n') || buf[i] == 127) buf[i] = '?';
    buf[n] = 0;
}

static int contains_ci(const char *hay, const char *needle) {
    size_t i, j, hn = strlen(hay), nn = strlen(needle);
    for (i = 0; i + nn <= hn; ++i) {
        for (j = 0; j < nn; ++j) {
            char a = hay[i + j], b = needle[j];
            if (a >= 'a' && a <= 'z') a = (char)(a - 32);
            if (b >= 'a' && b <= 'z') b = (char)(b - 32);
            if (a != b) break;
        }
        if (j == nn) return 1;
    }
    return 0;
}

/* Mutates a template: byte swaps with separators, insertions, deletions, a long run, line duplication. */
static size_t mutate(const char *src, char *out, size_t cap) {
    static const char hot[] = ":\t\n\r\"\\ #,0aA\x00\xff=";
    size_t n = strlen(src), m, muts = 1 + rnd(8);
    if (n >= cap) n = cap - 1;
    memcpy(out, src, n);
    for (m = 0; m < muts; ++m) {
        size_t at = n ? rnd((uint32_t)n) : 0;
        switch (rnd(6)) {
        case 0: if (n) out[at] = hot[rnd(sizeof(hot) - 1)]; break;
        case 1: if (n + 1 < cap) memmove(out + at + 1, out + at, n - at), out[at] = hot[rnd(sizeof(hot) - 1)], ++n; break;
        case 2: if (n) memmove(out + at, out + at + 1, n - at - 1), --n; break;
        case 3: { /* a run long enough to cross the line buffer */
            size_t run = rnd(2) ? 9000 : 100 + rnd(400);
            if (n + run < cap) {
                memmove(out + at + run, out + at, n - at);
                /* blanks too: glibc skips them before a line, however many there are */
                memset(out + at, rnd(3) ? 'A' + (int)rnd(3) : ' ', run);
                n += run;
            }
            break;
        }
        case 4: { /* duplicate the bytes from at to the next newline */
            const char *nl = (const char *)memchr(out + at, '\n', n - at);
            size_t len = nl ? (size_t)(nl - (out + at)) + 1 : n - at;
            if (n + len < cap) memmove(out + at + len, out + at, n - at), n += len;
            break;
        }
        default: n = at; break;
        }
    }
    return n;
}

#define INV_LINE_KEPT 8191 /* bytes of a system file line inventory.c keeps (its INV_LINE - 1) */

#if defined(__GLIBC__)
/* After an oracle failure: the start of each line of path holding name, byte by byte. */
static void dump_lines(const char *path, const char *name) {
    static char buf[32768];
    size_t n, i, start = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = 0;
    for (i = 0; i <= n; ++i)
        if (i == n || buf[i] == '\n') {
            if (i > start && (!*name || memmem(buf + start, i - start, name, strlen(name)))) {
                size_t k, colons = 0;
                fprintf(stderr, "  line (%zu bytes, ':' at", i - start);
                for (k = start; k < i && colons < 12; ++k)
                    if (buf[k] == ':') fprintf(stderr, " %zu", k - start), ++colons;
                fputs("):", stderr);
                for (k = start; k < i && k < start + 64; ++k) fprintf(stderr, " %02x", (unsigned char)buf[k]);
                fprintf(stderr, "%s\n", i - start > 64 ? " ..." : "");
                if (i - start > 64) {
                    fputs("  ends:", stderr);
                    for (k = i - 48; k < i; ++k) fprintf(stderr, " %02x", (unsigned char)buf[k]);
                    fputc('\n', stderr);
                }
            }
            start = i + 1;
        }
}
#endif

/* The text as glibc reads it, for traceability. An indented line keeps its last k bytes twice,
 * k being the blanks before it, when no newline is copied along to cut it: the last line with
 * no newline, or a line with a NUL (glibc's __nss_readline; see read_entry in inventory.c).
 * Each such line, as glibc reads it, is appended after a newline: the original text stays. */
static size_t glibc_view(char *buf, size_t n, size_t cap) {
    size_t start = 0, end, out = n;
    while (start < n) {
        size_t k = 0, len;
        const char *nul;
        for (end = start; end < n && buf[end] != '\n'; ++end) {}
        while (start + k < end && (buf[start + k] == ' ' || (buf[start + k] >= '\t' && buf[start + k] <= '\r')))
            ++k;
        nul = (const char *)memchr(buf + start, 0, end - start);
        len = (nul ? (size_t)(nul - buf) : end) - start; /* the line up to a NUL, blanks included */
        if (k && len > k && (nul || end == n) && out + len + 1 < cap) {
            buf[out++] = '\n';
            memcpy(buf + out, buf + start + k, len - k);           /* the line without its blanks */
            memcpy(buf + out + len - k, buf + start + len - k, k); /* and its last k bytes again */
            out += len;
        }
        start = end + 1;
    }
    return out;
}

/* Whether a detail says uid 0, or an unknown uid (a line cut before its gid). */
static int uid0_in(const char *detail) {
    const char *p;
    for (p = detail; (p = strstr(p, "uid=")) != NULL; p += 4)
        if ((p[4] == '0' && (p[5] < '0' || p[5] > '9')) || p[4] == '?') return 1;
    return 0;
}

/* Whether the report names this uid 0 account: as a user, as a uid 0 alert, or as an
 * abnormal account (the name cut to the key's room) whose detail says uid 0. */
static int root_visible(const InvList *l, const char *name) {
    char clean[INV_ITEM_MAX];
    size_t i, n = 0, cut;
    for (; *name && n + 1 < sizeof(clean); ++name) clean[n++] = (unsigned char)*name < 32 || *name == 127 ? '?' : *name;
    clean[n] = 0;
    cut = n < 48 ? n : 48;
    for (i = 0; i < l->count; ++i) {
        const char *s = l->items[i], *k = strchr(s, '\t') + 1, *d = strchr(k, '\t') + 1;
        size_t kl = (size_t)(d - 1 - k);
        if (!strncmp(s, "usuario\t", 8) && kl == n && !strncmp(k, clean, n) && uid0_in(d)) return 1;
        if (strncmp(s, "alerta\t", 7)) continue;
        if (kl == 5 + n && !strncmp(k, "uid0:", 5) && !strncmp(k + 5, clean, n)) return 1;
        if (kl >= 14 + cut && !strncmp(k, "conta_anormal:", 14) && !strncmp(k + 14, clean, cut) &&
            (n > 48 || kl == 14 + n) && uid0_in(d))
            return 1;
    }
    return 0;
}

static void fuzz_collection(void) {
    static const char *const allowed[] = {"usuario", "alerta", "admin", "porta", "suid",
                                          "gravavel", "inicio", "config", "chave_ssh"};
    const char *passwd_t = "root:x:0:0:root:/root:/bin/bash\nana:x:1000:1000:Ana:/home/ana:/bin/zsh\n"
                           "svc:x:5:5::/:/usr/sbin/nologin\n";
    const char *shadow_t = "root:$6$a:1::::::\nana::1::::::\n";
    const char *group_t = "sudo:x:27:ana,root\ndocker:x:999:ana\nusers:x:100:ana\n";
    const char *tcp_t = "  sl  local_address rem_address   st\n"
                        "   0: 00000000:0016 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 55 1 0\n"
                        "   1: 0100007F:0CEA 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 66 1 0\n";
    const char *keys_t = "ssh-ed25519 AAAAC3Nz ana@pc\nfrom=\"1.2.3.4\",command=\"x y\" ssh-rsa AAAAB3 b\n";
    /* mutated up to 32 KiB; twice that leaves room for glibc_view */
    static char passwd[65536], shadow[65536], group[65536], tcp[32768], keys[32768];
    unsigned it, n = iters(150);
    new_root();
    if (!root[0]) return;
    dir("/home", 0755);
    dir("/home/ana", 0755);
    dir("/home/ana/.ssh", 0700);
    for (it = 0; it < n; ++it) {
        InvList items, notes;
        size_t pl = mutate(passwd_t, passwd, 32768), sl = mutate(shadow_t, shadow, 32768);
        size_t gl = mutate(group_t, group, 32768), tl = mutate(tcp_t, tcp, sizeof(tcp));
        size_t kl = mutate(keys_t, keys, sizeof(keys)), i, a;
        put_bytes("/etc/passwd", passwd, pl, 0644);
        put_bytes("/etc/shadow", shadow, sl, 0600);
        put_bytes("/etc/group", group, gl, 0644);
        put_bytes("/proc/net/tcp", tcp, tl, 0644);
        put_bytes("/home/ana/.ssh/authorized_keys", keys, kl, 0600);
        pl = glibc_view(passwd, pl, sizeof(passwd));
        sl = glibc_view(shadow, sl, sizeof(shadow));
        gl = glibc_view(group, gl, sizeof(group));
        shown_as(passwd, pl);
        shown_as(shadow, sl);
        shown_as(group, gl);
        shown_as(tcp, tl);
        shown_as(keys, kl);
        collect(&items, &notes);
        for (i = 0; i < items.count; ++i) {
            const char *s = items.items[i], *key = strchr(s, '\t') + 1;
            char cat[16], k[INV_ITEM_MAX];
            size_t cl = (size_t)(key - 1 - s), klen;
            CHECK(well_formed(s), "malformed item: [%s]", s);
            if (cl >= sizeof(cat)) continue;
            memcpy(cat, s, cl);
            cat[cl] = 0;
            klen = (size_t)(strchr(key, '\t') - key);
            memcpy(k, key, klen);
            k[klen] = 0;
            for (a = 0; a < sizeof(allowed) / sizeof(allowed[0]) && strcmp(cat, allowed[a]); ++a) {}
            CHECK(a < sizeof(allowed) / sizeof(allowed[0]), "unknown category %s", cat);
            /* Traceability: what the item names is in the input. */
            if (!strcmp(cat, "usuario")) {
                char probe[INV_ITEM_MAX + 2];
                snprintf(probe, sizeof(probe), "%s:", k);
                CHECK(strstr(passwd, probe) != NULL, "user %s not in passwd", k);
            } else if (!strcmp(cat, "alerta") && !strncmp(k, "uid0:", 5)) {
                CHECK(strstr(passwd, k + 5) != NULL, "uid0 alert for %s not in passwd", k + 5);
            } else if (!strcmp(cat, "alerta") && !strncmp(k, "conta_anormal:", 14)) {
                CHECK(strstr(passwd, k + 14) != NULL, "abnormal account %s not in passwd (iteration %u)", k + 14, it);
                if (!strstr(passwd, k + 14)) {
                    size_t q, from = pl > 80 ? pl - 80 : 0;
                    fprintf(stderr, "  key %zu bytes:", strlen(k + 14));
                    for (q = 14; k[q] && q < 14 + 70; ++q) fprintf(stderr, " %02x", (unsigned char)k[q]);
                    fprintf(stderr, "\n  passwd (%zu bytes) ends:", pl);
                    for (q = from; q < pl; ++q) fprintf(stderr, " %02x", (unsigned char)passwd[q]);
                    fputc('\n', stderr);
                }
            } else if (!strcmp(cat, "alerta") && !strncmp(k, "senha_vazia:", 12)) {
                CHECK(strstr(shadow, k + 12) != NULL, "empty password for %s not in shadow", k + 12);
            } else if (!strcmp(cat, "admin")) {
                const char *member = strchr(k, ':');
                CHECK(member && strstr(group, member + 1), "admin %s not in group", k);
            } else if (!strcmp(cat, "porta")) {
                const char *colon = strrchr(k, ':');
                char hex[16];
                snprintf(hex, sizeof(hex), "%X", (unsigned)strtoul(colon ? colon + 1 : "0", NULL, 10));
                CHECK(colon && contains_ci(tcp, hex), "port %s not in /proc/net/tcp", k);
            } else if (!strcmp(cat, "chave_ssh")) {
                const char *type = strstr(s, "tipo=");
                char t[64] = "";
                if (type) sscanf(type + 5, "%63s", t);
                CHECK(t[0] && strstr(keys, t), "key type %s not in authorized_keys", t);
            }
        }
#if defined(__GLIBC__)
        {
            /* glibc's own reader is the oracle: every uid 0 account it returns is in the report.
             * "+" and "-" lines are nss_compat directives that nss_files lookups skip. */
            char p[512];
            FILE *pf;
            struct passwd *pw;
            snprintf(p, sizeof(p), "%s/etc/passwd", root);
            if ((pf = fopen(p, "rb")) != NULL) {
                while ((pw = fgetpwent(pf)) != NULL)
                    if (pw->pw_uid == 0 && pw->pw_name[0] != '+' && pw->pw_name[0] != '-')
                        CHECK(root_visible(&items, pw->pw_name), "uid 0 account [%s] hidden (iteration %u)",
                              pw->pw_name, it);
                fclose(pf);
            }
            /* And every empty password glibc reads from the shadow file is an alert. */
            snprintf(p, sizeof(p), "%s/etc/shadow", root);
            if ((pf = fopen(p, "rb")) != NULL) {
                struct spwd *sp;
                while ((sp = fgetspent(pf)) != NULL) {
                    char want[600], cut[80];
                    size_t q;
                    if (sp->sp_pwdp[0] || sp->sp_namp[0] == '+' || sp->sp_namp[0] == '-') continue;
                    /* By name, or, for a name longer than the line kept, as an abnormal shadow line. */
                    snprintf(want, sizeof(want), "alerta\tsenha_vazia:%.500s", sp->sp_namp);
                    snprintf(cut, sizeof(cut), "alerta\tshadow_anormal:%.48s", sp->sp_namp);
                    for (q = 7; want[q]; ++q)
                        if ((unsigned char)want[q] < 32 || want[q] == 127) want[q] = '?';
                    for (q = 7; cut[q]; ++q)
                        if ((unsigned char)cut[q] < 32 || cut[q] == 127) cut[q] = '?';
                    if (has_prefix(&items, want) || (strlen(sp->sp_namp) >= INV_LINE_KEPT && has_prefix(&items, cut)))
                        continue;
                    CHECK(0, "empty password of [%s] hidden (iteration %u)", sp->sp_namp, it);
                    dump_lines(p, sp->sp_namp);
                }
                fclose(pf);
            }
        }
#endif
        inv_free(&items);
        inv_free(&notes);
    }
    remove_root();
}

/* Paths and link targets near the limits still give readable report lines. */
static void test_long_paths(void) {
    InvList items, notes, back;
    char target[4000], err[256], path[] = "inv-test-long.txt", name[200];
    size_t i;
    new_root();
    if (!root[0]) return;
    put("/etc/passwd", "root:x:0:0:root:/root:/bin/bash\n", 0644);
    dir("/etc/cron.d", 0755);
    memset(target, 'T', sizeof(target) - 1);
    target[0] = '/';
    target[sizeof(target) - 1] = 0;
    memset(name, 'n', sizeof(name) - 1);
    name[sizeof(name) - 1] = 0;
    {
        /* As long a target as the system takes: Linux allows 4095 bytes, macOS 1023. */
        char rel[300], p[1024];
        snprintf(rel, sizeof(rel), "/etc/cron.d/%s", name);
        snprintf(p, sizeof(p), "%s%s", root, rel);
        if (symlink(target, p)) {
            target[1000] = 0;
            link_to(target, rel);
        }
    }
    /* 600 sockets on the same port (SO_REUSEPORT): one merged, bounded item. */
    {
        static char tcp[600 * 120 + 64];
        size_t tl = (size_t)sprintf(tcp, "header\n");
        for (i = 0; i < 600; ++i)
            tl += (size_t)sprintf(tcp + tl,
                                  "%4zu: 00000000:01BB 00000000:0000 0A 00000000:00000000 00:00000000 00000000 %5zu 0 %zu 1 0\n",
                                  i, 1000 + i, 50000 + i);
        put_bytes("/proc/net/tcp", tcp, tl, 0644);
    }
    collect(&items, &notes);
    inv_init(&back);
    for (i = 0; i < items.count; ++i) CHECK(well_formed(items.items[i]), "item %zu too long or malformed", i);
    CHECK(has_prefix(&items, "porta\ttcp 0.0.0.0:443\t"), "merged port missing");
    write_report(path, &items);
    CHECK(inventory_read(path, &back, NULL, err, sizeof(err)) && same_list(&items, &back), "long report unreadable: %s", err);
    remove(path);
    inv_free(&items);
    inv_free(&notes);
    inv_free(&back);
    remove_root();
}

static const char *item_detail(const InvList *l, const char *prefix) {
    size_t i;
    for (i = 0; i < l->count; ++i)
        if (!strncmp(l->items[i], prefix, strlen(prefix))) return l->items[i] + strlen(prefix);
    return NULL;
}

/* Fixes from the PR review, one check each. */
static void test_review_fixes(void) {
    InvList a, na, b, nb, back;
    char outside[512], cwd[400], text[8192], err[256], path[] = "inv-test-meta.txt";
    const char *d;
    FILE *f;
    size_t i;

    /* A root that does not exist, or has no /etc/passwd, is an error: no empty baseline from a typo. */
    inv_init(&a);
    inv_init(&na);
    CHECK(!inventory_collect("inv-root-does-not-exist", &a, &na) && a.count == 0, "missing root accepted");
    inv_free(&a);
    inv_free(&na);

    new_root();
    if (!root[0]) return;
    CHECK(!inventory_collect(root, &a, &na), "root without /etc/passwd accepted");
    inv_free(&a);
    inv_free(&na);

    /* Symlinks in the image must not lead to the machine doing the scan. */
    if (!getcwd(cwd, sizeof(cwd))) cwd[0] = 0;
    snprintf(outside, sizeof(outside), "%s/inv-outside-XXXXXX", cwd);
    if (mkdtemp(outside)) {
        char p[700], linkp[700];
        FILE *o;
        snprintf(p, sizeof(p), "%s/passwd", outside);
        if ((o = fopen(p, "wb"))) fputs("hostuser:x:0:0::/:/bin/sh\n", o), fclose(o);
        snprintf(p, sizeof(p), "%s/ana", outside);
        mkdir(p, 0755);
        snprintf(p, sizeof(p), "%s/ana/.ssh", outside);
        mkdir(p, 0700);
        snprintf(p, sizeof(p), "%s/ana/.ssh/authorized_keys", outside);
        if ((o = fopen(p, "wb"))) fputs("ssh-ed25519 HOSTKEY host@outside\n", o), fclose(o);
        snprintf(p, sizeof(p), "%s/passwd", outside);
        snprintf(linkp, sizeof(linkp), "%s/etc/passwd", root);
        CHECK(!symlink(p, linkp), "symlink passwd");
        collect(&a, &na);
        CHECK(!any_contains(&a, "hostuser"), "--raiz followed /etc/passwd out of the image");
        inv_free(&a);
        inv_free(&na);
        unlink(linkp);
        put("/etc/passwd", "root:x:0:0:root:/root:/bin/bash\nana:x:1000:27::/home/ana:/bin/bash\n", 0644);
        put("/etc/group", "sudo:x:27:\nusers:x:100:\n", 0644);
        snprintf(linkp, sizeof(linkp), "%s/home", root);
        CHECK(!symlink(outside, linkp), "symlink home");
        collect(&a, &na);
        CHECK(!any_contains(&a, "HOSTKEY") && !has_prefix(&a, "chave_ssh\t"), "--raiz followed /home out of the image");
        {
            size_t noted = 0;
            for (i = 0; i < na.count; ++i) noted += strstr(na.items[i], "fora da raiz") != NULL;
            CHECK(noted > 0, "escape not noted");
        }
        /* Primary group: ana's gid is sudo's, though sudo lists no members. */
        CHECK(has(&a, "admin\tsudo:ana\tgrupo primário"), "admin by primary group missing");
        inv_free(&a);
        inv_free(&na);
        unlink(linkp);
        remove_tree(outside);
    }

    /* Changes that used to leave the report identical. */
    dir("/home", 0755);
    dir("/home/ana", 0755);
    dir("/home/ana/.ssh", 0700);
    dir("/lib", 0755);
    dir("/etc/systemd", 0755);
    dir("/etc/systemd/system", 0755);
    dir("/etc/systemd/system/multi-user.target.wants", 0755);
    put("/lib/x.service", "[Service]\nExecStart=/bin/true\n", 0644);
    link_to("/lib/x.service", "/etc/systemd/system/multi-user.target.wants/x.service");
    put("/etc/crontab", "* * * * * root true\n", 0644);
    put("/home/ana/.ssh/authorized_keys", "from=\"10.0.0.1\" ssh-ed25519 KEY ana@pc\n", 0600);
    collect(&a, &na);
    d = item_detail(&a, "inicio\t/etc/crontab\t");
    CHECK(d && strstr(d, "sha256=") && strlen(strstr(d, "sha256=") + 7) == 64, "hash is not the full SHA-256: %s", d ? d : "-");
    d = item_detail(&a, "inicio\t/etc/systemd/system/multi-user.target.wants/x.service\t");
    CHECK(d && strstr(d, "-> /lib/x.service sha256="), "link target not hashed: %s", d ? d : "-");
    put("/lib/x.service", "[Service]\nExecStart=/tmp/evil\n", 0644); /* the unit, not the link */
    {
        char p[600];
        snprintf(p, sizeof(p), "%s/etc/crontab", root);
        chmod(p, 0666); /* mode only, same content */
    }
    put("/home/ana/.ssh/authorized_keys", "from=\"0.0.0.0/0\" ssh-ed25519 KEY ana@pc\n", 0600); /* options only */
    collect(&b, &nb);
    diff_text(&a, &b, text, sizeof(text));
    CHECK(strstr(text, "~\tinicio\t/etc/systemd/system/multi-user.target.wants/x.service\t") != NULL,
          "unit change behind a link not seen:\n%s", text);
    CHECK(strstr(text, "~\tinicio\t/etc/crontab\tmodo=0644") != NULL, "mode change not seen:\n%s", text);
    CHECK(strstr(text, "from=\"0.0.0.0/0\"") != NULL && strstr(text, "~\tchave_ssh\t") != NULL,
          "ssh option change not seen:\n%s", text);
    inv_free(&a);
    inv_free(&na);
    inv_free(&b);
    inv_free(&nb);

    /* Shared sockets: every owner, in a fixed order. */
    dir("/proc/net", 0755);
    dir("/proc/10", 0755);
    dir("/proc/10/fd", 0755);
    dir("/proc/20", 0755);
    dir("/proc/20/fd", 0755);
    put("/proc/10/comm", "zeta\n", 0644);
    put("/proc/20/comm", "alfa\n", 0644);
    link_to("socket:[777]", "/proc/10/fd/3");
    link_to("socket:[777]", "/proc/20/fd/4");
    put("/proc/net/tcp",
        "  sl  local_address rem_address   st\n"
        "   0: 00000000:0050 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 777 1 0\n",
        0644);
    collect(&a, &na);
    CHECK(has(&a, "porta\ttcp 0.0.0.0:80\tprocesso=alfa+zeta uid=0 exposta=sim"), "shared socket owners");
    inv_free(&a);
    inv_free(&na);
    remove_root();

    /* A report cut after a whole line, and metadata with a newline. */
    inv_init(&a);
    inv_init(&back);
    inv_add(&a, "porta", "k1", "d");
    inv_add(&a, "porta", "k2", "d");
    inv_finish(&a);
    f = fopen(path, "wb");
    if (f) {
        inventory_write(f, &a, NULL, "host\nporta\tfake\tx", "t", "/mnt/a\nb");
        fclose(f);
        CHECK(inventory_read(path, &back, NULL, err, sizeof(err)) && same_list(&a, &back),
              "metadata broke the report: %s", err);
    }
    f = fopen(path, "wb");
    if (f) {
        fputs("# smolmoe-inventario 1\n# itens: 2\nporta\tk1\td\n", f);
        fclose(f);
        inv_free(&back);
        CHECK(!inventory_read(path, &back, NULL, err, sizeof(err)) && strstr(err, "incompleto"),
              "truncated report accepted: %s", err);
    }
    remove(path);
    inv_free(&a);
    inv_free(&back);
}

/* Accounts that glibc accepts, read the way glibc reads them (nss_files: blanks and
 * comments skipped, uid through strtou32, missing fields empty, the line ending at a
 * NUL): none may be missing from the report. Odd spellings are alerts with the line. */
static void test_hidden_accounts(void) {
    static const char *const roots[] = {"neg", "plus", "sp", "zeros", "short", "lead", "nulroot", "bighome", "evil"};
    static const char nul_line[] = "nulroot:x:0:0::/:/bin/sh\0tail\n";
    static char text[32768];
    char longname[71], home[601], gecos[9001];
    const char *d;
    InvList items, notes;
    size_t i, n = 0;
    new_root();
    if (!root[0]) return;
    memset(longname, 'n', 70);
    longname[70] = 0;
    home[0] = '/';
    memset(home + 1, 'h', 599);
    home[600] = 0;
    memset(gecos, 'G', 9000);
    gecos[9000] = 0;
    n += (size_t)sprintf(text + n, "root:x:0:0:root:/root:/bin/bash\n");                                   /* 1 */
    n += (size_t)sprintf(text + n, "neg:x:-0:0::/root:/bin/bash\nplus:x:+0:0::/root:/bin/bash\n");        /* 2, 3 */
    n += (size_t)sprintf(text + n, "sp:x: 0:0::/root:/bin/bash\nzeros:x:00000000000:0::/root:/bin/bash\n"); /* 4, 5 */
    n += (size_t)sprintf(text + n, "minus1:x:-1:0::/home/m:/bin/bash\nshort:x:0:0\n");                    /* 6, 7 */
    n += (size_t)sprintf(text + n, "  # hidden:x:0:0::/:/bin/sh\n \tlead:x:0:0::/:/bin/sh\n");            /* 8, 9 */
    memcpy(text + n, nul_line, sizeof(nul_line) - 1);                                                     /* 10 */
    n += sizeof(nul_line) - 1;
    n += (size_t)sprintf(text + n, "%s:x:0:0::/:/bin/sh\n", longname);                                    /* 11 */
    n += (size_t)sprintf(text + n, "bighome:x:0:0::%s:/bin/sh\n", home);                                  /* 12 */
    n += (size_t)sprintf(text + n, "evil:x:0:0:%s:/root:/bin/bash\n", gecos);                             /* 13 */
    n += (size_t)sprintf(text + n, "after:x:1003:1003::/home/after:/bin/bash\n");                         /* 14 */
    memset(text + n, ' ', 9000);                                                                          /* 15 */
    n += 9000;
    n += (size_t)sprintf(text + n, "blank9k:x:0:0::/root:/bin/bash\n");
    put_bytes("/etc/passwd", text, n, 0644);
    collect(&items, &notes);

    for (i = 0; i < sizeof(roots) / sizeof(roots[0]); ++i)
        CHECK(root_visible(&items, roots[i]), "uid 0 account %s hidden", roots[i]);
    CHECK(root_visible(&items, "blank9k"), "uid 0 account after 9000 blanks hidden");
    CHECK(root_visible(&items, longname), "uid 0 account with a 70-byte name hidden");
    d = item_detail(&items, "alerta\tconta_anormal:neg\t");
    CHECK(d && strstr(d, "linha=2"), "uid -0 not flagged with its line: %s", d ? d : "-");
    CHECK(has_prefix(&items, "alerta\tconta_anormal:plus\t") && has_prefix(&items, "alerta\tconta_anormal:sp\t") &&
              has_prefix(&items, "alerta\tconta_anormal:zeros\t"),
          "uid 0 written as +0, \" 0\" or with leading zeros not flagged");
    /* strtoul wraps -1 and strtou32 saturates it: glibc's uid is 4294967295, not root. */
    CHECK(has(&items, "usuario\tminus1\tuid=4294967295 shell=/bin/bash"), "uid -1 not read as glibc does");
    d = item_detail(&items, "alerta\tconta_anormal:minus1\t");
    CHECK(d && strstr(d, "uid=4294967295"), "uid -1 not flagged: %s", d ? d : "-");
    CHECK(has_prefix(&items, "alerta\tuid0:short\t"), "account without its last fields hidden");
    CHECK(!any_contains(&items, "hidden"), "a comment after blanks read as an account");
    CHECK(has_prefix(&items, "alerta\tuid0:lead\t"), "blanks before a name: glibc reads the name without them");
    CHECK(has_prefix(&items, "alerta\tuid0:nulroot\t"), "account before a NUL hidden");
    CHECK(!has_prefix(&items, "usuario\tbighome") && !has_prefix(&items, "usuario\tevil") &&
              !has_prefix(&items, "usuario\tnnnn"),
          "an account with a field cut kept as a user");
    d = item_detail(&items, "alerta\tconta_anormal:nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn");
    CHECK(d && strstr(d, "cortado"), "cut name not marked: %s", d ? d : "-");
    d = item_detail(&items, "alerta\tconta_anormal:evil\t");
    CHECK(d && strstr(d, "linha=13"), "long line not flagged with its line: %s", d ? d : "-");
    CHECK(has(&items, "usuario\troot\tuid=0 shell=/bin/bash") && !has_prefix(&items, "alerta\tconta_anormal:root"),
          "plain root changed or flagged");
    CHECK(has(&items, "usuario\tafter\tuid=1003 shell=/bin/bash"), "account after the odd lines lost");
    inv_free(&items);
    inv_free(&notes);
    remove_root();
}

/* Items and notes in one list, as --comparar compares them. */
static void with_notes(const InvList *items, const InvList *notes, InvList *out) {
    size_t i;
    inv_init(out);
    for (i = 0; i < items->count; ++i) add_line(out, items->items[i]);
    for (i = 0; i < notes->count; ++i) inv_add(out, "nota", notes->items[i], "");
    inv_finish(out);
}

/* --comparar must see a second bad line next to one the base already had: both used
 * to get the same note, and equal notes merge into one. */
static void test_second_bad_account(void) {
    static const char *const base = "root:x:0:0:root:/root:/bin/bash\nblank:x::0::/:/bin/sh\n";
    InvList a, na, b, nb, ma, mb;
    char text[8192], passwd[512];
    new_root();
    if (!root[0]) return;
    put("/etc/passwd", base, 0644);
    collect(&a, &na);
    with_notes(&a, &na, &ma);

    snprintf(passwd, sizeof(passwd), "%sneg:x:-0:0::/root:/bin/bash\n", base);
    put("/etc/passwd", passwd, 0644);
    collect(&b, &nb);
    with_notes(&b, &nb, &mb);
    CHECK(diff_text(&ma, &mb, text, sizeof(text)) > 0 && strstr(text, "neg"), "second account with uid 0 unseen:\n%s",
          text);
    inv_free(&b);
    inv_free(&nb);
    inv_free(&mb);

    snprintf(passwd, sizeof(passwd), "%sword:x:abc:0::/:/bin/sh\n", base);
    put("/etc/passwd", passwd, 0644);
    collect(&b, &nb);
    with_notes(&b, &nb, &mb);
    CHECK(diff_text(&ma, &mb, text, sizeof(text)) > 0, "second line that is not an account unseen");
    inv_free(&a);
    inv_free(&na);
    inv_free(&ma);
    inv_free(&b);
    inv_free(&nb);
    inv_free(&mb);
    remove_root();
}

/* /etc/shadow and /etc/group read the way glibc reads them too. A group grants what its gid
 * grants, whatever the line is called: initgroups collects every gid a user is listed under,
 * and sudo matches %sudo by getgrnam's gid and by getgrgid's name. */
static void test_shadow_and_groups(void) {
    /* The last line, indented and without a newline, is read by glibc with its last byte twice:
     * "\tzed::1::" becomes "zed::1:::", which it accepts (see read_entry). */
    static const char rejected[] = "eve:\0$6$x:19000:0:99999:7:::\ncarol::19000\nten::1:2:3:4:5:6:7:8\n"
                                   "nine::1:2:3:4:5:6:\n nul2::1::\0:::\n\tzed::1::";
    static char text[16384];
    const char *d;
    InvList items, notes;
    size_t n = 0;
    new_root();
    if (!root[0]) return;
    put("/etc/passwd",
        "root:x:0:0:root:/root:/bin/bash\nana:x:1000:1000::/home/ana:/bin/bash\nbia:x:1002:1002::/home/bia:/bin/bash\n"
        "carl:x:1003:1003::/home/carl:/bin/bash\nbob:x:1004:27::/home/bob:/bin/bash\n"
        "dan:x:1005:999::/home/dan:/bin/bash\n",
        0644);
    n += (size_t)sprintf(text + n, "root:$6$x:19000:0:99999:7:::\n");
    memset(text + n, ' ', 9000);
    n += 9000;
    n += (size_t)sprintf(text + n, "bob::19000:0:99999:7:::\n");
    /* Lines glibc's shadow parser rejects (it needs the fields up to sp_max): no password to report. */
    memcpy(text + n, rejected, sizeof(rejected) - 1);
    n += sizeof(rejected) - 1;
    put_bytes("/etc/shadow", text, n, 0600);
    put("/etc/group", "root:x:0:\nsudo:x:27:\nsudo2:x:27:ana\n sudo:x:27:bia\nsudo:x:999:carl\nusers:x:100:ana\n", 0644);
    collect(&items, &notes);

    CHECK(has(&items, "alerta\tsenha_vazia:bob\tentra sem senha"), "empty password after 9000 blanks hidden");
    CHECK(!has_prefix(&items, "alerta\tsenha_vazia:eve") && !has_prefix(&items, "alerta\tsenha_vazia:carol") &&
              !has_prefix(&items, "alerta\tsenha_vazia:ten"),
          "empty password invented from a shadow line glibc rejects");
    CHECK(has(&items, "alerta\tsenha_vazia:nine\tentra sem senha"), "9-field shadow line with an empty flag hidden");
    /* An indented line with a NUL also keeps its last byte twice: " nul2::1::" is read as "nul2::1:::". */
    CHECK(has(&items, "alerta\tsenha_vazia:nul2\tentra sem senha"), "indented shadow line with a NUL hidden");
    CHECK(has(&items, "alerta\tsenha_vazia:zed\tentra sem senha"), "indented last shadow line without a newline hidden");
    CHECK(has(&items, "admin\tsudo:ana\tmembro"), "member of another name with sudo's gid hidden");
    CHECK(has(&items, "admin\tsudo:bia\tmembro"), "member of \" sudo\" (glibc skips the blank) hidden");
    CHECK(has(&items, "admin\tsudo:carl\tmembro"), "member of a second \"sudo\" line hidden");
    CHECK(has(&items, "admin\tsudo:bob\tgrupo primário"), "primary group sudo by gid hidden");
    CHECK(has(&items, "admin\tsudo:dan\tgrupo primário"), "primary gid of a second \"sudo\" line hidden");
    d = item_detail(&items, "alerta\tgrupo_anormal:sudo2\t");
    CHECK(d && strstr(d, "gid=27"), "another name for sudo's gid not flagged: %s", d ? d : "-");
    d = item_detail(&items, "alerta\tgrupo_anormal:sudo\t");
    CHECK(d && strstr(d, "gid=999"), "a second sudo with another gid not flagged: %s", d ? d : "-");
    CHECK(!has_prefix(&items, "admin\tusers:"), "ordinary group listed");
    inv_free(&items);
    inv_free(&notes);
    remove_root();
}

/* Group membership as glibc's initgroups reads it, for that is what login, sshd and sudo
 * get: every line read whole with getline (comments too, with no repeated tail), blanks
 * skipped before each member. Checked with `id` against getent on glibc 2.39. */
static void test_initgroups_view(void) {
    InvList items, notes;
    new_root();
    if (!root[0]) return;
    put("/etc/passwd",
        "root:x:0:0:root:/root:/bin/bash\ncmt:x:1001:1001::/home/cmt:/bin/bash\nsp2:x:1002:1002::/home/sp2:/bin/bash\n"
        "tail:x:1003:1003::/home/tail:/bin/bash\n",
        0644);
    put("/etc/group", "root:x:0:\nsudo:x:27:\n#c:x:27:cmt\nlist:x:27:aa, sp2\nend:x:27:ana \ncrlf:x:27:carl\r\n"
                      "  last:x:27:tail",
        0644);
    collect(&items, &notes);
    /* "ana " and "carl\r" are not ana and carl (id gives them no 27): the report must not look as if they were. */
    CHECK(has_prefix(&items, "admin\tsudo:ana \tmembro; ") && !has(&items, "admin\tsudo:ana \tmembro"),
          "member with a trailing blank shown as a plain member");
    CHECK(has_prefix(&items, "admin\tsudo:carl?\tmembro; "), "member with a trailing CR shown as a plain member");
    CHECK(has(&items, "admin\tsudo:cmt\tmembro"), "member of a comment line (initgroups reads it) hidden");
    CHECK(has_prefix(&items, "alerta\tgrupo_anormal:#c\t"), "comment line granting sudo not flagged");
    CHECK(has(&items, "admin\tsudo:sp2\tmembro"), "member after \", \" hidden");
    CHECK(has(&items, "admin\tsudo:tail\tmembro") && !has_prefix(&items, "admin\tsudo:tailil"),
          "indented last line without a newline: initgroups's member replaced by getgrnam's");
    inv_free(&items);
    inv_free(&notes);
    remove_root();
}

/* Collects root in a child process and reads its report back: a scan that blocks fails
 * the case after 5 s instead of hanging this binary. 0 on a timeout or a crash. */
static int collect_in_child(InvList *items, InvList *notes) {
    static const char path[] = "inv-child-report.txt";
    struct timespec tick = {0, 10 * 1000 * 1000};
    char err[256];
    int status = 0, waited = 0, ok;
    pid_t pid;
    inv_init(items);
    inv_init(notes);
    fflush(NULL);
    if ((pid = fork()) < 0) return 0;
    if (pid == 0) {
        InvList ci, cn;
        FILE *f;
        inv_init(&ci);
        inv_init(&cn);
        inventory_collect(root, &ci, &cn);
        if (!(f = fopen(path, "wb"))) _exit(3);
        inventory_write(f, &ci, &cn, "h", "t", root);
        _exit(fclose(f) ? 3 : 0);
    }
    while (waitpid(pid, &status, WNOHANG) == 0) {
        if (++waited > 500) {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            remove(path);
            return 0;
        }
        nanosleep(&tick, NULL);
    }
    ok = WIFEXITED(status) && !WEXITSTATUS(status) && inventory_read(path, items, notes, err, sizeof(err));
    remove(path);
    return ok;
}

/* A process that writes NUL bytes into the FIFO at rel, without end, once a reader opens it. */
static pid_t nul_writer(const char *rel) {
    char p[512];
    pid_t pid;
    snprintf(p, sizeof(p), "%s%s", root, rel);
    fflush(NULL);
    if ((pid = fork()) == 0) {
        static const char zeros[4096];
        int fd = open(p, O_WRONLY);
        if (fd < 0) _exit(1);
        for (;;)
            if (write(fd, zeros, sizeof(zeros)) < 0) _exit(0);
    }
    return pid;
}

static void stop(pid_t pid) {
    int status;
    if (pid <= 0) return;
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
}

/* A sparse file of the given size: no disk used, every byte reads as NUL. 0 (and no
 * file) where the file system has no holes: a 64 MiB probe that takes real blocks
 * stops before 4 GiB of zeros get written. */
static int sparse(const char *rel, long long bytes) {
    char p[512];
    struct stat st;
    int fd, ok;
    snprintf(p, sizeof(p), "%s%s", root, rel);
    if ((fd = open(p, O_WRONLY | O_CREAT | O_TRUNC, 0644)) < 0) return 0;
    ok = !ftruncate(fd, (off_t)64 << 20) && !fstat(fd, &st) && (long long)st.st_blocks * 512 < (1LL << 20) &&
         !ftruncate(fd, (off_t)bytes);
    close(fd);
    if (!ok) unlink(p);
    return ok;
}

static void fifo(const char *rel) {
    char p[512];
    snprintf(p, sizeof(p), "%s%s", root, rel);
    if (mkfifo(p, 0644)) {
        fprintf(stderr, "FAIL mkfifo %s\n", p);
        ++failures;
    }
}

static void unlink_rel(const char *rel) {
    char p[512];
    snprintf(p, sizeof(p), "%s%s", root, rel);
    unlink(p);
}

/* --comparar base.txt <(ssh host smolmoe-inventario): a report can come through a pipe. */
static void test_report_from_pipe(void) {
    static const char path[] = "inv-test-pipe";
    InvList a, back;
    char err[256];
    pid_t writer;
    inv_init(&a);
    inv_init(&back);
    inv_add(&a, "porta", "tcp 0.0.0.0:22", "processo=sshd");
    inv_finish(&a);
    remove(path);
    if (mkfifo(path, 0600)) {
        fprintf(stderr, "FAIL mkfifo %s\n", path);
        ++failures;
        return;
    }
    fflush(NULL);
    if ((writer = fork()) == 0) {
        FILE *f = fopen(path, "wb");
        if (f) inventory_write(f, &a, NULL, "h", "t", "/"), fclose(f);
        _exit(0);
    }
    CHECK(inventory_read(path, &back, NULL, err, sizeof(err)) && same_list(&a, &back), "report through a pipe: %s",
          err);
    stop(writer);
    remove(path);
    inv_free(&a);
    inv_free(&back);
}

/* What the user owns in their home can be a FIFO, a device or a huge sparse file: the
 * scan, run as root, must neither block nor read without end, and must say so. */
static void test_special_files(void) {
    static const char keys[] = "/root/.ssh/authorized_keys", rc[] = "/home/ana/.bashrc",
                      ana_keys[] = "/home/ana/.ssh/authorized_keys";
    const long long huge = 4LL << 30;
    InvList items, notes;
    pid_t writer;
    int ok;
    new_root();
    if (!root[0]) return;
    dir("/root", 0700);
    dir("/root/.ssh", 0700);
    dir("/home", 0755);
    dir("/home/ana", 0755);
    dir("/home/ana/.ssh", 0700);
    dir("/etc/ssh", 0755);
    dir("/etc/ssh/keys", 0755);
    put("/etc/passwd", "root:x:0:0:root:/root:/bin/bash\nana:x:1000:1000::/home/ana:/bin/bash\n", 0644);

    /* A FIFO with no writer used to block in open. */
    fifo(keys);
    ok = collect_in_child(&items, &notes);
    CHECK(ok, "FIFO without a writer as authorized_keys: the scan blocked");
    CHECK(has_prefix(&items, "alerta\tnao_regular:/root/.ssh/authorized_keys\t"), "FIFO without a writer not flagged");
    inv_free(&items);
    inv_free(&notes);

    /* A FIFO fed NULs used to loop in read_line. */
    writer = nul_writer(keys);
    ok = collect_in_child(&items, &notes);
    stop(writer);
    CHECK(ok, "FIFO fed NULs as authorized_keys: the scan never ended");
    CHECK(has_prefix(&items, "alerta\tnao_regular:/root/.ssh/authorized_keys\t"), "FIFO fed NULs not flagged");
    inv_free(&items);
    inv_free(&notes);

    /* Opening is already an act (/dev/watchdog arms on open): what is not a regular file is never
     * opened. A writer's open of a FIFO returns only once a reader opens it, so it marks that. */
    {
        char marker[512];
        pid_t watcher;
        struct timespec tick = {0, 20 * 1000 * 1000};
        int k;
        snprintf(marker, sizeof(marker), "%s/opened", root);
        fflush(NULL);
        if ((watcher = fork()) == 0) {
            char p[512];
            int fd;
            snprintf(p, sizeof(p), "%s%s", root, keys);
            if ((fd = open(p, O_WRONLY)) >= 0) close(open(marker, O_WRONLY | O_CREAT, 0644));
            _exit(0);
        }
        ok = collect_in_child(&items, &notes);
        for (k = 0; k < 10 && access(marker, F_OK); ++k) nanosleep(&tick, NULL);
        CHECK(ok && access(marker, F_OK), "a FIFO in place of authorized_keys was opened");
        stop(watcher);
        inv_free(&items);
        inv_free(&notes);
    }
    unlink_rel(keys);

    /* A relative link inside the image is a legitimate authorized_keys (regression guard, passes before). */
    put("/etc/ssh/keys/root", "ssh-ed25519 ROOTKEY admin@pc\n", 0600);
    link_to("../../etc/ssh/keys/root", keys);
    ok = collect_in_child(&items, &notes);
    CHECK(ok && has_prefix(&items, "chave_ssh\troot:"), "authorized_keys behind a relative link inside the image lost");
    inv_free(&items);
    inv_free(&notes);

    /* A FIFO in place of a tracked file used to vanish from the report. */
    fifo(rc);
    ok = collect_in_child(&items, &notes);
    CHECK(ok && has_prefix(&items, "alerta\tnao_regular:/home/ana/.bashrc\t"), "FIFO as .bashrc not flagged");
    inv_free(&items);
    inv_free(&notes);
    unlink_rel(rc);

    /* Sparse files of 4 GiB: regular, so only a byte limit stops the read. */
    if (!sparse(ana_keys, huge)) {
        fprintf(stderr, "test-inventory: no sparse files here, size limits not tested\n");
    } else {
        ok = collect_in_child(&items, &notes);
        CHECK(ok, "4 GiB authorized_keys: the scan did not end in 5 s");
        CHECK(any_contains(&notes, ana_keys), "4 GiB authorized_keys read without a note");
        inv_free(&items);
        inv_free(&notes);
        unlink_rel(ana_keys);
        if (sparse(rc, huge)) {
            ok = collect_in_child(&items, &notes);
            CHECK(ok, "4 GiB .bashrc: the hash did not end in 5 s");
            CHECK(any_contains(&notes, rc) && !has_prefix(&items, "inicio\t/home/ana/.bashrc\t"),
                  "4 GiB .bashrc not noted");
            inv_free(&items);
            inv_free(&notes);
        }
    }
    remove_root();
}

#endif

int main(void) {
    test_sha256();
    test_list();
    test_bounds();
    test_diff_and_roundtrip();
    fuzz_report_reader();
    fuzz_diff();
#if !defined(_WIN32)
    test_collect();
    test_no_invented_items();
    test_oracle();
    fuzz_collection();
    test_long_paths();
    test_review_fixes();
    test_hidden_accounts();
    test_second_bad_account();
    test_shadow_and_groups();
    test_initgroups_view();
    test_report_from_pipe();
    test_special_files();
#endif
    if (failures) {
        fprintf(stderr, "test-inventory: %d failure(s)\n", failures);
        return 1;
    }
    puts("test-inventory: ok");
    return 0;
}
