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
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inventory.h"
#include "sha256.h"

#if !defined(_WIN32)
#  include <dirent.h>
#  include <sys/stat.h>
#  include <unistd.h>
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
    CHECK(inventory_read(path, &back, err, sizeof(err)), "own report rejected: %s", err);
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
        CHECK(inventory_read(path, &back, err, sizeof(err)), "read: %s", err);
        CHECK(same_list(&b, &back), "round trip");
    }
    f = fopen(path, "wb");
    if (f) {
        fputs("# smolmoe-inventario 1\nporta\tsem detalhe\n", f);
        fclose(f);
        inv_free(&back);
        CHECK(!inventory_read(path, &back, err, sizeof(err)) && strstr(err, ":2:"), "bad line accepted: %s", err);
    }
    f = fopen(path, "wb");
    if (f) {
        fputs("qualquer coisa\n", f);
        fclose(f);
        inv_free(&back);
        CHECK(!inventory_read(path, &back, err, sizeof(err)), "foreign file accepted");
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
            if (inventory_read(path, &back, err, sizeof(err))) {
                for (k = 0; k < back.count; ++k) CHECK(well_formed(back.items[k]), "accepted: [%s]", back.items[k]);
                write_report(path, &back);
                CHECK(inventory_read(path, &again, err, sizeof(err)) && same_list(&back, &again),
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

    CHECK(has_prefix(&items, "inicio\t/etc/crontab\tsha256="), "crontab");
    CHECK(has_prefix(&items, "inicio\t/etc/cron.d/backup\tsha256="), "cron.d");
    CHECK(has(&items, "inicio\t/etc/systemd/system/multi-user.target.wants/ssh.service\t-> "
                      "/lib/systemd/system/ssh.service"),
          "systemd link");
    CHECK(has_prefix(&items, "inicio\t/home/ana/.zshrc\tsha256="), "home rc");
    CHECK(has_prefix(&items, "alerta\tld.so.preload\t"), "preload alert");
    CHECK(has_prefix(&items, "config\t/etc/sudoers\tsha256="), "sudoers");

    for (i = 0; i < items.count; ++i)
        if (!strncmp(items.items[i], "chave_ssh\tana:", 14)) {
            ++keys;
            with_opts += strstr(items.items[i], "tipo=ssh-rsa comentario=- opcoes=sim") != NULL;
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
    /* 3. A uid that is empty or not a number: strtoul reads it as 0 (root). */
    n += (size_t)sprintf(line + n, "blank:x::0::/:/bin/sh\nword:x:abc:0::/:/bin/sh\nneg:x:-0:0::/:/bin/sh\n");
    /* 4. A name longer than the account buffer: cutting it could merge two accounts. */
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
    CHECK(!any_contains(&items, "blank") && !any_contains(&items, "word") && !any_contains(&items, "neg"),
          "malformed uid read as root");
    CHECK(!any_contains(&items, "bbbb"), "oversized name cut and kept");
    CHECK(has(&items, "usuario\troot\tuid=0 shell=/bin/bash"), "valid account before the bad lines lost");
    CHECK(has(&items, "usuario\tana2\tuid=1002 shell=/bin/bash"), "valid account after the bad lines lost");
    CHECK(!has_prefix(&items, "alerta\tuid0:"), "invented uid 0 alert");
    CHECK(!any_contains(&items, "FAKE"), "fake key from quoted options");
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
                memset(out + at, 'A' + (int)rnd(3), run);
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
    static char passwd[32768], shadow[32768], group[32768], tcp[32768], keys[32768];
    unsigned it, n = iters(150);
    new_root();
    if (!root[0]) return;
    dir("/home", 0755);
    dir("/home/ana", 0755);
    dir("/home/ana/.ssh", 0700);
    for (it = 0; it < n; ++it) {
        InvList items, notes;
        size_t pl = mutate(passwd_t, passwd, sizeof(passwd)), sl = mutate(shadow_t, shadow, sizeof(shadow));
        size_t gl = mutate(group_t, group, sizeof(group)), tl = mutate(tcp_t, tcp, sizeof(tcp));
        size_t kl = mutate(keys_t, keys, sizeof(keys)), i, a;
        put_bytes("/etc/passwd", passwd, pl, 0644);
        put_bytes("/etc/shadow", shadow, sl, 0600);
        put_bytes("/etc/group", group, gl, 0644);
        put_bytes("/proc/net/tcp", tcp, tl, 0644);
        put_bytes("/home/ana/.ssh/authorized_keys", keys, kl, 0600);
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
    CHECK(inventory_read(path, &back, err, sizeof(err)) && same_list(&items, &back), "long report unreadable: %s", err);
    remove(path);
    inv_free(&items);
    inv_free(&notes);
    inv_free(&back);
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
#endif
    if (failures) {
        fprintf(stderr, "test-inventory: %d failure(s)\n", failures);
        return 1;
    }
    puts("test-inventory: ok");
    return 0;
}
