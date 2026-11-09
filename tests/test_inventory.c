/* Unit tests for smolmoe-inventario: SHA-256 vectors, list merging, report
 * round trip and diff, and (POSIX) collection from a fake system tree. */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#  define _DARWIN_C_SOURCE 1
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inventory.h"
#include "sha256.h"

#if !defined(_WIN32)
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

static void test_sha256(void) {
    char hex[65];
    static const char *million_a_hex = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
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
        CHECK(!strcmp(hex, million_a_hex), "million a: %s", hex);
        free(big);
    }
}

static void test_list(void) {
    InvList l;
    inv_init(&l);
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=b");
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=a");
    inv_add(&l, "porta", "tcp 0.0.0.0:80", "processo=a");
    inv_add(&l, "porta", "tcp 0.0.0.0:8", "x\ty\nz");
    inv_add(&l, "admin", "sudo:ana", "membro");
    inv_finish(&l);
    CHECK(l.count == 3, "count %zu", l.count);
    CHECK(!strcmp(l.items[0], "admin\tsudo:ana\tmembro"), "%s", l.items[0]);
    CHECK(!strcmp(l.items[1], "porta\ttcp 0.0.0.0:8\tx?y?z"), "%s", l.items[1]);
    CHECK(!strcmp(l.items[2], "porta\ttcp 0.0.0.0:80\tprocesso=a, processo=b"), "%s", l.items[2]);
    inv_free(&l);
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

static void test_diff_and_roundtrip(void) {
    InvList a, b, back, notes;
    char text[1024], err[256], path[] = "inv-test-report.txt";
    FILE *f;
    size_t changes;
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
    changes = diff_text(&a, &b, text, sizeof(text));
    CHECK(changes == 3, "changes %zu", changes);
    CHECK(!strcmp(text, "~\tconfig\t/etc/sudoers\tsha256=aaaa -> sha256=bbbb\n"
                        "+\tporta\ttcp 0.0.0.0:4444\tprocesso=nc uid=1000 exposta=sim\n"
                        "-\tusuario\tbob\tuid=1001 shell=/bin/bash\n"),
          "diff:\n%s", text);
    CHECK(diff_text(&a, &a, text, sizeof(text)) == 0 && !text[0], "self diff: %s", text);

    inv_add(&notes, "x", "y", "z");
    f = fopen(path, "wb");
    CHECK(f != NULL, "cannot write %s", path);
    if (f) {
        inventory_write(f, &b, &notes, "host", "2026-11-09T10:00", "");
        fclose(f);
        CHECK(inventory_read(path, &back, err, sizeof(err)), "read: %s", err);
        CHECK(back.count == b.count && diff_text(&b, &back, text, sizeof(text)) == 0, "round trip: %s", text);
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

#if !defined(_WIN32)

static char root[256];

static void put(const char *rel, const char *text, unsigned mode) {
    char p[512];
    FILE *f;
    snprintf(p, sizeof(p), "%s%s", root, rel);
    f = fopen(p, "wb");
    if (!f) {
        fprintf(stderr, "FAIL cannot create %s\n", p);
        ++failures;
        return;
    }
    fputs(text, f);
    fclose(f);
    chmod(p, mode);
}

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

static void build_fake_system(void) {
    dir("/etc", 0755);
    dir("/etc/cron.d", 0755);
    dir("/etc/systemd", 0755);
    dir("/etc/systemd/system", 0755);
    dir("/etc/systemd/system/multi-user.target.wants", 0755);
    dir("/usr", 0755);
    dir("/usr/bin", 0755);
    dir("/usr/share", 0755);
    dir("/usr/share/drop", 0777);
    dir("/tmpish", 0755);
    dir("/home", 0755);
    dir("/home/ana", 0755);
    dir("/home/ana/.ssh", 0700);
    dir("/proc", 0755);
    dir("/proc/net", 0755);
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
    InvList items, notes, again;
    char text[4096];
    snprintf(root, sizeof(root), "inv-root-XXXXXX");
    if (!mkdtemp(root)) {
        fprintf(stderr, "FAIL mkdtemp\n");
        ++failures;
        return;
    }
    build_fake_system();
    inv_init(&items);
    inv_init(&notes);
    inv_init(&again);
    inventory_collect(root, &items, &notes);

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

    {
        size_t i, keys = 0, with_opts = 0;
        for (i = 0; i < items.count; ++i)
            if (!strncmp(items.items[i], "chave_ssh\tana:", 14)) {
                ++keys;
                with_opts += strstr(items.items[i], "tipo=ssh-rsa comentario=- opcoes=sim") != NULL;
            }
        CHECK(keys == 2 && with_opts == 1, "keys %zu with options %zu", keys, with_opts);
    }

    /* A second run over the same tree gives the same report. */
    inventory_collect(root, &again, &notes);
    CHECK(diff_text(&items, &again, text, sizeof(text)) == 0, "not deterministic:\n%s", text);

    /* A new cron job and a new listening port show up as additions. */
    put("/etc/cron.d/miner", "* * * * * root /tmp/x\n", 0644);
    inv_free(&again);
    inventory_collect(root, &again, &notes);
    CHECK(diff_text(&items, &again, text, sizeof(text)) == 1 && !strncmp(text, "+\tinicio\t/etc/cron.d/miner\t", 27),
          "cron addition:\n%s", text);

    inv_free(&items);
    inv_free(&notes);
    inv_free(&again);
    {
        char cmd[300];
        snprintf(cmd, sizeof(cmd), "rm -rf '%s'", root);
        if (system(cmd)) fprintf(stderr, "warning: could not remove %s\n", root);
    }
}

#endif

int main(void) {
    test_sha256();
    test_list();
    test_diff_and_roundtrip();
#if !defined(_WIN32)
    test_collect();
#endif
    if (failures) {
        fprintf(stderr, "test-inventory: %d failure(s)\n", failures);
        return 1;
    }
    puts("test-inventory: ok");
    return 0;
}
