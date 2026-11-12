/* The POSIX side of the command check (cmdcheck.h): splitting shell snippets
 * into commands, the commands that change the system, the sh script that
 * looks names up and reads manual pages, and the index of options carried in
 * the binary for machines without manual pages. */
#include "cmdcheck.h"

#include <stdio.h>
#include <string.h>

#define WORDS_MAX 48
#define WORD_MAX 160
#define NAME_MAX_LEN 64
#define OPTION_MAX 40
#define OPTIONS_PER_LINE 12
#define LINE_MAX_LEN 640
#define DELIM_MAX 64

typedef struct Word {
    char text[WORD_MAX];
    size_t len;
    int quoted; /* began with a quote: never a name or an option */
    int subst;  /* holds $(...) or backticks: never a name */
} Word;

typedef struct Cmd {
    Word w[WORDS_MAX];
    int n;
} Cmd;

typedef struct Sink {
    char *out;
    size_t cap, *len;
    char *changes;
    size_t changes_cap;
    int lines;
} Sink;

static int in_set(const char *s, const char *const *set) {
    for (; *set; ++set)
        if (!strcmp(s, *set)) return 1;
    return 0;
}

static int is_alnum(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

/* ---- the tables ---------------------------------------------------------- */

static const char *const KEYWORDS[] = {"if", "then", "else", "elif", "fi", "do", "done", "while", "until", "{",
                                       "}", "!", "[[", "]]", "function", NULL};
/* Their following words are not a command: for x in a b; case $x in. */
static const char *const LIST_KEYWORDS[] = {"for", "case", "select", "esac", "in", NULL};

/* Programs whose first word after the name is a subcommand worth checking.
 * Not ip or nmcli: they take abbreviations (ip a, nmcli d). */
static const char *const SUBCOMMANDS[] = {"apt", "apt-get", "apt-cache", "apk", "systemctl", "git", "docker",
                                          "podman", "dnf", "yum", "zypper", "snap", "flatpak", "ufw", "loginctl",
                                          "timedatectl", "hostnamectl", "resolvectl", "brew", "launchctl",
                                          "diskutil", NULL};

/* Any use changes the system. */
static const char *const CHANGERS[] = {
    "rm", "rmdir", "mv", "dd", "shred", "wipefs", "mkfs", "mkswap", "fdisk", "sfdisk", "cfdisk", "gdisk", "sgdisk",
    "parted", "truncate", "chmod", "chown", "chgrp", "chattr", "kill", "pkill", "killall", "reboot", "shutdown",
    "poweroff", "halt", "useradd", "userdel", "usermod", "groupadd", "groupdel", "gpasswd", "passwd", "chpasswd",
    "mount", "umount", "swapoff", "swapon", "iptables", "ip6tables", "nft", "modprobe", "rmmod", "insmod",
    "update-grub", "grub-install", NULL};

typedef struct SubChange {
    const char *name;
    const char *const *subs;
} SubChange;

static const char *const SYSTEMCTL_CHANGES[] = {"start", "stop", "restart", "reload", "try-restart",
                                                "reload-or-restart", "try-reload-or-restart", "enable", "disable",
                                                "reenable", "mask", "unmask", "kill", "isolate", "poweroff",
                                                "reboot", "halt", "suspend", "hibernate", "set-default", "edit",
                                                "revert", NULL};
static const char *const APT_CHANGES[] = {"install", "reinstall", "remove", "purge", "autoremove", "upgrade",
                                          "full-upgrade", "dist-upgrade", "build-dep", NULL};
static const char *const APK_CHANGES[] = {"add", "del", "upgrade", NULL};
static const char *const DNF_CHANGES[] = {"install", "remove", "erase", "upgrade", "update", "reinstall",
                                          "downgrade", "autoremove", "distro-sync", NULL};
static const char *const SNAP_CHANGES[] = {"install", "remove", "uninstall", "refresh", "update", NULL};
static const char *const UFW_CHANGES[] = {"enable", "disable", "allow", "deny", "reject", "limit", "delete",
                                          "reset", NULL};
static const char *const BREW_CHANGES[] = {"install", "uninstall", "remove", "upgrade", "reinstall", NULL};
static const char *const LAUNCHCTL_CHANGES[] = {"load", "unload", "bootstrap", "bootout", "remove", "kickstart",
                                                "stop", "enable", "disable", NULL};
static const char *const DISKUTIL_CHANGES[] = {"eraseDisk", "eraseVolume", "partitionDisk", "unmount",
                                               "unmountDisk", "reformat", NULL};
static const char *const LOGINCTL_CHANGES[] = {"terminate-session", "terminate-user", "kill-session", "kill-user",
                                               "poweroff", "reboot", NULL};
static const char *const HOSTNAMECTL_CHANGES[] = {"set-hostname", "hostname", NULL};
static const char *const TIMEDATECTL_CHANGES[] = {"set-time", "set-timezone", "set-ntp", "set-local-rtc", NULL};
static const char *const CONTAINER_CHANGES[] = {"rm", "rmi", "stop", "kill", "restart", "prune", NULL};

static const SubChange SUB_CHANGES[] = {
    {"systemctl", SYSTEMCTL_CHANGES}, {"apt", APT_CHANGES},         {"apt-get", APT_CHANGES},
    {"apk", APK_CHANGES},             {"dnf", DNF_CHANGES},         {"yum", DNF_CHANGES},
    {"zypper", DNF_CHANGES},          {"snap", SNAP_CHANGES},       {"flatpak", SNAP_CHANGES},
    {"ufw", UFW_CHANGES},             {"brew", BREW_CHANGES},       {"launchctl", LAUNCHCTL_CHANGES},
    {"diskutil", DISKUTIL_CHANGES},   {"loginctl", LOGINCTL_CHANGES}, {"hostnamectl", HOSTNAMECTL_CHANGES},
    {"timedatectl", TIMEDATECTL_CHANGES}, {"docker", CONTAINER_CHANGES}, {"podman", CONTAINER_CHANGES}};

static const char *const SERVICE_ACTIONS[] = {"start", "stop", "restart", "reload", "force-reload", NULL};
static const char *const DPKG_CHANGES[] = {"-i", "-r", "-P", "--install", "--remove", "--purge", NULL};
static const char *const CRONTAB_CHANGES[] = {"-r", "-e", NULL};
static const char *const DATE_CHANGES[] = {"-s", "--set", NULL};
static const char *const HWCLOCK_CHANGES[] = {"--systohc", "-w", "--set", NULL};
static const char *const DELETERS[] = {"rm", "mv", "shred", NULL};

/* Wrappers: the command follows them. Those that are programs worth looking
 * up get a line of their own; their options that take a value are listed. */
static const char *const SUDO_VALUE_OPTIONS[] = {"-u", "-g", "-U", "-C", "-D", "-h", "-p", "-r", "-t", "-T", NULL};
static const char *const ENV_VALUE_OPTIONS[] = {"-u", "-C", "-S", NULL};
static const char *const NICE_VALUE_OPTIONS[] = {"-n", NULL};
static const char *const TIMEOUT_VALUE_OPTIONS[] = {"-s", "-k", NULL};
static const char *const XARGS_VALUE_OPTIONS[] = {"-I", "-n", "-P", "-d", "-L", "-s", "-E", "-a", NULL};
static const char *const NO_VALUE_OPTIONS[] = {NULL};

typedef struct Wrapper {
    const char *name;
    int looked_up;  /* a line of its own: may be missing (sudo in a container) */
    int has_operand; /* a word before the command: timeout's duration */
    const char *const *value_options;
} Wrapper;

static const Wrapper WRAPPERS[] = {
    {"sudo", 1, 0, SUDO_VALUE_OPTIONS},    {"doas", 1, 0, SUDO_VALUE_OPTIONS}, {"env", 0, 0, ENV_VALUE_OPTIONS},
    {"time", 0, 0, NO_VALUE_OPTIONS},      {"nohup", 0, 0, NO_VALUE_OPTIONS},  {"exec", 0, 0, NO_VALUE_OPTIONS},
    {"nice", 0, 0, NICE_VALUE_OPTIONS},    {"timeout", 1, 1, TIMEOUT_VALUE_OPTIONS},
    {"xargs", 1, 0, XARGS_VALUE_OPTIONS}};

/* ---- one command --------------------------------------------------------- */

static int is_assignment(const Word *w) {
    size_t i = 0;
    if (w->quoted || !(w->text[0] == '_' || (is_alnum(w->text[0]) && !(w->text[0] >= '0' && w->text[0] <= '9'))))
        return 0;
    while (w->text[i] == '_' || is_alnum(w->text[i])) ++i;
    return i > 0 && w->text[i] == '=';
}

/* "2>&1", ">", "<<EOF", "&>log": returns 1, or 2 when the target is the next word. */
static int redirection(const Word *w) {
    const char *t = w->text;
    size_t i = 0, ops = 0;
    if (w->quoted) return 0;
    while (t[i] >= '0' && t[i] <= '9') ++i;
    if (t[i] == '&' && t[i + 1] == '>') ++i;
    while (t[i] == '<' || t[i] == '>' || (ops && (t[i] == '&' || t[i] == '|' || t[i] == '-'))) ++i, ++ops;
    if (!ops) return 0;
    return t[i] ? 1 : 2;
}

static int valid_name(const Word *w) {
    size_t i;
    if (w->quoted || w->subst || !w->len || w->len > NAME_MAX_LEN) return 0;
    if (!is_alnum(w->text[0]) && w->text[0] != '_') return 0;
    for (i = 0; i < w->len; ++i)
        if (!is_alnum(w->text[i]) && !strchr("_.+:-", w->text[i])) return 0;
    return 1;
}

static int valid_subcommand(const Word *w) {
    size_t i;
    if (w->quoted || w->subst || !w->len || w->len > NAME_MAX_LEN) return 0;
    if (!((w->text[0] >= 'a' && w->text[0] <= 'z') || (w->text[0] >= 'A' && w->text[0] <= 'Z'))) return 0;
    for (i = 0; i < w->len; ++i)
        if (!is_alnum(w->text[i]) && w->text[i] != '-') return 0;
    return 1;
}

/* The option without "=value", in out; 0 when it is not one worth checking. */
static int option_of(const Word *w, char out[OPTION_MAX]) {
    size_t i, n = 0;
    if (w->quoted || w->subst || w->len < 2 || w->text[0] != '-') return 0;
    if (!strcmp(w->text, "--")) return 0;
    for (i = 0; i < w->len && w->text[i] != '='; ++i) {
        if (n + 1 >= OPTION_MAX || (!is_alnum(w->text[i]) && w->text[i] != '-')) return 0;
        out[n++] = w->text[i];
    }
    out[n] = 0;
    if (out[1] == '-' ? !is_alnum(out[2]) : !is_alnum(out[1])) return 0;
    return 1;
}

static void add_change(Sink *s, const char *name) {
    char line[NAME_MAX_LEN + 8];
    size_t used = strlen(s->changes), n;
    snprintf(line, sizeof(line), "W\t%s\t-\n", name);
    n = strlen(line);
    if (strstr(s->changes, line) || used + n >= s->changes_cap) return;
    memcpy(s->changes + used, line, n + 1);
}

/* Whether out already holds `line` as a whole line. */
static int has_line(const char *out, size_t len, const char *line, size_t n) {
    size_t i = 0;
    while (i + n <= len) {
        if (!memcmp(out + i, line, n)) return 1;
        while (i < len && out[i] != '\n') ++i;
        ++i;
    }
    return 0;
}

static void add_line(Sink *s, const char *line, size_t n) {
    if (has_line(s->out, *s->len, line, n) || *s->len + n >= s->cap) return;
    memcpy(s->out + *s->len, line, n);
    *s->len += n;
    s->out[*s->len] = 0;
    ++s->lines;
}

static void add_name_only(Sink *s, const char *name) {
    char line[NAME_MAX_LEN + 8];
    int n = snprintf(line, sizeof(line), "%s\t.\n", name);
    add_line(s, line, (size_t)n);
}

/* pacman -S... syncs (changes) unless it only searches or shows: -Ss, -Si, -Sl, -Sg, -Sp. */
static int pacman_changes(const char *opt) {
    if (opt[0] != '-' || opt[1] == '-') return 0;
    if (opt[1] == 'R' || opt[1] == 'U') return 1;
    return opt[1] == 'S' && !strpbrk(opt + 2, "silgp");
}

static int changes_system(const char *name, const char *sub, const Word *args, int from, int m) {
    size_t i;
    int j;
    if (in_set(name, CHANGERS) || !strncmp(name, "mkfs.", 5)) return 1;
    for (i = 0; i < sizeof(SUB_CHANGES) / sizeof(SUB_CHANGES[0]); ++i)
        if (!strcmp(name, SUB_CHANGES[i].name) && sub && in_set(sub, SUB_CHANGES[i].subs)) return 1;
    for (j = from; j < m; ++j) {
        const char *t = args[j].text;
        if (args[j].quoted) continue;
        if (!strcmp(name, "service") && in_set(t, SERVICE_ACTIONS)) return 1;
        if ((!strcmp(name, "docker") || !strcmp(name, "podman")) && !strcmp(t, "prune")) return 1;
        if (!strcmp(name, "sed") && ((t[0] == '-' && t[1] == 'i') || !strncmp(t, "--in-place", 10))) return 1;
        if (!strcmp(name, "find") && (!strcmp(t, "-delete") ||
                                      ((!strcmp(t, "-exec") || !strcmp(t, "-execdir") || !strcmp(t, "-ok")) &&
                                       j + 1 < m && in_set(args[j + 1].text, DELETERS))))
            return 1;
        if (!strcmp(name, "pacman") && pacman_changes(t)) return 1;
        if (!strcmp(name, "dpkg") && in_set(t, DPKG_CHANGES)) return 1;
        if (!strcmp(name, "rpm") && t[0] == '-' &&
            ((t[1] && strchr("ieUF", t[1]) && t[1] != '-') || !strcmp(t, "--install") || !strcmp(t, "--erase") ||
             !strcmp(t, "--upgrade") || !strcmp(t, "--freshen")))
            return 1;
        if (!strcmp(name, "crontab") && in_set(t, CRONTAB_CHANGES)) return 1;
        if (!strcmp(name, "sysctl") && (!strcmp(t, "-w") || strchr(t, '='))) return 1;
        if (!strcmp(name, "date") && (in_set(t, DATE_CHANGES) || !strncmp(t, "--set=", 6))) return 1;
        if (!strcmp(name, "hwclock") && in_set(t, HWCLOCK_CHANGES)) return 1;
    }
    return 0;
}

static const Wrapper *wrapper_of(const Word *w) {
    size_t i;
    if (w->quoted) return NULL;
    for (i = 0; i < sizeof(WRAPPERS) / sizeof(WRAPPERS[0]); ++i)
        if (!strcmp(w->text, WRAPPERS[i].name)) return &WRAPPERS[i];
    return NULL;
}

/* The words of one command, without redirections and their targets. */
static int strip_redirections(const Cmd *c, const Word **args) {
    int i, m = 0;
    for (i = 0; i < c->n; ++i) {
        int r = redirection(&c->w[i]);
        if (r == 2) ++i;
        if (!r) args[m++] = &c->w[i];
    }
    return m;
}

static void process(Sink *s, const Cmd *c) {
    const Word *args[WORDS_MAX];
    Word copy[WORDS_MAX];
    char line[LINE_MAX_LEN], opt[OPTION_MAX];
    const char *sub = NULL;
    int m = strip_redirections(c, args), k = 0, j, options = 0;
    size_t n;
    for (j = 0; j < m; ++j) copy[j] = *args[j];
    while (k < m) {
        const Wrapper *wr;
        if (is_assignment(&copy[k])) {
            ++k;
        } else if (!copy[k].quoted && in_set(copy[k].text, LIST_KEYWORDS)) {
            return;
        } else if (!copy[k].quoted && in_set(copy[k].text, KEYWORDS)) {
            ++k;
        } else if (!copy[k].quoted && !strcmp(copy[k].text, "command")) {
            if (k + 1 < m && (!strcmp(copy[k + 1].text, "-v") || !strcmp(copy[k + 1].text, "-V"))) return;
            ++k;
        } else if ((wr = wrapper_of(&copy[k])) != NULL) {
            if (wr->looked_up) add_name_only(s, wr->name);
            for (++k; k < m && copy[k].text[0] == '-' && !copy[k].quoted; ++k)
                if (in_set(copy[k].text, wr->value_options)) ++k;
            while (!strcmp(wr->name, "env") && k < m && is_assignment(&copy[k])) ++k;
            if (wr->has_operand && k < m) ++k;
        } else {
            break;
        }
    }
    if (k >= m || !valid_name(&copy[k])) return;
    j = k + 1;
    if (in_set(copy[k].text, SUBCOMMANDS) && j < m && valid_subcommand(&copy[j])) sub = copy[j++].text;
    n = (size_t)snprintf(line, sizeof(line), "%s\t%s", copy[k].text, sub ? sub : ".");
    for (; j < m && options < OPTIONS_PER_LINE; ++j) {
        char padded[OPTION_MAX + 3];
        if (n + 2 >= sizeof(line) || (!copy[j].quoted && !strcmp(copy[j].text, "--"))) break;
        if (!option_of(&copy[j], opt)) continue;
        snprintf(padded, sizeof(padded), "\t%s\t", opt);
        line[n] = '\t';
        line[n + 1] = 0;
        if (strstr(line, padded)) continue; /* the same option twice */
        line[n] = 0;
        if (n + strlen(opt) + 3 >= sizeof(line)) break;
        n += (size_t)snprintf(line + n, sizeof(line) - n, "\t%s", opt);
        ++options;
    }
    line[n++] = '\n';
    line[n] = 0;
    add_line(s, line, n);
    if (changes_system(copy[k].text, sub, copy, k + 1, m)) add_change(s, copy[k].text);
}

/* ---- the tokenizer ------------------------------------------------------- */

typedef struct Lexer {
    const char *s;
    size_t n, i;
    Cmd cmd;
    Word cur;
    int in_word;
    int want_delim;           /* the last word was "<<": the next one is a here-document's end */
    char delim[DELIM_MAX];    /* a here-document starts at the next line */
    int delim_tabs;           /* <<-: its lines may start with tabs */
} Lexer;

static void put_char(Lexer *lx, char c) {
    if (!lx->in_word) {
        memset(&lx->cur, 0, sizeof(lx->cur));
        lx->in_word = 1;
    }
    if (lx->cur.len + 1 < WORD_MAX) lx->cur.text[lx->cur.len++] = c;
}

static void set_delim(Lexer *lx, const char *d, int tabs) {
    size_t n = strlen(d);
    if (n >= sizeof(lx->delim)) n = sizeof(lx->delim) - 1; /* a longer end line never matches: the rest is skipped */
    memcpy(lx->delim, d, n);
    lx->delim[n] = 0;
    lx->delim_tabs = tabs;
}

static void end_word(Lexer *lx) {
    Word *w = &lx->cur;
    if (!lx->in_word) return;
    lx->in_word = 0;
    w->text[w->len] = 0;
    if (lx->want_delim) {
        lx->want_delim = 0;
        set_delim(lx, w->text, lx->delim_tabs);
    } else if (!w->quoted && !strncmp(w->text, "<<", 2) && w->text[2] != '<') {
        int tabs = w->text[2] == '-';
        if (w->text[2 + tabs]) set_delim(lx, w->text + 2 + tabs, tabs);
        else lx->want_delim = 1, lx->delim_tabs = tabs;
    }
    if (lx->cmd.n < WORDS_MAX) lx->cmd.w[lx->cmd.n++] = *w;
}

static void end_command(Lexer *lx, Sink *s) {
    end_word(lx);
    if (lx->cmd.n) process(s, &lx->cmd);
    lx->cmd.n = 0;
}

/* Past a here-document's lines, up to and including its end line. */
static void skip_heredoc(Lexer *lx) {
    size_t dl = strlen(lx->delim);
    while (lx->i < lx->n) {
        size_t b = lx->i, e = b;
        while (e < lx->n && lx->s[e] != '\n') ++e;
        lx->i = e < lx->n ? e + 1 : e;
        if (lx->delim_tabs)
            while (b < e && lx->s[b] == '\t') ++b;
        if (e > b && lx->s[e - 1] == '\r') --e;
        if (e - b == dl && !memcmp(lx->s + b, lx->delim, dl)) break;
    }
    lx->delim[0] = 0;
}

/* $(...) with nesting and quotes, or `...`: kept out of the word's text. */
static void skip_substitution(Lexer *lx) {
    int depth = 0;
    char q = 0;
    put_char(lx, '$');
    lx->cur.subst = 1;
    if (lx->s[lx->i] == '`') {
        for (++lx->i; lx->i < lx->n && lx->s[lx->i] != '`'; ++lx->i) {}
        ++lx->i;
        return;
    }
    for (; lx->i < lx->n; ++lx->i) {
        char c = lx->s[lx->i];
        if (q) {
            if (c == q) q = 0;
        } else if (c == '\'' || c == '"') {
            q = c;
        } else if (c == '(') {
            ++depth;
        } else if (c == ')' && --depth == 0) {
            ++lx->i;
            return;
        }
    }
}

static void quoted(Lexer *lx, char q) {
    if (!lx->in_word) {
        put_char(lx, q);
        lx->cur.len = 0;
        lx->cur.quoted = 1;
    }
    for (++lx->i; lx->i < lx->n && lx->s[lx->i] != q; ++lx->i) {
        if (q == '"' && lx->s[lx->i] == '\\' && lx->i + 1 < lx->n) ++lx->i;
        put_char(lx, lx->s[lx->i]);
    }
    ++lx->i;
}

/* A line's start: blanks, then a "$ " prompt from a transcript. */
static void line_start(Lexer *lx) {
    while (lx->i < lx->n && (lx->s[lx->i] == ' ' || lx->s[lx->i] == '\t')) ++lx->i;
    if (lx->i + 1 < lx->n && lx->s[lx->i] == '$' && lx->s[lx->i + 1] == ' ') lx->i += 2;
}

static void lex(Lexer *lx, Sink *s) {
    line_start(lx);
    while (lx->i < lx->n) {
        char c = lx->s[lx->i], next = lx->i + 1 < lx->n ? lx->s[lx->i + 1] : 0;
        char last = lx->in_word && lx->cur.len ? lx->cur.text[lx->cur.len - 1] : 0;
        if (c == '\\' && next == '\n') {
            lx->i += 2;
        } else if (c == '\\' && next == '\r' && lx->i + 2 < lx->n && lx->s[lx->i + 2] == '\n') {
            lx->i += 3;
        } else if (c == '\n') {
            end_command(lx, s);
            ++lx->i;
            if (lx->delim[0]) skip_heredoc(lx);
            line_start(lx);
        } else if (c == '#' && !lx->in_word) {
            while (lx->i < lx->n && lx->s[lx->i] != '\n') ++lx->i;
        } else if (c == ' ' || c == '\t' || c == '\r') {
            end_word(lx);
            ++lx->i;
        } else if ((c == '&' && (next == '>' || last == '>' || last == '<')) || (c == '|' && last == '>')) {
            put_char(lx, c); /* &>, 2>&1, >| */
            ++lx->i;
        } else if (strchr("|&;()", c)) {
            end_command(lx, s);
            ++lx->i;
        } else if (c == '\'' || c == '"') {
            quoted(lx, c);
        } else if (c == '\\') {
            if (next) put_char(lx, next);
            lx->i += 2;
        } else if ((c == '$' && next == '(') || c == '`') {
            if (c == '$') ++lx->i;
            skip_substitution(lx);
        } else {
            put_char(lx, c);
            ++lx->i;
        }
    }
    end_command(lx, s);
}

int cmdcheck_sh_commands(const char *text, size_t n, char *out, size_t cap, size_t *len, char *changes,
                         size_t changes_cap) {
    static Lexer lx; /* large (a command of words): kept off the stack; one check runs at a time */
    Sink s;
    memset(&lx, 0, sizeof(lx));
    lx.s = text;
    lx.n = n;
    s.out = out, s.cap = cap, s.len = len, s.changes = changes, s.changes_cap = changes_cap, s.lines = 0;
    if (cap && *len < cap) out[*len] = 0;
    lex(&lx, &s);
    return s.lines;
}

/* ---- the sh script ------------------------------------------------------- */

/* Reads "name<TAB>sub<TAB>options" lines up to a line "E" and answers with
 * N/O/S/M findings and a line "E". Request lines are only split (set -f, no
 * expansion): nothing from a reply runs. It runs `command -v`, `man -w`, a
 * decompressor and grep; never the program itself, not even with --help. */
static const char *const SH_SCRIPT[] = {
    "T=$(printf '\\t')",
    "tmp=$(mktemp 2>/dev/null) || tmp=${TMPDIR:-/tmp}/smolmoe-man.$$",
    "trap 'rm -f \"$tmp\"' EXIT",
    "norm() {",
    "  sed -e 's/\\\\f[BIRPV]//g' -e 's/\\\\f([A-Za-z][A-Za-z]//g' -e 's/\\\\f\\[[^]]*\\]//g' -e 's/\\\\-/-/g' \\",
    "      -e 's/\\\\(hy/-/g' -e 's/\\\\(en/-/g' -e 's/\\\\(em/-/g' -e 's/\\\\&//g' -e 's/\\\\[ ~]/ /g' -e 's/\"/ /g'",
    "}",
    "cites() {",
    "  grep -q -E -e \"(^|[^A-Za-z0-9-])$1([^A-Za-z0-9-]|\\$)\" \"$tmp\"",
    "}",
    "while IFS= read -r line; do",
    "  [ \"$line\" = E ] && break",
    "  old=$IFS; IFS=$T; set -f; set -- $line; set +f; IFS=$old",
    "  name=$1; sub=$2; shift 2",
    "  if ! command -v \"$name\" >/dev/null 2>&1; then printf 'N\\t%s\\t\\n' \"$name\"; continue; fi",
    "  case $(command -v \"$name\") in /*) ;; *) continue ;; esac",
    "  [ \"$sub\" = . ] && [ $# -eq 0 ] && continue",
    "  page=$(man -w \"$name\" 2>/dev/null | head -n 1)",
    "  if [ -z \"$page\" ] || [ ! -r \"$page\" ]; then printf 'M\\t%s\\n' \"$name\"; continue; fi",
    "  case $page in",
    "    *.gz) gzip -dc \"$page\" ;;",
    "    *.bz2) bzip2 -dc \"$page\" ;;",
    "    *.xz) xz -dc \"$page\" ;;",
    "    *.zst) zstd -dc \"$page\" ;;",
    "    *) cat \"$page\" ;;",
    "  esac 2>/dev/null | norm > \"$tmp\"",
    "  if [ ! -s \"$tmp\" ]; then printf 'M\\t%s\\n' \"$name\"; continue; fi",
    "  if [ \"$sub\" != . ] && ! grep -q -w -e \"$sub\" \"$tmp\"; then printf 'S\\t%s\\t%s\\n' \"$name\" \"$sub\"; fi",
    "  for o in \"$@\"; do",
    "    cites \"$o\" && continue",
    "    case $o in",
    "      --*) ;;",
    "      -??*)",
    "        r=${o#-}; ok=1",
    "        while [ -n \"$r\" ]; do",
    "          c=${r%\"${r#?}\"}; r=${r#?}",
    "          case $c in [0-9]) break ;; esac",
    "          cites \"-$c\" || { ok=0; break; }",
    "        done",
    "        [ \"$ok\" = 1 ] && continue ;;",
    "    esac",
    "    printf 'O\\t%s\\t%s\\n' \"$name\" \"$o\"",
    "  done",
    "done",
    "printf 'E\\n'",
};

/* ---- the index in the binary --------------------------------------------- */

typedef struct IndexEntry {
    const char *program, *options, *subs; /* " a b c ": each word between spaces */
} IndexEntry;

#include "cmdcheck_index.inc"

const char *cmdcheck_index_source(void) { return CMDCHECK_INDEX_SOURCE; }

int cmdcheck_index_has(const char *program) {
    size_t i;
    for (i = 0; i < sizeof(INDEX) / sizeof(INDEX[0]); ++i)
        if (!strcmp(INDEX[i].program, program)) return 1;
    return 0;
}

static int index_cites(const char *program, const char *word, int sub) {
    char padded[NAME_MAX_LEN + OPTION_MAX + 3];
    size_t i;
    if (strlen(word) + 3 > sizeof(padded)) return 0;
    snprintf(padded, sizeof(padded), " %s ", word);
    for (i = 0; i < sizeof(INDEX) / sizeof(INDEX[0]); ++i)
        if (!strcmp(INDEX[i].program, program) && strstr(sub ? INDEX[i].subs : INDEX[i].options, padded)) return 1;
    return 0;
}

int cmdcheck_index_option(const char *program, const char *option) {
    const char *p;
    if (index_cites(program, option, 0)) return 1;
    if (option[0] != '-' || !option[1] || option[1] == '-' || !option[2]) return 0;
    for (p = option + 1; *p && !(*p >= '0' && *p <= '9'); ++p) {
        char one[3];
        one[0] = '-', one[1] = *p, one[2] = 0;
        if (!index_cites(program, one, 0)) return 0;
    }
    return 1;
}

int cmdcheck_index_sub(const char *program, const char *subcommand) { return index_cites(program, subcommand, 1); }

size_t cmdcheck_sh_script(char *out, size_t cap) {
    size_t i, n = 0;
    for (i = 0; i < sizeof(SH_SCRIPT) / sizeof(SH_SCRIPT[0]); ++i) {
        size_t len = strlen(SH_SCRIPT[i]);
        if (n + len + 2 > cap) {
            if (cap) out[0] = 0;
            return 0;
        }
        memcpy(out + n, SH_SCRIPT[i], len);
        n += len;
        out[n++] = '\n';
    }
    out[n] = 0;
    return n;
}
