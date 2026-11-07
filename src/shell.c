/* "!comando" in the chat: see shell.h. */
#include "shell.h"
#include "process.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COMMAND_LINE_MAX 32768 /* CreateProcess limit, in characters */
#define STATUS_CONTROL_C_EXIT 0xC000013Au

static const char *const POSIX_SHELLS[] = {"bash", "sh", "zsh", "dash", "ksh", "mksh", "ash", "fish"};

static const char *base_name(const char *path) {
    const char *base = path, *p;
    for (p = path; *p; ++p)
        if (*p == '/' || *p == '\\') base = p + 1;
    return base;
}

static char lower(char c) { return c >= 'A' && c <= 'Z' ? (char)(c + 32) : c; }

/* The file name in lowercase ASCII, without a trailing ".exe". */
static int stem(const char *path, char *out, size_t cap) {
    const char *base = base_name(path);
    size_t n = strlen(base), i;
    if (n > 4 && base[n - 4] == '.' && lower(base[n - 3]) == 'e' && lower(base[n - 2]) == 'x' &&
        lower(base[n - 1]) == 'e')
        n -= 4;
    if (!n || n >= cap) return 0;
    for (i = 0; i < n; ++i) out[i] = lower(base[i]);
    out[n] = 0;
    return 1;
}

static void set(Shell *out, ShellKind kind, const char *name, const char *program) {
    out->kind = kind;
    snprintf(out->name, sizeof(out->name), "%s", name);
    snprintf(out->program, sizeof(out->program), "%s", program);
}

int shell_classify(const char *program, Shell *out) {
    char s[32];
    size_t i;
    memset(out, 0, sizeof(*out));
    if (!program || strlen(program) >= sizeof(out->program) || !stem(program, s, sizeof(s))) return 0;
    if (!strcmp(s, "pwsh")) set(out, SHELL_POWERSHELL, "PowerShell 7", program);
    else if (!strcmp(s, "powershell")) set(out, SHELL_POWERSHELL, "Windows PowerShell", program);
    else if (!strcmp(s, "cmd")) set(out, SHELL_CMD, "cmd", program);
    else {
        for (i = 0; i < sizeof(POSIX_SHELLS) / sizeof(POSIX_SHELLS[0]); ++i)
            if (!strcmp(s, POSIX_SHELLS[i])) break;
        if (i == sizeof(POSIX_SHELLS) / sizeof(POSIX_SHELLS[0])) return 0;
        set(out, SHELL_POSIX, s, program);
    }
    return 1;
}

/* A bare name is looked up in PATH here, so that CreateProcess never picks a
 * program of the same name from the current directory. */
static int resolve(Shell *sh) {
    char found[SHELL_PATH_MAX], name[SHELL_PATH_MAX + 4]; /* room for ".exe" */
    if (strchr(sh->program, '/') || strchr(sh->program, '\\')) return 1;
#if defined(_WIN32)
    if (!strchr(sh->program, '.')) {
        snprintf(name, sizeof(name), "%s.exe", sh->program);
    } else
#endif
        snprintf(name, sizeof(name), "%s", sh->program);
    if (!find_program(name, found, sizeof(found))) return 0;
    snprintf(sh->program, sizeof(sh->program), "%s", found);
    return 1;
}

static int system_shell(Shell *out) {
#if defined(_WIN32)
    static const char *const CANDIDATES[] = {"pwsh.exe", "powershell.exe", "cmd.exe"};
    char found[SHELL_PATH_MAX];
    const char *comspec = getenv("ComSpec");
    size_t i;
    for (i = 0; i < sizeof(CANDIDATES) / sizeof(CANDIDATES[0]); ++i)
        if (find_program(CANDIDATES[i], found, sizeof(found)) && shell_classify(found, out)) return 1;
    return comspec && shell_classify(comspec, out);
#else
    const char *sh = getenv("SHELL");
    return (sh && *sh && shell_classify(sh, out) && resolve(out)) || shell_classify("/bin/sh", out);
#endif
}

int shell_choose(const char *choice, const char *parent, Shell *out, char *err, size_t cap) {
    if (choice) {
        if (!shell_classify(choice, out)) {
            snprintf(err, cap, "shell desconhecido: '%s' (use pwsh, powershell, cmd, bash, sh ou o caminho de um deles)",
                     choice);
            return 0;
        }
        if (!resolve(out)) {
            snprintf(err, cap, "não encontrei '%s' no PATH", choice);
            return 0;
        }
        return 1;
    }
    if (parent && shell_classify(parent, out)) return 1;
    if (system_shell(out)) return 1;
    snprintf(err, cap, "nenhum shell encontrado");
    return 0;
}

typedef struct Text {
    char *p;
    size_t n, cap;
    int ok;
} Text;

static void put(Text *t, const char *s, size_t n) {
    if (!t->ok || n >= t->cap - t->n) {
        t->ok = 0;
        return;
    }
    memcpy(t->p + t->n, s, n);
    t->n += n;
    t->p[t->n] = 0;
}

static void put_str(Text *t, const char *s) { put(t, s, strlen(s)); }

/* The next code point of strict UTF-8 (no overlongs, surrogates or values past
 * U+10FFFF), or -1. */
static long next_code_point(const unsigned char **s) {
    const unsigned char *p = *s;
    long cp;
    int extra, i;
    if (p[0] < 0x80) {
        *s = p + 1;
        return p[0];
    }
    if (p[0] >= 0xc2 && p[0] <= 0xdf) extra = 1, cp = p[0] & 0x1f;
    else if (p[0] >= 0xe0 && p[0] <= 0xef) extra = 2, cp = p[0] & 0x0f;
    else if (p[0] >= 0xf0 && p[0] <= 0xf4) extra = 3, cp = p[0] & 0x07;
    else return -1;
    for (i = 1; i <= extra; ++i) {
        if ((p[i] & 0xc0) != 0x80) return -1;
        cp = (cp << 6) | (p[i] & 0x3f);
    }
    if ((extra == 2 && cp < 0x800) || (extra == 3 && cp < 0x10000) || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
        return -1;
    *s = p + extra + 1;
    return cp;
}

/* One argument the way CommandLineToArgvW (and so PowerShell) splits it back:
 * in quotes, a quote escaped with a backslash and the backslashes before a
 * quote doubled. */
static void put_quoted(Text *t, const char *s) {
    size_t slashes = 0, i;
    put_str(t, "\"");
    for (; *s; ++s) {
        if (*s == '\\') {
            ++slashes;
            continue;
        }
        for (i = 0; i < (*s == '"' ? 2 * slashes + 1 : slashes); ++i) put(t, "\\", 1);
        put(t, s, 1);
        slashes = 0;
    }
    for (i = 0; i < 2 * slashes; ++i) put(t, "\\", 1);
    put_str(t, "\"");
}

static int valid_utf8(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    while (*p)
        if (next_code_point(&p) < 0) return 0;
    return 1;
}

size_t shell_command_line(const Shell *sh, const char *command, char *out, size_t cap) {
    Text t;
    t.p = out;
    t.n = 0;
    t.cap = cap;
    t.ok = cap > 0;
    if (!t.ok) return 0;
    out[0] = 0;
    if (!valid_utf8(command)) return 0;
    put_str(&t, "\"");
    put_str(&t, sh->program);
    put_str(&t, "\"");
    if (sh->kind == SHELL_POWERSHELL) {
        /* Not -EncodedCommand: Windows PowerShell then writes errors as CLIXML
         * whenever the output is redirected. */
        put_str(&t, " -NoLogo -NoProfile -Command ");
        put_quoted(&t, command);
    } else if (sh->kind == SHELL_CMD) {
        put_str(&t, " /d /s /c \"");
        put_str(&t, command);
        put_str(&t, "\"");
    } else {
        put_str(&t, " -c \"eval \\\"$" SHELL_COMMAND_ENV "\\\"\"");
    }
    if (!t.ok) {
        out[0] = 0;
        return 0;
    }
    return t.n;
}

static int is_space(char c) { return c == ' ' || c == '\t'; }

/* Removes one pair of matching quotes around `s` (length *n). */
static const char *unquote(const char *s, size_t *n) {
    if (*n >= 2 && (s[0] == '"' || s[0] == '\'') && s[*n - 1] == s[0]) {
        *n -= 2;
        return s + 1;
    }
    return s;
}

int shell_cd_target(const char *command, const char *home, int windows, char *target, size_t cap) {
    const char *s = command, *arg;
    size_t n, word;
    while (is_space(*s)) ++s;
    word = !strncmp(s, "chdir", 5) ? 5 : !strncmp(s, "cd", 2) ? 2 : 0;
    if (!word || (s[word] && !is_space(s[word]))) return 0;
    s += word;
    while (is_space(*s)) ++s;
    if (windows && (s[0] == '/') && (s[1] == 'd' || s[1] == 'D') && is_space(s[2])) {
        s += 3;
        while (is_space(*s)) ++s;
    }
    if (strpbrk(s, "&|;<>`") || strstr(s, "$(")) return 0;
    n = strlen(s);
    while (n && is_space(s[n - 1])) --n;
    arg = unquote(s, &n);
    if (n && arg[0] == '~' && home && (n == 1 || arg[1] == '/' || arg[1] == '\\')) {
        if (strlen(home) + n >= cap) return -1;
        snprintf(target, cap, "%s%.*s", home, (int)(n - 1), arg + 1);
        return 1;
    }
    if (windows && n >= 2 && arg[0] == '/' && ((arg[1] | 32) >= 'a' && (arg[1] | 32) <= 'z') && (n == 2 || arg[2] == '/')) {
        if (n + 2 >= cap) return -1;
        snprintf(target, cap, "%c:/%.*s", arg[1] & ~32, (int)(n > 3 ? n - 3 : 0), n > 3 ? arg + 3 : "");
        return 1;
    }
    if (n >= cap) return -1;
    memcpy(target, arg, n);
    target[n] = 0;
    return 1;
}

static void change_dir(const char *command) {
    char target[SHELL_PATH_MAX], home[SHELL_PATH_MAX], where[SHELL_PATH_MAX], err[SHELL_PATH_MAX + 128];
#if defined(_WIN32)
    const int windows = 1;
#else
    const int windows = 0;
#endif
    int cd = shell_cd_target(command, home_directory(home, sizeof(home)) ? home : NULL, windows, target, sizeof(target));
    if (cd < 0) fputs("(caminho longo demais)\n", stderr);
    else if (target[0] && !change_directory(target, err, sizeof(err))) fprintf(stderr, "(não consegui entrar: %s)\n", err);
    else if (current_directory(where, sizeof(where))) fprintf(stderr, "(pasta atual: %s)\n", where);
}

static void report(int ok, int status, const char *err) {
    if (!ok) fprintf(stderr, "(o comando não rodou: %s)\n", err);
    else if ((unsigned)status == STATUS_CONTROL_C_EXIT) fputs("(interrompido)\n", stderr);
    else if (status < 0) fprintf(stderr, "(código de saída 0x%08X)\n", (unsigned)status); /* a Windows NTSTATUS */
    else fprintf(stderr, "(código de saída %d)\n", status);
}

void shell_execute(const Shell *sh, const char *command, int null_stdin) {
    char target[SHELL_PATH_MAX], err[512], *line = NULL;
    const char *argv[6] = {NULL};
    int status = 0, ok;
    while (is_space(*command)) ++command;
    if (!*command) {
        fputs("(escreva o comando depois do !, por exemplo: !dir)\n", stderr);
        return;
    }
    if (shell_cd_target(command, NULL, 0, target, sizeof(target))) {
        change_dir(command);
        return;
    }
#if defined(_WIN32)
    line = (char *)malloc(COMMAND_LINE_MAX);
    if (!line || !shell_command_line(sh, command, line, COMMAND_LINE_MAX)) {
        fputs(line ? "(comando longo demais ou com texto inválido: não executado)\n" : "(sem memória)\n", stderr);
        free(line);
        return;
    }
#else
    if (sh->kind == SHELL_POWERSHELL) {
        argv[0] = sh->program, argv[1] = "-NoLogo", argv[2] = "-NoProfile", argv[3] = "-Command", argv[4] = command;
    } else {
        argv[0] = sh->program, argv[1] = "-c", argv[2] = command;
    }
#endif
    ok = run_program(argv, line, sh->kind == SHELL_POSIX ? SHELL_COMMAND_ENV : NULL, command, null_stdin, &status, err,
                     sizeof(err));
    free(line);
    report(ok, status, err);
}
