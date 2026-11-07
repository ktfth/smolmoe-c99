/* Unit tests for "!comando" in the chat: recognizing shells, building their
 * Windows command lines, and the "cd" the chat runs itself. Portable: nothing
 * here starts a process (tests/shell.test.ts does). */
#include <stdio.h>
#include <string.h>

#include "shell.h"

static int failures;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                      \
            fputc('\n', stderr);                               \
            ++failures;                                        \
        }                                                      \
    } while (0)

static void expect_shell(const char *program, ShellKind kind, const char *name) {
    Shell sh;
    int ok = shell_classify(program, &sh);
    CHECK(ok && sh.kind == kind && !strcmp(sh.name, name) && !strcmp(sh.program, program),
          "%s -> ok=%d kind=%d name=%s", program, ok, ok ? (int)sh.kind : -1, ok ? sh.name : "");
}

static void test_classify(void) {
    Shell sh;
    expect_shell("C:\\Program Files\\PowerShell\\7\\pwsh.exe", SHELL_POWERSHELL, "PowerShell 7");
    expect_shell("C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", SHELL_POWERSHELL,
                 "Windows PowerShell");
    expect_shell("C:\\WINDOWS\\system32\\cmd.exe", SHELL_CMD, "cmd");
    expect_shell("C:\\WINDOWS\\SYSTEM32\\CMD.EXE", SHELL_CMD, "cmd");
    expect_shell("C:\\Program Files\\Git\\usr\\bin\\bash.exe", SHELL_POSIX, "bash");
    expect_shell("C:\\Windows\\System32\\bash.exe", SHELL_POSIX, "bash");
    expect_shell("/usr/bin/zsh", SHELL_POSIX, "zsh");
    expect_shell("/data/data/com.termux/files/usr/bin/bash", SHELL_POSIX, "bash");
    expect_shell("/usr/local/bin/fish", SHELL_POSIX, "fish");
    expect_shell("pwsh", SHELL_POWERSHELL, "PowerShell 7");
    expect_shell("sh", SHELL_POSIX, "sh");
    CHECK(!shell_classify("C:\\Windows\\explorer.exe", &sh), "explorer is not a shell");
    CHECK(!shell_classify("C:\\Users\\x\\.bun\\bin\\bun.exe", &sh), "bun is not a shell");
    CHECK(!shell_classify("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell_ise.exe", &sh), "the ISE");
    CHECK(!shell_classify("/usr/bin/bashful", &sh), "a name that only starts like a shell");
    CHECK(!shell_classify("", &sh), "empty");
    CHECK(!shell_classify(".exe", &sh), "only the extension");
}

static void test_choose(void) {
    Shell sh;
    char err[256] = "";
    CHECK(shell_choose(NULL, "C:\\Program Files\\Git\\usr\\bin\\bash.exe", &sh, err, sizeof(err)) &&
              sh.kind == SHELL_POSIX && !strcmp(sh.program, "C:\\Program Files\\Git\\usr\\bin\\bash.exe"),
          "the parent shell, with its full path (%s)", err);
    CHECK(shell_choose("/bin/sh", "C:\\WINDOWS\\system32\\cmd.exe", &sh, err, sizeof(err)) && sh.kind == SHELL_POSIX,
          "--shell beats the parent (%s)", err);
    CHECK(!shell_choose("notepad", NULL, &sh, err, sizeof(err)) && strstr(err, "desconhecido"),
          "unknown --shell: %s", err);
    CHECK(shell_choose(NULL, "C:\\Windows\\explorer.exe", &sh, err, sizeof(err)) || strstr(err, "shell"),
          "a parent that is not a shell falls back to the system's: %s", err);
}

static void expect_line(ShellKind kind, const char *program, const char *command, const char *want) {
    Shell sh;
    char out[512];
    size_t n;
    memset(&sh, 0, sizeof(sh));
    sh.kind = kind;
    strcpy(sh.program, program);
    n = shell_command_line(&sh, command, out, sizeof(out));
    CHECK(n == strlen(want) && !strcmp(out, want), "%s\n  got:  %s\n  want: %s", command, n ? out : "(0)", want);
}

static void test_command_line(void) {
    Shell sh;
    char small[40];
    expect_line(SHELL_CMD, "C:\\WINDOWS\\system32\\cmd.exe", "dir \"a b\" & echo %PATH% | more",
                "\"C:\\WINDOWS\\system32\\cmd.exe\" /d /s /c \"dir \"a b\" & echo %PATH% | more\"");
    /* PowerShell: one argument in the CommandLineToArgvW rules. */
    expect_line(SHELL_POWERSHELL, "C:\\Program Files\\PowerShell\\7\\pwsh.exe",
                "echo 'ol\xc3\xa1' > \"a b.txt\"; exit 3 \xf0\x9f\x98\x80",
                "\"C:\\Program Files\\PowerShell\\7\\pwsh.exe\" -NoLogo -NoProfile -Command "
                "\"echo 'ol\xc3\xa1' > \\\"a b.txt\\\"; exit 3 \xf0\x9f\x98\x80\"");
    expect_line(SHELL_POWERSHELL, "pwsh", "dir C:\\dir\\ ; echo x\\\"y",
                "\"pwsh\" -NoLogo -NoProfile -Command \"dir C:\\dir\\ ; echo x\\\\\\\"y\""); /* \" -> \\\" */
    expect_line(SHELL_POWERSHELL, "pwsh", "dir C:\\dir\\",
                "\"pwsh\" -NoLogo -NoProfile -Command \"dir C:\\dir\\\\\""); /* a final \ is doubled */
    /* sh-like: the command goes in SMOLMOE_SHELL_COMMAND; the line never changes. */
    expect_line(SHELL_POSIX, "C:\\Program Files\\Git\\usr\\bin\\bash.exe", "echo \"C:\\dir\\\" 'x' \\",
                "\"C:\\Program Files\\Git\\usr\\bin\\bash.exe\" -c \"eval \\\"$" SHELL_COMMAND_ENV "\\\"\"");

    memset(&sh, 0, sizeof(sh));
    sh.kind = SHELL_POWERSHELL;
    strcpy(sh.program, "pwsh");
    CHECK(!shell_command_line(&sh, "echo \xc3(", small, sizeof(small)), "invalid UTF-8 is refused");
    CHECK(!shell_command_line(&sh, "echo \xed\xa0\x80", small, sizeof(small)), "an encoded surrogate is refused");
    CHECK(!shell_command_line(&sh, "echo this command is too long for the buffer", small, sizeof(small)),
          "a line that does not fit is refused, not cut");
    sh.kind = SHELL_CMD;
    CHECK(!shell_command_line(&sh, "echo this one does not fit either, at all", small, sizeof(small)),
          "cmd too");
}

static void expect_cd(const char *command, int windows, int want, const char *target) {
    char out[64] = "?";
    int got = shell_cd_target(command, "/home/ana", windows, out, sizeof(out));
    CHECK(got == want && (want != 1 || !strcmp(out, target)), "%s -> %d '%s' (want %d '%s')", command, got, out, want,
          target ? target : "");
}

static void test_cd(void) {
    expect_cd("cd src", 0, 1, "src");
    expect_cd("  cd   \"C:\\Meus Arquivos\"  ", 1, 1, "C:\\Meus Arquivos");
    expect_cd("cd 'pasta com espaço'", 0, 1, "pasta com espaço");
    expect_cd("cd /d D:\\projetos", 1, 1, "D:\\projetos");
    expect_cd("cd /d D:\\projetos", 0, 1, "/d D:\\projetos"); /* "/d" is cmd's: elsewhere it is a path */
    expect_cd("chdir ..", 1, 1, "..");
    expect_cd("cd", 0, 1, "");
    expect_cd("cd   ", 1, 1, "");
    expect_cd("cd ~", 0, 1, "/home/ana");
    expect_cd("cd ~/war", 0, 1, "/home/ana/war");
    expect_cd("cd /c/Users/Ana", 1, 1, "C:/Users/Ana");
    expect_cd("cd /c", 1, 1, "C:/");
    expect_cd("cd /c/Users/Ana", 0, 1, "/c/Users/Ana");
    expect_cd("cd /cache", 1, 1, "/cache");
    expect_cd("cd src && ls", 0, 0, NULL);
    expect_cd("cd src; ls", 0, 0, NULL);
    expect_cd("cd $(mktemp -d)", 0, 0, NULL);
    expect_cd("cd `pwd`", 0, 0, NULL);
    expect_cd("cd x > y", 1, 0, NULL);
    expect_cd("cdx", 0, 0, NULL);
    expect_cd("dir", 1, 0, NULL);
    expect_cd("cd uma/pasta/com/um/nome/longo/demais/para/caber/no/espaco/do/teste/x", 0, -1, NULL);
}

int main(void) {
    test_classify();
    test_choose();
    test_command_line();
    test_cd();
    if (failures) {
        fprintf(stderr, "%d shell test(s) failed\n", failures);
        return 1;
    }
    puts("shell tests passed");
    return 0;
}
