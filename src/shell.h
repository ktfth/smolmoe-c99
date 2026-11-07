/* "!comando" in the chat: a line the user types after "!" runs in a shell.
 * Never anything the model writes, and never a line pasted along with others
 * (chat.c decides that). This module is portable: it recognizes shells, builds
 * their command lines and handles "cd"; process.c starts the program. */
#ifndef SMOLMOE_SHELL_H
#define SMOLMOE_SHELL_H

#include <stddef.h>

#define SHELL_PATH_MAX 1024
/* With an sh-like shell on Windows the command travels in this variable, so the
 * command line is constant and needs no quoting (MSYS and the C runtime quote
 * differently). */
#define SHELL_COMMAND_ENV "SMOLMOE_SHELL_COMMAND"

typedef enum { SHELL_POWERSHELL, SHELL_CMD, SHELL_POSIX } ShellKind;

typedef struct Shell {
    ShellKind kind;
    char program[SHELL_PATH_MAX]; /* path, or a name looked up in PATH */
    char name[32];                /* for the banner: "PowerShell 7", "cmd", "bash" */
} Shell;

/* Recognizes a shell by its executable, a path or a name in any case, with or
 * without ".exe": pwsh, powershell, cmd, and the sh-like shells (bash, sh, zsh,
 * dash, ksh, mksh, ash, fish). Returns 0 for anything else. */
int shell_classify(const char *program, Shell *out);

/* The shell for "!":
 *  1. `choice` (--shell), a name looked up in PATH or a path;
 *  2. else `parent` (the program that started the chat), when it is a shell;
 *  3. else the system's: PowerShell 7, Windows PowerShell or cmd on Windows,
 *     $SHELL or /bin/sh elsewhere.
 * Returns 0 with a reason in Portuguese. */
int shell_choose(const char *choice, const char *parent, Shell *out, char *err, size_t cap);

/* The Windows command line that runs `command` (UTF-8):
 *  - PowerShell: -NoLogo -NoProfile -Command "command", quoted the way
 *    CommandLineToArgvW splits it back;
 *  - cmd: /d /s /c "command", which cmd reads as typed;
 *  - sh-like: -c "eval \"$SMOLMOE_SHELL_COMMAND\"", the command in that variable.
 * Returns its length, or 0 when it does not fit or `command` is not UTF-8. */
size_t shell_command_line(const Shell *sh, const char *command, char *out, size_t cap);

/* Whether `command` is a plain "cd DIR" (or "chdir", or cmd's "cd /d DIR"),
 * which must change the chat's own directory to last. *target gets DIR with
 * quotes removed, "~" expanded from `home` and, with `windows`, "/c/x" read as
 * "C:/x"; "" for a bare "cd". A "cd" with shell operators (cd x && ls) is left
 * to the shell. Returns 1, 0 (not a plain cd) or -1 (DIR does not fit). */
int shell_cd_target(const char *command, const char *home, int windows, char *target, size_t cap);

/* Runs the text after "!" (a plain cd changes the chat's directory) and reports
 * the exit code or the error on stderr. `null_stdin` when input is piped. */
void shell_execute(const Shell *sh, const char *command, int null_stdin);

#endif
