/* The command check: after a chat reply, the commands in it are looked up on
 * this machine, and nothing in the reply runs.
 *  - PowerShell (the chat's shell is one, or Windows): PowerShell parses the
 *    snippets and Get-Command answers for each name. Reported: cmdlets and
 *    parameters that do not exist here, and real cmdlets that change the
 *    system (those that take -WhatIf).
 *  - POSIX sh (the chat's shell is bash/sh/zsh, or Linux and macOS without
 *    one): the commands are split here (cmdcheck_sh.c), and sh only runs
 *    `command -v`, `man -w` and reads the manual page. Reported: programs not
 *    on PATH, options and subcommands the manual here does not cite, programs
 *    without a manual (options not checked), and commands that change the
 *    system, from a table. No program from the reply runs, not even --help.
 *
 * It never says a command is right. A real command can still do the wrong
 * thing (stop a service instead of restarting it, delete the wrong folder);
 * what is missing here can exist on the machine being fixed, and the other way
 * round; and a block for the other kind of shell is named, not checked. The
 * notices say so every time. */
#ifndef SMOLMOE_CMDCHECK_H
#define SMOLMOE_CMDCHECK_H

#include <stddef.h>

#include "shell.h"

#define CMDCHECK_FOREIGN_MAX 64
#define CMDCHECK_CHANGES_MAX 1024
#define CMDCHECK_NOTICE_MAX 4096

typedef enum { CMDCHECK_POWERSHELL, CMDCHECK_POSIX } CmdCheckKind;

typedef struct CmdExtract {
    CmdCheckKind kind;
    int checked;                          /* snippets (PowerShell) or command lines (POSIX) in the request */
    char foreign[CMDCHECK_FOREIGN_MAX];   /* tags of the blocks for the other kind of shell, each once */
    char changes[CMDCHECK_CHANGES_MAX];   /* POSIX: "W" findings for commands that change the system */
} CmdExtract;

/* The request for the checker, from:
 *  - ``` blocks, untagged or tagged for this kind of shell (powershell, ps1,
 *    pwsh / bash, sh, zsh; shell, console and terminal for both); blocks for
 *    the other kind (and cmd/bat) go to x->foreign, other languages are
 *    skipped;
 *  - `inline code`: starting with Verb-Noun (PowerShell), or a lowercase
 *    command name followed by more words (POSIX: `sudo systemctl restart x`);
 *  - PowerShell only, prose: an approved verb, a dash and a capitalized noun
 *    (Get-Process), up to the end of the line or sentence or the next one.
 * PowerShell: snippets joined by 0x1E, each starting with 'B' (a block: every
 * command is looked up) or 'I' (only Verb-Noun names are). POSIX: one line
 * per command, cmdcheck_sh_commands. A snippet that does not fit is dropped
 * whole. Returns the length. */
size_t cmdcheck_extract(const char *reply, size_t n, CmdCheckKind kind, char *out, size_t cap, CmdExtract *x);

/* POSIX: the commands of a shell snippet, one line each,
 * "name<TAB>subcommand<TAB>option<TAB>option...", with "." for no subcommand
 * (only programs known to take one get it). Pipes, &&, ||, ;, keywords (if,
 * then, do...), NAME=value prefixes, sudo/doas/env/time/nohup/nice/exec,
 * redirections, here-documents, comments and a leading "$ " prompt are
 * understood; words with $(...), backticks or quotes as names, and paths
 * (./configure), are skipped. Options lose any "=value" and keep only
 * [A-Za-z0-9-]. Commands that change the system add "W<TAB>name<TAB>-" lines
 * to `changes`. Repeated lines are dropped. Appends to out at *len; returns
 * the number of lines added. */
int cmdcheck_sh_commands(const char *text, size_t n, char *out, size_t cap, size_t *len, char *changes,
                         size_t changes_cap);

/* The notice for the findings, one per line:
 *   N<TAB>name<TAB>similar names        not found here
 *   P<TAB>cmdlet<TAB>param<TAB>similar  the cmdlet has no such parameter (PowerShell)
 *   O<TAB>program<TAB>option            the manual here does not cite the option (POSIX)
 *   S<TAB>program<TAB>subcommand        the manual here does not cite the subcommand (POSIX)
 *   M<TAB>program                       no manual here: options not checked (POSIX)
 *   W<TAB>name<TAB>impact               changes the system ("-": no -WhatIf, POSIX)
 * Malformed lines are skipped and control bytes shown as '?'. Empty when
 * nothing was checked or named. */
void cmdcheck_format(const char *findings, size_t n, const CmdExtract *x, const char *shell_name, char *out,
                     size_t cap);

/* Standard base64 with padding, NUL-terminated; 0 when it does not fit. */
size_t cmdcheck_base64(const void *data, size_t n, char *out, size_t cap);

/* The shell the check uses, and so its kind:
 *  - `choice` (--shell): a PowerShell or a POSIX shell; cmd is refused;
 *  - else the chat's shell, when it is a PowerShell or a POSIX shell;
 *  - else, on Windows, pwsh or powershell from PATH; elsewhere sh.
 * A POSIX check runs in sh (from PATH) when there is one, else in the chosen
 * shell when it is sh, bash, dash, ksh or ash. Returns 0 with a reason in
 * Portuguese. */
int cmdcheck_shell(const char *choice, const Shell *chat, Shell *out, char *err, size_t cap);

/* The checker for a shell from cmdcheck_shell. A PowerShell is kept running
 * in the background: with `warm` it starts here and fills its command cache
 * while the user types, so each reply is answered in a fraction of a second;
 * without, it starts at the first reply that needs it. A POSIX sh starts for
 * each reply (a few milliseconds). NULL with a reason when it cannot start. */
typedef struct CmdChecker CmdChecker;
CmdChecker *cmdcheck_start(const Shell *sh, int warm, char *err, size_t cap);

/* Checks one reply and writes the notice to out (possibly empty); returns 1
 * when there is one. A PowerShell that left or stopped answering is started
 * again for the next reply. */
int cmdcheck_reply(CmdChecker *c, const char *reply, size_t n, char *out, size_t cap);

void cmdcheck_stop(CmdChecker *c);

/* The sh script of the POSIX check (cmdcheck_sh.c), with its line endings. */
size_t cmdcheck_sh_script(char *out, size_t cap);

/* The index carried in the binary for machines without manual pages
 * (containers, minimal installs): the options and subcommands that the manual
 * pages of common maintenance programs cite, generated by
 * scripts/cmdcheck/gen-index.ts. cmdcheck_index_source names where they come
 * from ("Ubuntu 24.04"). An option passes when the index cites it, or, for a
 * single dash and several letters (-la), when it cites each letter up to the
 * first digit (-n5). */
const char *cmdcheck_index_source(void);
int cmdcheck_index_has(const char *program);
int cmdcheck_index_option(const char *program, const char *option);
int cmdcheck_index_sub(const char *program, const char *subcommand);

#endif
