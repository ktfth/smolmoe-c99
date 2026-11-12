/* The command check: after a chat reply, the commands in it are looked up in
 * this machine's PowerShell, which only parses them (nothing in the reply
 * runs). It reports the cmdlets and parameters that do not exist here and the
 * real cmdlets that change the system (those that take -WhatIf).
 *
 * It never says a command is right. A real command can still do the wrong
 * thing (stop a service instead of restarting it, delete the wrong folder);
 * a cmdlet missing here can exist on the machine being fixed, and the other
 * way round; and a block written for bash or cmd is named, not checked. The
 * notices say so every time. */
#ifndef SMOLMOE_CMDCHECK_H
#define SMOLMOE_CMDCHECK_H

#include <stddef.h>

#include "shell.h"

#define CMDCHECK_FOREIGN_MAX 64
#define CMDCHECK_NOTICE_MAX 4096

typedef struct CmdExtract {
    int checked;                        /* snippets put in the request */
    char foreign[CMDCHECK_FOREIGN_MAX]; /* tags of the blocks for other shells, each once: "bash, cmd" */
} CmdExtract;

/* The request for the checker: the reply's commands as snippets joined by
 * 0x1E, each starting with 'B' (a code block: every command in it is looked
 * up) or 'I' (inline code or prose: only Verb-Noun names are). Snippets come
 * from:
 *  - ``` blocks, untagged or tagged powershell/ps1/pwsh/shell/console; blocks
 *    tagged bash/sh/zsh/fish/cmd/bat go to x->foreign instead, other
 *    languages are skipped;
 *  - `inline code` that starts with Verb-Noun;
 *  - prose: an approved PowerShell verb, a dash and a capitalized noun
 *    (Get-Process), up to the end of the line or sentence or the next one.
 * A snippet that does not fit is dropped whole. Returns the length. */
size_t cmdcheck_extract(const char *reply, size_t n, char *out, size_t cap, CmdExtract *x);

/* The notice for the checker's findings, one per line:
 *   N<TAB>name<TAB>similar names      the command is not found here
 *   P<TAB>cmdlet<TAB>param<TAB>similar  the cmdlet has no such parameter
 *   W<TAB>cmdlet<TAB>impact           a real cmdlet that changes the system
 * Malformed lines are skipped and control bytes shown as '?'. Empty when
 * nothing was checked or named. */
void cmdcheck_format(const char *findings, size_t n, const CmdExtract *x, const char *shell_name, char *out,
                     size_t cap);

/* Standard base64 with padding, NUL-terminated; 0 when it does not fit. */
size_t cmdcheck_base64(const void *data, size_t n, char *out, size_t cap);

/* The PowerShell the check uses: `choice` (--shell), which must be one; else
 * the chat's shell when it is one; else pwsh, else powershell, from PATH.
 * Returns 0 with a reason in Portuguese. */
int cmdcheck_shell(const char *choice, const Shell *chat, Shell *out, char *err, size_t cap);

/* A PowerShell kept running in the background. With `warm` it starts here and
 * fills its command cache while the user types, so each reply is answered in
 * a fraction of a second; without, it starts at the first reply that has a
 * command to look up. NULL with a reason when it cannot start. */
typedef struct CmdChecker CmdChecker;
CmdChecker *cmdcheck_start(const Shell *ps, int warm, char *err, size_t cap);

/* Checks one reply and writes the notice to out (possibly empty); returns 1
 * when there is one. A PowerShell that left or stopped answering is started
 * again for the next reply. */
int cmdcheck_reply(CmdChecker *c, const char *reply, size_t n, char *out, size_t cap);

void cmdcheck_stop(CmdChecker *c);

#endif
