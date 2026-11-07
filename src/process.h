/* Child processes and the user's terminal, for "!comando" in the chat. Like
 * platform.c, this is the operating-system layer: Win32 and POSIX (WebAssembly
 * has neither processes nor terminal control: "!" is refused there). */
#ifndef SMOLMOE_PROCESS_H
#define SMOLMOE_PROCESS_H

#include <stddef.h>

/* Whether stdin is a terminal a person types into: a console, a POSIX tty, or
 * the pipe of an MSYS/Cygwin terminal (Git Bash in mintty). */
int stdin_is_terminal(void);
/* Reads a terminal stdin byte by byte, so input_pending() sees every line that
 * stdio would otherwise have buffered. Call before the first read. */
void terminal_input_unbuffered(void);
/* Whether more input arrives on the terminal within `ms` milliseconds. Lines
 * that arrive together came from a paste: a person types one at a time. Always
 * 0 when stdin is not a terminal. */
int input_pending(unsigned ms);
/* POSIX terminals: asks the terminal to wrap every paste in ESC[200~ ... ESC[201~
 * (bracketed paste), which tells a paste from typing even for a single line, and
 * stops echoing control characters so the markers do not show. Undone at exit,
 * on SIGINT/SIGTERM/SIGHUP and while a child program runs. Returns 1 when on;
 * always 0 on Windows, whose console has no such markers in line input. */
int bracketed_paste_start(void);

/* Path of the program that started this process (usually the user's shell);
 * 0 when it cannot be known. */
int parent_executable(char *out, size_t cap);
/* First `name` found in the PATH directories, never in the current directory. */
int find_program(const char *name, char *out, size_t cap);
/* The user's home directory (USERPROFILE on Windows, HOME elsewhere), UTF-8. */
int home_directory(char *out, size_t cap);
int current_directory(char *out, size_t cap);
int change_directory(const char *path, char *err, size_t errlen);

/* Runs a program on this terminal and waits for it. Its stdout and stderr are
 * ours; its stdin too, or the null device with `null_stdin`. While it runs,
 * Ctrl+C stops it and not this process.
 *  - Windows: `command_line` (UTF-8) goes to CreateProcess as it is, and
 *    `env_name`=`env_value` is set for the child (also listed in WSLENV, so
 *    WSL's bash sees it); `argv` is ignored.
 *  - POSIX: `argv` runs, argv[0] looked up in PATH; the rest is ignored.
 * Returns 1 with the exit code in *status, or 0 with a reason in err. */
int run_program(const char *const *argv, const char *command_line, const char *env_name, const char *env_value,
                int null_stdin, int *status, char *err, size_t errlen);

#endif
