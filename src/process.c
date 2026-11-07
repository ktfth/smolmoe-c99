/* Child processes and the user's terminal: see process.h. */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#  define _DARWIN_C_SOURCE 1
#endif

#include "process.h"
#include "platform.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <tlhelp32.h>
#else
#  include <fcntl.h>
#  include <poll.h>
#  include <signal.h>
#  include <spawn.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  include <termios.h>
#  include <unistd.h>
extern char **environ;
#endif

#if defined(_WIN32)

#define WIDE_PATH 32768

/* NULL when `s` is not valid UTF-8 (user input must not end the program). */
static wchar_t *to_wide(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    wchar_t *w;
    if (n <= 0 || !(w = (wchar_t *)malloc((size_t)n * sizeof(*w)))) return NULL;
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n);
    return w;
}

static int to_utf8(const wchar_t *w, char *out, size_t cap) {
    return WideCharToMultiByte(CP_UTF8, 0, w, -1, out, (int)cap, NULL, NULL) > 0;
}

/* "what: <system message>", in the system's language. */
static void last_error(const char *what, char *err, size_t errlen) {
    wchar_t wide[256];
    char msg[512];
    size_t n;
    DWORD got = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, GetLastError(), 0,
                               wide, (DWORD)(sizeof(wide) / sizeof(*wide)), NULL);
    if (!got || !to_utf8(wide, msg, sizeof(msg))) snprintf(msg, sizeof(msg), "erro %lu", (unsigned long)GetLastError());
    for (n = strlen(msg); n && (msg[n - 1] == '\n' || msg[n - 1] == '\r' || msg[n - 1] == ' ' || msg[n - 1] == '.');)
        msg[--n] = 0;
    snprintf(err, errlen, "%s: %s", what, msg);
}

static HANDLE input_handle(void) { return GetStdHandle(STD_INPUT_HANDLE); }

static int console_input(void) {
    DWORD mode;
    return GetConsoleMode(input_handle(), &mode) != 0;
}

/* Git Bash in mintty hands native programs a pipe named like
 * \msys-1888ae32e00d56aa-pty0-from-master: a terminal all the same. */
static int msys_terminal(HANDLE h) {
    union {
        FILE_NAME_INFO info;
        char bytes[sizeof(FILE_NAME_INFO) + MAX_PATH * sizeof(WCHAR)];
    } u;
    wchar_t name[MAX_PATH + 1];
    size_t chars;
    if (!h || h == INVALID_HANDLE_VALUE || GetFileType(h) != FILE_TYPE_PIPE) return 0;
    if (!GetFileInformationByHandleEx(h, FileNameInfo, &u, sizeof(u))) return 0;
    chars = u.info.FileNameLength / sizeof(WCHAR);
    if (chars > MAX_PATH) return 0;
    memcpy(name, u.info.FileName, chars * sizeof(WCHAR));
    name[chars] = 0;
    return (wcsstr(name, L"msys-") || wcsstr(name, L"cygwin-")) && wcsstr(name, L"-pty") &&
           wcsstr(name, L"-from-master");
}

int stdin_is_terminal(void) {
    return console_input() || msys_terminal(input_handle());
}

void terminal_input_unbuffered(void) {
    /* The console is read with ReadConsoleW, which stdio does not buffer. */
    if (!console_input() && msys_terminal(input_handle())) setvbuf(stdin, NULL, _IONBF, 0);
}

/* Whether a key press that types a character waits in the console. Other events
 * (key releases, focus, mouse) are dropped, as ReadConsoleW would drop them. */
static int console_key_waiting(HANDLE in) {
    INPUT_RECORD rec[128];
    DWORD n = 0, i;
    if (!PeekConsoleInputW(in, rec, (DWORD)(sizeof(rec) / sizeof(*rec)), &n) || !n) return 0;
    for (i = 0; i < n; ++i)
        if (rec[i].EventType == KEY_EVENT && rec[i].Event.KeyEvent.bKeyDown && rec[i].Event.KeyEvent.uChar.UnicodeChar)
            return 1;
    if (n == sizeof(rec) / sizeof(*rec)) return 1; /* a long burst of events: count it as input */
    ReadConsoleInputW(in, rec, n, &n);
    return 0;
}

int input_pending(unsigned ms) {
    HANDLE in = input_handle();
    DWORD start = GetTickCount(), avail;
    if (console_input()) {
        for (;;) {
            DWORD elapsed;
            if (console_key_waiting(in)) return 1;
            elapsed = GetTickCount() - start;
            if (elapsed >= ms || WaitForSingleObject(in, ms - elapsed) != WAIT_OBJECT_0) return console_key_waiting(in);
        }
    }
    if (!msys_terminal(in)) return 0;
    for (;;) {
        if (PeekNamedPipe(in, NULL, 0, NULL, &avail, NULL) && avail) return 1;
        if (GetTickCount() - start >= ms) return 0;
        Sleep(2);
    }
}

int bracketed_paste_start(void) {
    return 0;
}

int parent_executable(char *out, size_t cap) {
    DWORD self = GetCurrentProcessId(), parent = 0, size = WIDE_PATH;
    PROCESSENTRY32W entry;
    FILETIME parent_start, self_start, unused[3];
    wchar_t *path;
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0), h;
    int ok;
    if (snap == INVALID_HANDLE_VALUE) return 0;
    entry.dwSize = sizeof(entry);
    if (Process32FirstW(snap, &entry)) {
        do {
            if (entry.th32ProcessID == self) parent = entry.th32ParentProcessID;
        } while (!parent && Process32NextW(snap, &entry));
    }
    CloseHandle(snap);
    if (!parent || !(h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, parent))) return 0;
    path = (wchar_t *)malloc(WIDE_PATH * sizeof(*path));
    /* The parent may have exited and its ID been reused: it must be older than us. */
    ok = path && GetProcessTimes(h, &parent_start, &unused[0], &unused[1], &unused[2]) &&
         GetProcessTimes(GetCurrentProcess(), &self_start, &unused[0], &unused[1], &unused[2]) &&
         CompareFileTime(&parent_start, &self_start) <= 0 && QueryFullProcessImageNameW(h, 0, path, &size) &&
         to_utf8(path, out, cap);
    CloseHandle(h);
    free(path);
    return ok;
}

int find_program(const char *name, char *out, size_t cap) {
    wchar_t *paths = (wchar_t *)malloc(WIDE_PATH * sizeof(wchar_t)), *wname = to_wide(name), *dir, *next;
    wchar_t candidate[MAX_PATH * 2];
    DWORD n = paths ? GetEnvironmentVariableW(L"PATH", paths, WIDE_PATH) : 0; /* too long: the size it needs */
    int found = 0;
    if (wname && n && n < WIDE_PATH) {
        for (dir = paths; dir && !found; dir = next) {
            DWORD attributes;
            size_t len;
            next = wcschr(dir, L';');
            if (next) *next++ = 0;
            if (*dir == L'"') ++dir;
            len = wcslen(dir);
            if (len && dir[len - 1] == L'"') dir[--len] = 0;
            /* Only absolute entries: "." or "bin" would search the current directory. */
            if (!((len >= 3 && dir[1] == L':' && (dir[2] == L'\\' || dir[2] == L'/')) ||
                  (len >= 2 && dir[0] == L'\\' && dir[1] == L'\\')))
                continue;
            if (len + wcslen(wname) + 2 > sizeof(candidate) / sizeof(*candidate)) continue;
            wcscpy(candidate, dir);
            if (dir[len - 1] != L'\\' && dir[len - 1] != L'/') wcscat(candidate, L"\\");
            wcscat(candidate, wname);
            attributes = GetFileAttributesW(candidate);
            found = attributes != INVALID_FILE_ATTRIBUTES && !(attributes & FILE_ATTRIBUTE_DIRECTORY) &&
                    to_utf8(candidate, out, cap);
        }
    }
    free(paths);
    free(wname);
    return found;
}

int home_directory(char *out, size_t cap) {
    wchar_t buf[MAX_PATH * 2];
    DWORD n = GetEnvironmentVariableW(L"USERPROFILE", buf, (DWORD)(sizeof(buf) / sizeof(*buf)));
    return n && n < sizeof(buf) / sizeof(*buf) && to_utf8(buf, out, cap);
}

int current_directory(char *out, size_t cap) {
    wchar_t buf[MAX_PATH * 2];
    DWORD n = GetCurrentDirectoryW((DWORD)(sizeof(buf) / sizeof(*buf)), buf);
    return n && n < sizeof(buf) / sizeof(*buf) && to_utf8(buf, out, cap);
}

int change_directory(const char *path, char *err, size_t errlen) {
    wchar_t *w = to_wide(path);
    int ok = w && SetCurrentDirectoryW(w);
    if (!w) snprintf(err, errlen, "o caminho não é UTF-8 válido");
    else if (!ok) last_error(path, err, errlen);
    free(w);
    return ok;
}

static BOOL WINAPI ignore_interrupt(DWORD type) {
    return type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT;
}

/* An inheritable copy of a standard handle, or NULL. */
static HANDLE inheritable(HANDLE h) {
    HANDLE copy = NULL;
    if (!h || h == INVALID_HANDLE_VALUE) return NULL;
    if (!DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &copy, 0, TRUE, DUPLICATE_SAME_ACCESS))
        return NULL;
    return copy;
}

/* Sets `name` for the child and adds it to WSLENV; *saved keeps WSLENV as it was.
 * On failure nothing is left changed. */
static int set_child_env(const char *name, const char *value, wchar_t **saved) {
    wchar_t *wname = to_wide(name), *wvalue = to_wide(value), *wsl = NULL;
    DWORD n = GetEnvironmentVariableW(L"WSLENV", NULL, 0);
    int ok = wname && wvalue;
    *saved = NULL;
    if (ok && n) {
        *saved = (wchar_t *)malloc(n * sizeof(wchar_t));
        ok = *saved && GetEnvironmentVariableW(L"WSLENV", *saved, n) < n;
    }
    if (ok) ok = (wsl = (wchar_t *)malloc((n + wcslen(wname) + 2) * sizeof(wchar_t))) != NULL;
    if (ok) {
        wcscpy(wsl, *saved && **saved ? *saved : L"");
        if (*saved && **saved) wcscat(wsl, L":");
        wcscat(wsl, wname);
        ok = SetEnvironmentVariableW(wname, wvalue) && SetEnvironmentVariableW(L"WSLENV", wsl);
        if (!ok) SetEnvironmentVariableW(wname, NULL);
    }
    if (!ok) {
        free(*saved);
        *saved = NULL;
    }
    free(wsl);
    free(wname);
    free(wvalue);
    return ok;
}

static void clear_child_env(const char *name, wchar_t *saved) {
    wchar_t *wname = to_wide(name);
    if (wname) SetEnvironmentVariableW(wname, NULL);
    SetEnvironmentVariableW(L"WSLENV", saved);
    free(wname);
    free(saved);
}

/* The console as the chat left it: a child may change modes (echo, line input,
 * VT input) or code pages (chcp) and not put them back. */
typedef struct ConsoleState {
    DWORD mode[3];
    int has_mode[3];
    UINT output_cp, input_cp;
} ConsoleState;

static const DWORD STD_HANDLES[3] = {STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};

static void console_save(ConsoleState *s) {
    int i;
    for (i = 0; i < 3; ++i) s->has_mode[i] = GetConsoleMode(GetStdHandle(STD_HANDLES[i]), &s->mode[i]) != 0;
    s->output_cp = GetConsoleOutputCP();
    s->input_cp = GetConsoleCP();
}

static void console_restore(const ConsoleState *s) {
    int i;
    for (i = 0; i < 3; ++i)
        if (s->has_mode[i]) SetConsoleMode(GetStdHandle(STD_HANDLES[i]), s->mode[i]);
    if (s->output_cp) SetConsoleOutputCP(s->output_cp);
    if (s->input_cp) SetConsoleCP(s->input_cp);
}

/* Inheritable copies of our standard handles for the child; stdin may be NUL. */
static void child_handles(int null_stdin, HANDLE std[3]) {
    SECURITY_ATTRIBUTES sa;
    int i;
    sa.nLength = sizeof(sa);
    sa.lpSecurityDescriptor = NULL;
    sa.bInheritHandle = TRUE;
    for (i = 0; i < 3; ++i) std[i] = inheritable(GetStdHandle(STD_HANDLES[i]));
    if (null_stdin) {
        if (std[0]) CloseHandle(std[0]);
        std[0] = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, NULL);
        if (std[0] == INVALID_HANDLE_VALUE) std[0] = NULL;
    }
}

static int wait_child(const PROCESS_INFORMATION *pi) {
    DWORD code = 0;
    WaitForSingleObject(pi->hProcess, INFINITE);
    GetExitCodeProcess(pi->hProcess, &code);
    CloseHandle(pi->hThread);
    CloseHandle(pi->hProcess);
    return (int)code;
}

int run_program(const char *const *argv, const char *command_line, const char *env_name, const char *env_value,
                int null_stdin, int *status, char *err, size_t errlen) {
    ConsoleState console;
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    HANDLE std[3];
    wchar_t *cmd = to_wide(command_line), *saved_wslenv = NULL;
    UINT user_cp = (UINT)console_original_codepage();
    BOOL ok;
    int i;
    (void)argv;
    if (!cmd) {
        snprintf(err, errlen, "o comando não é UTF-8 válido");
        return 0;
    }
    if (env_name && !set_child_env(env_name, env_value, &saved_wslenv)) {
        free(cmd);
        last_error("não consegui preparar o ambiente do shell", err, errlen);
        return 0;
    }
    child_handles(null_stdin, std);
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = std[0];
    si.hStdOutput = std[1];
    si.hStdError = std[2];
    fflush(stdout);
    fflush(stderr);
    console_save(&console);
    if (user_cp) SetConsoleOutputCP(user_cp); /* the child prints in the user's code page, not the chat's UTF-8 */
    SetConsoleCtrlHandler(ignore_interrupt, TRUE); /* before the child exists: Ctrl+C never ends the chat */
    ok = CreateProcessW(NULL, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    if (!ok) last_error("não consegui iniciar o shell", err, errlen);
    for (i = 0; i < 3; ++i)
        if (std[i]) CloseHandle(std[i]);
    free(cmd);
    if (env_name) clear_child_env(env_name, saved_wslenv);
    if (ok) *status = wait_child(&pi);
    SetConsoleCtrlHandler(ignore_interrupt, FALSE);
    console_restore(&console);
    return ok != 0;
}

#else /* POSIX */

int stdin_is_terminal(void) {
    return isatty(STDIN_FILENO);
}

void terminal_input_unbuffered(void) {
    if (stdin_is_terminal()) setvbuf(stdin, NULL, _IONBF, 0);
}

int input_pending(unsigned ms) {
    struct pollfd p;
    int n;
    if (!stdin_is_terminal()) return 0;
    p.fd = STDIN_FILENO;
    p.events = POLLIN;
    p.revents = 0;
    while ((n = poll(&p, 1, (int)ms)) < 0 && errno == EINTR) {}
    return n > 0 && (p.revents & POLLIN);
}

static struct termios saved_tty;
static volatile sig_atomic_t bracketed;

/* Async-signal-safe: it also runs from the signal handler. */
static void paste_markers_off(void) {
    if (!bracketed) return;
    if (write(STDERR_FILENO, "\033[?2004l", 8) < 0) {}
    tcsetattr(STDIN_FILENO, TCSANOW, &saved_tty);
}

static void paste_markers_on(void) {
    struct termios t = saved_tty;
#if defined(ECHOCTL)
    t.c_lflag &= ~(tcflag_t)ECHOCTL; /* markers echo raw (terminals ignore them), not as ^[[200~ */
#endif
    tcsetattr(STDIN_FILENO, TCSANOW, &t);
    if (write(STDERR_FILENO, "\033[?2004h", 8) < 0) {}
}

static void paste_markers_exit(void) {
    paste_markers_off();
    bracketed = 0;
}

static void paste_markers_signal(int sig) {
    paste_markers_exit();
    signal(sig, SIG_DFL);
    raise(sig);
}

int bracketed_paste_start(void) {
    static const int SIGNALS[] = {SIGINT, SIGTERM, SIGHUP};
    size_t i;
    if (bracketed) return 1;
    if (!isatty(STDIN_FILENO) || !isatty(STDERR_FILENO) || tcgetattr(STDIN_FILENO, &saved_tty)) return 0;
    bracketed = 1;
    paste_markers_on();
    atexit(paste_markers_exit);
    for (i = 0; i < sizeof(SIGNALS) / sizeof(SIGNALS[0]); ++i) {
        struct sigaction old, sa;
        if (sigaction(SIGNALS[i], NULL, &old) || old.sa_handler != SIG_DFL) continue; /* an ignored one stays so */
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = paste_markers_signal;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGNALS[i], &sa, NULL);
    }
    return 1;
}

int parent_executable(char *out, size_t cap) {
#if defined(__linux__)
    char link[64];
    ssize_t n;
    snprintf(link, sizeof(link), "/proc/%ld/exe", (long)getppid());
    n = readlink(link, out, cap - 1);
    if (n <= 0) return 0;
    out[n] = 0;
    return 1;
#else
    (void)out;
    (void)cap;
    return 0;
#endif
}

int find_program(const char *name, char *out, size_t cap) {
    const char *path = getenv("PATH"), *dir, *end;
    struct stat st;
    if (!path) return 0;
    for (dir = path; *dir; dir = *end ? end + 1 : end) {
        size_t len;
        end = strchr(dir, ':');
        if (!end) end = dir + strlen(dir);
        len = (size_t)(end - dir);
        /* Only absolute entries: an empty one, "." or "bin" would search the current directory. */
        if (dir[0] != '/' || len + strlen(name) + 2 > cap) continue;
        memcpy(out, dir, len);
        out[len] = '/';
        strcpy(out + len + 1, name);
        if (!stat(out, &st) && S_ISREG(st.st_mode) && !access(out, X_OK)) return 1;
    }
    return 0;
}

int home_directory(char *out, size_t cap) {
    const char *home = getenv("HOME");
    if (!home || !*home || strlen(home) >= cap) return 0;
    strcpy(out, home);
    return 1;
}

int current_directory(char *out, size_t cap) {
    return getcwd(out, cap) != NULL;
}

int change_directory(const char *path, char *err, size_t errlen) {
    if (!chdir(path)) return 1;
    snprintf(err, errlen, "%s: %s", path, strerror(errno));
    return 0;
}

int run_program(const char *const *argv, const char *command_line, const char *env_name, const char *env_value,
                int null_stdin, int *status, char *err, size_t errlen) {
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    struct sigaction ignore, old_int, old_quit;
    sigset_t defaults;
    pid_t pid;
    int rc, st = 0, markers = bracketed;
    (void)command_line;
    (void)env_name;
    (void)env_value;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    if (null_stdin) posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    /* Like system(): this process ignores Ctrl+C while the child runs, and the
     * child gets the default handling back. */
    sigemptyset(&defaults);
    sigaddset(&defaults, SIGINT);
    sigaddset(&defaults, SIGQUIT);
    posix_spawnattr_setsigdefault(&attr, &defaults);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSIGDEF);
    memset(&ignore, 0, sizeof(ignore));
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    fflush(stdout);
    fflush(stderr);
    if (markers) paste_markers_off(); /* the child gets the terminal as the user had it */
    sigaction(SIGINT, &ignore, &old_int);
    sigaction(SIGQUIT, &ignore, &old_quit);
    rc = posix_spawnp(&pid, argv[0], &actions, &attr, (char *const *)argv, environ);
    if (!rc) {
        while (waitpid(pid, &st, 0) < 0) {
            if (errno != EINTR) {
                rc = errno;
                break;
            }
        }
    }
    sigaction(SIGINT, &old_int, NULL);
    sigaction(SIGQUIT, &old_quit, NULL);
    if (markers) paste_markers_on();
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (rc) {
        snprintf(err, errlen, "não consegui iniciar %s: %s", argv[0], strerror(rc));
        return 0;
    }
    *status = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    return 1;
}

#endif
