/* Test helper (Windows only): runs a program in a new, hidden console and types
 * into it, so tests can tell a paste from typing the way a person would do it.
 *
 *   console-input.exe STEP... -- PROGRAM [ARGS...]
 *
 * Each STEP is sent as console key presses. Everything in one step arrives at
 * once, as a paste does; a step "+N" waits N milliseconds, as a person pauses
 * between lines, and "?TEXT" waits until TEXT shows on the screen (the program
 * is ready). "\n" in a step is the Enter key. Exits with the program's exit
 * code, 94 if TEXT never showed, or 99 if it did not finish within 60 s. */
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static wchar_t *widen(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    wchar_t *w = (wchar_t *)malloc((size_t)n * sizeof(*w));
    if (!w) exit(98);
    MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}

/* Key down and key up for every character; "\n" is Enter. */
static void type_text(HANDLE in, const char *text) {
    wchar_t *w = widen(text);
    size_t n = wcslen(w), i, k = 0;
    INPUT_RECORD *rec = (INPUT_RECORD *)calloc(2 * n + 1, sizeof(*rec));
    DWORD written;
    if (!rec) exit(98);
    for (i = 0; i < n; ++i) {
        int down;
        for (down = 1; down >= 0; --down, ++k) {
            KEY_EVENT_RECORD *key = &rec[k].Event.KeyEvent;
            rec[k].EventType = KEY_EVENT;
            key->bKeyDown = down;
            key->wRepeatCount = 1;
            key->wVirtualKeyCode = w[i] == L'\n' ? VK_RETURN : 0;
            key->uChar.UnicodeChar = w[i] == L'\n' ? L'\r' : w[i];
        }
    }
    if (!WriteConsoleInputW(in, rec, (DWORD)k, &written) || written != k) exit(97);
    free(rec);
    free(w);
}

/* Waits up to 60 s for `text` anywhere in the console's screen buffer. */
static int wait_for_text(const char *text) {
    wchar_t *want = widen(text), *screen = NULL;
    HANDLE out = CreateFileW(L"CONOUT$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                             OPEN_EXISTING, 0, NULL);
    int tries, found = 0;
    for (tries = 0; tries < 1200 && !found && out != INVALID_HANDLE_VALUE; ++tries) {
        CONSOLE_SCREEN_BUFFER_INFO info;
        COORD origin = {0, 0};
        DWORD size, got = 0;
        if (GetConsoleScreenBufferInfo(out, &info)) {
            size = (DWORD)info.dwSize.X * (DWORD)info.dwSize.Y;
            screen = (wchar_t *)realloc(screen, (size + 1) * sizeof(*screen));
            if (!screen) exit(98);
            ReadConsoleOutputCharacterW(out, screen, size, origin, &got);
            screen[got] = 0;
            found = wcsstr(screen, want) != NULL;
        }
        if (!found) Sleep(50);
    }
    if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
    free(screen);
    free(want);
    return found;
}

/* The program under test: ended on every way out, so no hidden console is left running. */
static HANDLE child;

static void stop_child(void) {
    DWORD code;
    if (child && GetExitCodeProcess(child, &code) && code == STILL_ACTIVE) TerminateProcess(child, 99);
    if (child) CloseHandle(child);
    child = NULL;
}

/* argv[first..] joined into one command line, each argument quoted (the
 * arguments here never hold quotes). */
static wchar_t *command_line(int argc, char **argv, int first) {
    size_t cap = 16, n = 0;
    int a;
    wchar_t *line;
    for (a = first; a < argc; ++a) cap += 3 * strlen(argv[a]) + 3;
    line = (wchar_t *)calloc(cap, sizeof(*line));
    if (!line) exit(98);
    for (a = first; a < argc; ++a) {
        wchar_t *w = widen(argv[a]);
        n += (size_t)swprintf(line + n, cap - n, L"%ls\"%ls\"", a > first ? L" " : L"", w);
        free(w);
    }
    return line;
}

int main(int argc, char **argv) {
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    HANDLE in;
    DWORD code = 99;
    int sep, a, tries;
    for (sep = 1; sep < argc && strcmp(argv[sep], "--"); ++sep) {}
    if (sep + 1 >= argc) {
        fputs("usage: console-input STEP... -- PROGRAM [ARGS...]\n", stderr);
        return 2;
    }
    memset(&si, 0, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    if (!CreateProcessW(NULL, command_line(argc, argv, sep + 1), NULL, NULL, FALSE, CREATE_NEW_CONSOLE, NULL, NULL,
                        &si, &pi)) {
        fprintf(stderr, "cannot start %s (%lu)\n", argv[sep + 1], (unsigned long)GetLastError());
        return 2;
    }
    child = pi.hProcess;
    atexit(stop_child);
    CloseHandle(pi.hThread);
    FreeConsole();
    for (tries = 0; !AttachConsole(pi.dwProcessId); ++tries) {
        if (tries == 100) return 96;
        Sleep(20);
    }
    in = CreateFileW(L"CONIN$", GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                     OPEN_EXISTING, 0, NULL);
    if (in == INVALID_HANDLE_VALUE) return 95;
    for (a = 1; a < sep; ++a) {
        if (argv[a][0] == '+') Sleep((DWORD)atoi(argv[a] + 1));
        else if (argv[a][0] == '?') {
            if (!wait_for_text(argv[a] + 1)) return 94;
        } else type_text(in, argv[a]);
    }
    CloseHandle(in);
    if (WaitForSingleObject(child, 60000) == WAIT_OBJECT_0) GetExitCodeProcess(child, &code);
    return (int)code;
}
