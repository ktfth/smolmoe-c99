#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE 1
#  define _DARWIN_C_SOURCE 1
#  define _FILE_OFFSET_BITS 64
#endif

#include "platform.h"
#include "util.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <shellapi.h>
#  include <psapi.h>
#  include <io.h>
#  include <fcntl.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  if !defined(__wasm__)
#    include <sys/resource.h>
#  endif
#  include <sys/stat.h>
#  include <time.h>
#  include <unistd.h>
#  if defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <sys/sysctl.h>
#  endif
#endif

#define PATH_CAPACITY 4096

/* Removes a trailing "\n" or "\r\n"; returns the new length. */
static int strip_line_end(char *buf, size_t n) {
    while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    return (int)n;
}

/* Length of buf[0..n) without a trailing incomplete UTF-8 sequence. */
static size_t utf8_complete(const char *buf, size_t n) {
    size_t back;
    for (back = 1; back <= 3 && back <= n; ++back) {
        unsigned char c = (unsigned char)buf[n - back];
        size_t need;
        if ((c & 0xc0) == 0x80) continue;
        need = (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : (c & 0xf8) == 0xf0 ? 4 : 1;
        return back < need ? n - back : n;
    }
    return n;
}

/* An overlong line is cut at a character boundary and the rest of it is
 * discarded, so it never turns into extra messages. */
static int read_stream_line(char *buf, size_t cap, int *truncated) {
    size_t n;
    *truncated = 0;
    if (!fgets(buf, (int)cap, stdin)) return -1;
    n = strlen(buf);
    if (n && buf[n - 1] != '\n' && !feof(stdin)) {
        int ch;
        *truncated = 1;
        while ((ch = getchar()) != EOF && ch != '\n') {}
        n = utf8_complete(buf, n);
        buf[n] = 0;
    }
    return strip_line_end(buf, n);
}

#if defined(_WIN32)

static wchar_t *widen(const char *s) {
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, NULL, 0);
    wchar_t *w;
    if (n <= 0) die("not valid UTF-8: %s", s);
    w = (wchar_t *)malloc((size_t)n * sizeof(*w));
    if (!w) die("allocation failed");
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s, -1, w, n);
    return w;
}

static char *narrow(const wchar_t *w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char *s;
    if (n <= 0) die("cannot convert command line to UTF-8");
    s = (char *)malloc((size_t)n);
    if (!s) die("allocation failed");
    WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL);
    return s;
}

/* System message in UTF-8 (the ANSI variant would use the local code page). */
static const char *win_error(void) {
    static char buf[512];
    wchar_t wide[256];
    int len;
    DWORD n = FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, GetLastError(), 0,
                             wide, (DWORD)(sizeof(wide) / sizeof(*wide)), NULL);
    if (!n) return "unknown Windows error";
    len = WideCharToMultiByte(CP_UTF8, 0, wide, (int)n, buf, (int)sizeof(buf) - 1, NULL, NULL);
    if (len <= 0) return "unknown Windows error";
    buf[len] = 0;
    return trim(buf);
}

int try_map_file(const char *path, MappedFile *m, char *err, size_t errlen) {
    wchar_t *w = widen(path);
    HANDLE file = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    LARGE_INTEGER size;
    HANDLE mapping = NULL;
    void *view = NULL;
    free(w);
    memset(m, 0, sizeof(*m));
    if (file == INVALID_HANDLE_VALUE) {
        snprintf(err, errlen, "cannot open %s: %s", path, win_error());
        return 0;
    }
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0) snprintf(err, errlen, "empty or unreadable file: %s", path);
    else if ((uint64_t)size.QuadPart > (uint64_t)SIZE_MAX)
        snprintf(err, errlen, "file too large to map on this platform: %s", path);
    else if (!(mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL)))
        snprintf(err, errlen, "cannot map %s: %s", path, win_error());
    else if (!(view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0)))
        snprintf(err, errlen, "cannot map view of %s: %s", path, win_error());
    if (!view) {
        if (mapping) CloseHandle(mapping);
        CloseHandle(file);
        return 0;
    }
    m->data = (const unsigned char *)view;
    m->size = (uint64_t)size.QuadPart;
    m->handle[0] = view;
    m->handle[1] = mapping;
    m->handle[2] = file;
    return 1;
}

int replace_file(const char *from, const char *to) {
    wchar_t *wf = widen(from), *wt = widen(to);
    BOOL ok = MoveFileExW(wf, wt, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
    free(wf);
    free(wt);
    return ok != 0;
}

void unmap_file(MappedFile *m) {
    if (m->handle[0]) UnmapViewOfFile(m->handle[0]);
    if (m->handle[1]) CloseHandle((HANDLE)m->handle[1]);
    if (m->handle[2]) CloseHandle((HANDLE)m->handle[2]);
    memset(m, 0, sizeof(*m));
}

FILE *open_file(const char *path, const char *mode) {
    wchar_t *wp = widen(path), *wm = widen(mode);
    FILE *f = _wfopen(wp, wm);
    free(wp);
    free(wm);
    return f;
}

static int file_identity(const char *path, BY_HANDLE_FILE_INFORMATION *info) {
    wchar_t *w = widen(path);
    HANDLE h = CreateFileW(w, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, NULL);
    int ok;
    free(w);
    if (h == INVALID_HANDLE_VALUE) return 0;
    ok = GetFileInformationByHandle(h, info) != 0;
    CloseHandle(h);
    return ok;
}

int same_file(const char *a, const char *b) {
    BY_HANDLE_FILE_INFORMATION x, y;
    if (!a || !b) return 0;
    if (!strcmp(a, b)) return 1;
    return file_identity(a, &x) && file_identity(b, &y) &&
           x.dwVolumeSerialNumber == y.dwVolumeSerialNumber &&
           x.nFileIndexHigh == y.nFileIndexHigh && x.nFileIndexLow == y.nFileIndexLow;
}

const char *self_executable_path(void) {
    static char *path;
    wchar_t buf[PATH_CAPACITY];
    DWORD n;
    if (path) return path;
    n = GetModuleFileNameW(NULL, buf, PATH_CAPACITY);
    if (!n || n >= PATH_CAPACITY) return NULL;
    path = narrow(buf);
    return path;
}

void make_executable(const char *path) {
    (void)path;
}

double now_seconds(void) {
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (!frequency.QuadPart && !QueryPerformanceFrequency(&frequency)) die("no performance counter");
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

uint64_t rss_peak_bytes(void) {
    PROCESS_MEMORY_COUNTERS counters;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) return 0;
    return (uint64_t)counters.PeakWorkingSetSize;
}

uint32_t cpu_logical_count(void) {
    DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n ? (uint32_t)n : 1;
}

uint32_t cpu_physical_cores(void) {
    DWORD length = 0, offset = 0;
    unsigned char *buffer;
    uint32_t cores = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &length);
    if (!length) return cpu_logical_count();
    buffer = (unsigned char *)malloc(length);
    if (!buffer) return cpu_logical_count();
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                         (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer, &length)) {
        while (offset < length) {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
                (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer + offset);
            if (info->Relationship == RelationProcessorCore) ++cores;
            offset += info->Size;
        }
    }
    free(buffer);
    return cores ? cores : cpu_logical_count();
}


/* Drops one character (both halves of a surrogate pair) from the end. */
static size_t drop_last_char(const wchar_t *w, size_t n) {
    if (n >= 2 && w[n - 1] >= 0xdc00 && w[n - 1] <= 0xdfff && w[n - 2] >= 0xd800 && w[n - 2] <= 0xdbff) return n - 2;
    return n - 1;
}

/* Console input is collected as UTF-16 for the whole line and converted once,
 * so surrogate pairs and CR/LF split across reads stay intact. A line that
 * does not fit is cut at a character boundary and reported as truncated. */
int read_line_utf8(char *buf, size_t cap, int *truncated) {
    static wchar_t *wide;
    static size_t wide_cap;
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode;
    size_t n = 0;
    int newline = 0, bytes;
    if (cap < 8) die("internal error: line buffer too small");
    if (!GetConsoleMode(in, &mode)) return read_stream_line(buf, cap, truncated);
    *truncated = 0;
    if (wide_cap < cap) { /* one UTF-16 unit per output byte is always enough */
        free(wide);
        wide = (wchar_t *)malloc(cap * sizeof(*wide));
        if (!wide) die("allocation failed");
        wide_cap = cap;
    }
    while (!newline) {
        wchar_t chunk[512];
        DWORD got = 0, k;
        if (!ReadConsoleW(in, chunk, (DWORD)(sizeof(chunk) / sizeof(*chunk)), &got, NULL) || !got) {
            if (!n) return -1;
            break;
        }
        if (!n && chunk[0] == 0x1a) return -1; /* Ctrl+Z */
        for (k = 0; k < got && !newline; ++k) {
            if (chunk[k] == L'\n') newline = 1;
            else if (chunk[k] == L'\r') continue;
            else if (n < wide_cap) wide[n++] = chunk[k];
            else *truncated = 1;
        }
    }
    if (n && wide[n - 1] >= 0xd800 && wide[n - 1] <= 0xdbff) --n; /* dangling high surrogate */
    bytes = n ? WideCharToMultiByte(CP_UTF8, 0, wide, (int)n, NULL, 0, NULL, NULL) : 0;
    if (bytes > (int)cap - 1) { /* shrink proportionally, then character by character */
        n = (size_t)((uint64_t)n * (cap - 1) / (uint64_t)bytes);
        if (n && wide[n - 1] >= 0xd800 && wide[n - 1] <= 0xdbff) --n;
        *truncated = 1;
    }
    for (;;) {
        if (!n) { buf[0] = 0; return 0; }
        bytes = WideCharToMultiByte(CP_UTF8, 0, wide, (int)n, buf, (int)cap - 1, NULL, NULL);
        if (bytes > 0) break;
        n = drop_last_char(wide, n);
        *truncated = 1;
    }
    buf[bytes] = 0;
    return bytes;
}

static UINT original_output_cp;

static void restore_console(void) {
    if (original_output_cp) SetConsoleOutputCP(original_output_cp);
}

unsigned console_original_codepage(void) {
    return original_output_cp != CP_UTF8 ? (unsigned)original_output_cp : 0;
}

void platform_init(int *argc, char ***argv) {
    int n, i;
    LPWSTR *wide = CommandLineToArgvW(GetCommandLineW(), &n);
    char **out;
    if (!wide) die("cannot read command line");
    out = (char **)calloc((size_t)n + 1, sizeof(*out));
    if (!out) die("allocation failed");
    for (i = 0; i < n; ++i) out[i] = narrow(wide[i]);
    LocalFree(wide);
    *argc = n;
    *argv = out;
    _setmode(_fileno(stdout), _O_BINARY);
    /* 0 when no console is attached (pipes, services): nothing to change. */
    original_output_cp = GetConsoleOutputCP();
    if (original_output_cp && original_output_cp != CP_UTF8) {
        SetConsoleOutputCP(CP_UTF8);
        atexit(restore_console);
    }
}

#else /* POSIX */

/* WebAssembly has no shared mappings: its mmap reads a private copy. */
#if defined(__wasm__)
#  define MAP_MODE MAP_PRIVATE
#else
#  define MAP_MODE MAP_SHARED
#endif

int try_map_file(const char *path, MappedFile *m, char *err, size_t errlen) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    void *p = MAP_FAILED;
    memset(m, 0, sizeof(*m));
    if (fd < 0) {
        snprintf(err, errlen, "cannot open %s: %s", path, strerror(errno));
        return 0;
    }
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0)
        snprintf(err, errlen, "not a readable non-empty regular file: %s", path);
    else if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX)
        snprintf(err, errlen, "file too large to map on this platform: %s", path);
    else if ((p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_MODE, fd, 0)) == MAP_FAILED)
        snprintf(err, errlen, "cannot map %s: %s", path, strerror(errno));
    close(fd);
    if (p == MAP_FAILED) return 0;
    m->data = (const unsigned char *)p;
    m->size = (uint64_t)st.st_size;
    return 1;
}

int replace_file(const char *from, const char *to) {
    return rename(from, to) == 0; /* atomic replacement on POSIX */
}

void unmap_file(MappedFile *m) {
    if (m->data) munmap((void *)m->data, (size_t)m->size);
    memset(m, 0, sizeof(*m));
}

FILE *open_file(const char *path, const char *mode) {
    return fopen(path, mode);
}

int same_file(const char *a, const char *b) {
    struct stat x, y;
    if (!a || !b) return 0;
    if (!strcmp(a, b)) return 1;
    return !stat(a, &x) && !stat(b, &y) && x.st_dev == y.st_dev && x.st_ino == y.st_ino;
}

const char *self_executable_path(void) {
    static char path[PATH_CAPACITY];
#if defined(__APPLE__)
    uint32_t size = sizeof(path);
    if (path[0]) return path;
    return _NSGetExecutablePath(path, &size) == 0 ? path : NULL;
#elif defined(__wasm__)
    (void)path; /* a WebAssembly module is not a file the runtime can reopen */
    return NULL;
#else
    ssize_t n;
    if (path[0]) return path;
    n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) return NULL;
    path[n] = 0;
    return path;
#endif
}

void make_executable(const char *path) {
#if defined(__wasm__)
    (void)path; /* no file modes */
#else
    if (chmod(path, 0755)) die("cannot make %s executable: %s", path, strerror(errno));
#endif
}

double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) die("clock_gettime failed");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

uint64_t rss_peak_bytes(void) {
#if defined(__wasm__)
    /* Linear memory only grows: its size is the peak. */
    return (uint64_t)__builtin_wasm_memory_size(0) * 65536u;
#else
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r)) return 0;
#  if defined(__APPLE__)
    return (uint64_t)r.ru_maxrss;
#  else
    return (uint64_t)r.ru_maxrss * 1024;
#  endif
#endif
}

uint32_t cpu_logical_count(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t)n : 1;
}

uint32_t cpu_physical_cores(void) {
#if defined(__APPLE__)
    int cores = 0;
    size_t size = sizeof(cores);
    if (sysctlbyname("hw.physicalcpu", &cores, &size, NULL, 0) == 0 && cores > 0) return (uint32_t)cores;
    return cpu_logical_count();
#else
    /* A CPU is the first thread of its core when its sibling list starts with itself. */
    uint32_t logical = cpu_logical_count(), cpu, cores = 0;
    for (cpu = 0; cpu < logical; ++cpu) {
        char name[128];
        unsigned first;
        FILE *f;
        snprintf(name, sizeof(name), "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list", cpu);
        f = fopen(name, "r");
        if (!f) return logical;
        if (fscanf(f, "%u", &first) == 1 && first == cpu) ++cores;
        fclose(f);
    }
    return cores ? cores : logical;
#endif
}


int read_line_utf8(char *buf, size_t cap, int *truncated) {
    return read_stream_line(buf, cap, truncated);
}

void platform_init(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
}

unsigned console_original_codepage(void) {
    return 0;
}

#endif

void map_file(const char *path, MappedFile *m) {
    char err[PATH_CAPACITY + 256];
    if (!try_map_file(path, m, err, sizeof(err))) sm_fail(SM_ERR_IO, "%s", err);
}
