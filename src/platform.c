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
#  include <sys/resource.h>
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

static int read_stream_line(char *buf, size_t cap) {
    if (!fgets(buf, (int)cap, stdin)) return -1;
    return strip_line_end(buf, strlen(buf));
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

void map_file(const char *path, MappedFile *m) {
    wchar_t *w = widen(path);
    HANDLE file = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    LARGE_INTEGER size;
    HANDLE mapping;
    void *view;
    free(w);
    memset(m, 0, sizeof(*m));
    if (file == INVALID_HANDLE_VALUE) die("cannot open %s: %s", path, win_error());
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0) die("empty or unreadable file: %s", path);
    if ((uint64_t)size.QuadPart > (uint64_t)SIZE_MAX) die("file too large to map on this platform: %s", path);
    mapping = CreateFileMappingW(file, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mapping) die("cannot map %s: %s", path, win_error());
    view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (!view) die("cannot map view of %s: %s", path, win_error());
    m->data = (const unsigned char *)view;
    m->size = (uint64_t)size.QuadPart;
    m->handle[0] = view;
    m->handle[1] = mapping;
    m->handle[2] = file;
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


int read_line_utf8(char *buf, size_t cap) {
    HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
    DWORD mode, got = 0;
    wchar_t wide[2048];
    size_t used = 0;
    if (cap < 8) die("internal error: line buffer too small");
    if (!GetConsoleMode(in, &mode)) return read_stream_line(buf, cap);
    /* Console: read UTF-16 until the line ends (long lines arrive in pieces). */
    for (;;) {
        int n;
        if (!ReadConsoleW(in, wide, (DWORD)(sizeof(wide) / sizeof(*wide)), &got, NULL) || !got)
            return used ? strip_line_end(buf, used) : -1;
        if (!used && wide[0] == 0x1a) return -1; /* Ctrl+Z */
        n = WideCharToMultiByte(CP_UTF8, 0, wide, (int)got, buf + used, (int)(cap - 1 - used), NULL, NULL);
        if (n <= 0) die("input line too long");
        used += (size_t)n;
        buf[used] = 0;
        if (buf[used - 1] == '\n') return strip_line_end(buf, used);
        if (used + 4 >= cap) return strip_line_end(buf, used);
    }
}

static UINT original_output_cp;

static void restore_console(void) {
    if (original_output_cp) SetConsoleOutputCP(original_output_cp);
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

void map_file(const char *path, MappedFile *m) {
    int fd = open(path, O_RDONLY);
    struct stat st;
    void *p;
    memset(m, 0, sizeof(*m));
    if (fd < 0) die("cannot open %s: %s", path, strerror(errno));
    if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size <= 0)
        die("not a readable non-empty regular file: %s", path);
    if ((uint64_t)st.st_size > (uint64_t)SIZE_MAX) die("file too large to map on this platform: %s", path);
    p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) die("cannot map %s: %s", path, strerror(errno));
    close(fd);
    m->data = (const unsigned char *)p;
    m->size = (uint64_t)st.st_size;
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
    if (chmod(path, 0755)) die("cannot make %s executable: %s", path, strerror(errno));
}

double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts)) die("clock_gettime failed");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

uint64_t rss_peak_bytes(void) {
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r)) return 0;
#if defined(__APPLE__)
    return (uint64_t)r.ru_maxrss;
#else
    return (uint64_t)r.ru_maxrss * 1024;
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


int read_line_utf8(char *buf, size_t cap) {
    return read_stream_line(buf, cap);
}

void platform_init(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
}

#endif
