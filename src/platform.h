/* Operating-system layer: Linux/macOS (POSIX) and Windows (Win32).
 * Everything outside platform.c and threadpool.c is portable C99. */
#ifndef SMOLMOE_PLATFORM_H
#define SMOLMOE_PLATFORM_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

typedef struct MappedFile {
    const unsigned char *data;
    uint64_t size;
    void *handle[3];
} MappedFile;

/* Read-only shared mapping of a whole file. Dies on failure. On Windows the
 * file is opened without write sharing, so it cannot be modified while mapped. */
void map_file(const char *path, MappedFile *out);
void unmap_file(MappedFile *m);

/* fopen() that accepts UTF-8 paths on every platform. */
FILE *open_file(const char *path, const char *mode);
int same_file(const char *a, const char *b);
const char *self_executable_path(void);
/* Marks a file executable (chmod 0755); no-op on Windows. */
void make_executable(const char *path);

double now_seconds(void);
uint64_t rss_peak_bytes(void);
uint32_t cpu_logical_count(void);
uint32_t cpu_physical_cores(void);

/* Reads one line of user input as UTF-8 (the Windows console is read as UTF-16
 * and converted). Strips the line ending. Returns its length, or -1 at EOF. */
int read_line_utf8(char *buf, size_t cap);

/* Converts argv to UTF-8 and sets stdout to binary/UTF-8 on Windows. */
void platform_init(int *argc, char ***argv);

#endif
