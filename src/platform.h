/* Operating-system layer: Linux/macOS (POSIX), Windows (Win32) and WebAssembly
 * (POSIX through WASI or Emscripten, without processes or a terminal).
 * Everything outside platform.c, process.c and threadpool.c is portable C99. */
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
/* Same, but reports failure (message in err) instead of exiting. */
int try_map_file(const char *path, MappedFile *out, char *err, size_t errlen);
void unmap_file(MappedFile *m);

/* Atomically replaces `to` with `from` (rename; MoveFileEx on Windows). Returns 1 on success. */
int replace_file(const char *from, const char *to);

/* fopen() that accepts UTF-8 paths on every platform. */
FILE *open_file(const char *path, const char *mode);
int same_file(const char *a, const char *b);
const char *self_executable_path(void);
/* Marks a file executable (chmod 0755); no-op on Windows. */
void make_executable(const char *path);

/* Creates every missing directory along path. Returns 1 on success. */
int ensure_directory(const char *path);
/* The per-user cache directory, without a trailing slash. 0 when there is
 * nowhere persistent to put one (no home directory, or WebAssembly). */
int user_cache_dir(char *buf, size_t cap);
uint32_t process_id(void);

double now_seconds(void);
uint64_t rss_peak_bytes(void);
uint32_t cpu_logical_count(void);
uint32_t cpu_physical_cores(void);

/* Reads one line of user input as UTF-8 (the Windows console is read as UTF-16
 * and converted). Strips the line ending. Returns its length, or -1 at EOF.
 * A line longer than cap-1 bytes is cut at a character boundary, the rest of
 * it is discarded and *truncated is set. */
int read_line_utf8(char *buf, size_t cap, int *truncated);

/* Converts argv to UTF-8 and sets stdout to binary/UTF-8 on Windows. */
void platform_init(int *argc, char ***argv);
/* Windows: the console output code page before platform_init switched it to
 * UTF-8, so a child program can run with the user's own; 0 when nothing was
 * switched (and always 0 elsewhere). */
unsigned console_original_codepage(void);

#endif
