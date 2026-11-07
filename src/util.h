/* Shared helpers: fatal errors, counted heap, little-endian readers, parsing. */
#ifndef SMOLMOE_UTIL_H
#define SMOLMOE_UTIL_H

#include <stddef.h>
#include <stdio.h> /* defines __MINGW_PRINTF_FORMAT on MinGW */
#include <stdint.h>
#include <string.h>

#define SMOLMOE_VERSION "0.3.1"

#if defined(__GNUC__) || defined(__clang__)
#  define SM_NORETURN __attribute__((noreturn))
#  if defined(__MINGW_PRINTF_FORMAT)
#    define SM_PRINTF(f, a) __attribute__((format(__MINGW_PRINTF_FORMAT, f, a)))
#  else
#    define SM_PRINTF(f, a) __attribute__((format(printf, f, a)))
#  endif
#else
#  define SM_NORETURN
#  define SM_PRINTF(f, a)
#endif

SM_NORETURN void die(const char *fmt, ...) SM_PRINTF(1, 2);
/* Called by die() with the message, after printing it and before exiting. It
 * may also never return: the desktop chat parks the failing thread and shows
 * the message in its window. NULL (the default): nothing. */
void die_set_handler(void (*handler)(const char *message));

/* Counted heap. Every runtime allocation (tokenizer, activations, KV cache,
 * thread pool) is charged against a fixed budget and zero-initialized.
 * Model weights are memory-mapped, file-backed and not charged. */
typedef struct Memory {
    size_t limit, used, peak;
    void *head;
} Memory;

void *mem_alloc(Memory *m, size_t count, size_t size);
void mem_release(Memory *m);
size_t mul_size(size_t a, size_t b);
size_t add_size(size_t a, size_t b);

static inline uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static inline uint64_t rd_u64(const unsigned char *p) {
    return rd_u32(p) | (uint64_t)rd_u32(p + 4) << 32;
}
static inline float rd_f32(const unsigned char *p) {
    uint32_t u = rd_u32(p);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* Sequential reader over an in-memory buffer (mapped file or embedded blob). */
typedef struct ByteReader {
    const unsigned char *data;
    size_t size, pos;
} ByteReader;

static inline ByteReader byte_reader(const unsigned char *data, size_t size) {
    ByteReader r;
    r.data = data;
    r.size = size;
    r.pos = 0;
    return r;
}

/* Copies n bytes; 0 if fewer remain. */
static inline int br_read(ByteReader *r, void *out, size_t n) {
    if (n > r->size - r->pos) return 0;
    memcpy(out, r->data + r->pos, n);
    r->pos += n;
    return 1;
}

static inline int br_at_end(const ByteReader *r) { return r->pos == r->size; }

/* Reads one line (with its newline) as a C string. Returns 1, 0 at end of
 * input, and dies on a line longer than cap-1 bytes or an embedded NUL. */
int br_line(ByteReader *r, char *line, size_t cap, const char *what);

uint64_t mix64(uint64_t x);
uint32_t parse_u32(const char *s, uint32_t max, const char *what);
double parse_real(const char *s, const char *what);
char *trim(char *s);

#endif
