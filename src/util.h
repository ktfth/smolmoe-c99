/* Shared helpers: fatal errors, counted heap, little-endian readers, parsing. */
#ifndef SMOLMOE_UTIL_H
#define SMOLMOE_UTIL_H

#include <stddef.h>
#include <stdio.h> /* defines __MINGW_PRINTF_FORMAT on MinGW */
#include <stdint.h>
#include <string.h>

#define SMOLMOE_VERSION "0.5.0"

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

/* Recoverable failures of the library (load, parse, forward, sample, tokenize).
 * SM_OK is the only zero value, so a status is also a C truth test for failure.
 * die() is for the CLI and the other top-level programs: it prints "smolmoe: "
 * and exits. Inside an operation guarded by the functions below, die() and
 * sm_fail() do not exit; they record the message and return the status. */
typedef enum sm_status {
    SM_OK = 0,
    SM_ERR = 1,         /* unspecified; the message has the detail */
    SM_ERR_IO = 2,      /* a file could not be opened or read */
    SM_ERR_FORMAT = 3,  /* a model, config or tokenizer is not valid */
    SM_ERR_RANGE = 4,   /* a size, context, id or seed is out of range */
    SM_ERR_NUMERIC = 5, /* a non-finite weight or activation */
    SM_ERR_NOMEM = 6,   /* the heap budget or the allocator refused */
    SM_ERR_STATE = 7    /* the call does not fit the engine's current state */
} sm_status;

#define SM_MESSAGE_BYTES 1024

typedef struct SmError {
    sm_status status;
    char message[SM_MESSAGE_BYTES];
} SmError;

/* The error from the latest library call on this thread. The pointer is valid
 * until the next call on the same thread; copy it to keep it. Empty when the
 * status is SM_OK. */
sm_status sm_last_status(void);
const char *sm_last_error(void);
/* Copies that error into *err (SM_OK and an empty message when err is NULL). */
void sm_error_get(SmError *err);

/* Records a status and its message. Does not exit and does not unwind. */
sm_status sm_set(sm_status status, const char *fmt, ...) SM_PRINTF(2, 3);

/* Records the error and unwinds to the guarding operation (sm_protect). With no
 * guard, prints and exits, as die() does. The format is the same sentence
 * die() would print. */
SM_NORETURN void sm_fail(sm_status status, const char *fmt, ...) SM_PRINTF(2, 3);
/* sm_fail() again with the error already stored. */
SM_NORETURN void sm_reraise(void);

/* Runs fn(ctx). sm_fail() and die() inside fn return here as a status instead
 * of exiting the process. fn is not called again after a failure. */
typedef void (*SmProtectedFn)(void *ctx);
sm_status sm_protect(SmProtectedFn fn, void *ctx);

/* Top-level programs: die() with the stored message when status is not SM_OK. */
void sm_check(sm_status status);

SM_NORETURN void die(const char *fmt, ...) SM_PRINTF(1, 2);
/* Called by die() with the message, after printing it and before exiting, when
 * nothing is guarding the call. NULL (the default): nothing. A guard installed
 * by the library swallows die() instead, so an embedded caller is not exited. */
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
