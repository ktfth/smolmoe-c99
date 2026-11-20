#include "util.h"
#include "finite.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

/* wasm32-wasi has no threads, and a TLS qualifier there is unnecessary. The
 * browser build (atomics) does share the module across workers. */
#if defined(__wasm__) && !defined(__wasm_atomics__)
#  define SM_TLS
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
#  define SM_TLS _Thread_local
#elif defined(__GNUC__) || defined(__clang__)
#  define SM_TLS __thread
#elif defined(_MSC_VER)
#  define SM_TLS __declspec(thread)
#else
#  define SM_TLS
#endif

static void (*die_handler)(const char *message);
static SM_TLS sm_status tls_status;
static SM_TLS char tls_message[SM_MESSAGE_BYTES];

void die_set_handler(void (*handler)(const char *message)) { die_handler = handler; }

sm_status sm_last_status(void) { return tls_status; }
const char *sm_last_error(void) { return tls_message; }

void sm_error_get(SmError *err) {
    if (!err) return;
    err->status = tls_status;
    snprintf(err->message, sizeof(err->message), "%s", tls_message);
}

static void store_message(sm_status status, const char *fmt, va_list ap) {
    /* A stack buffer: callers pass sm_last_error(), which is tls_message, and
     * vsnprintf cannot read and write the same array. */
    char buf[SM_MESSAGE_BYTES];
    tls_status = status == SM_OK ? SM_ERR : status;
    vsnprintf(buf, sizeof(buf), fmt, ap);
    if (!buf[0]) snprintf(buf, sizeof(buf), "unknown error");
    memcpy(tls_message, buf, strlen(buf) + 1);
}

sm_status sm_set(sm_status status, const char *fmt, ...) {
    va_list ap;
    if (status == SM_OK) {
        tls_status = SM_OK;
        tls_message[0] = 0;
        (void)fmt;
        return SM_OK;
    }
    va_start(ap, fmt);
    store_message(status, fmt, ap);
    va_end(ap);
    return tls_status;
}

static SM_NORETURN void exit_message(const char *message) {
    fflush(stdout);
    fprintf(stderr, "smolmoe: %s\n", message);
    if (die_handler) die_handler(message);
    exit(EXIT_FAILURE);
}

void sm_check(sm_status status) {
    char message[SM_MESSAGE_BYTES];
    if (status == SM_OK) return;
    memcpy(message, tls_message, sizeof(message));
    die("%s", message);
}

sm_status sm_protect(SmProtectedFn fn, void *ctx) {
    sm_set(SM_OK, "%s", "");
    fn(ctx);
    return sm_last_status();
}

void die(const char *fmt, ...) {
    char buf[SM_MESSAGE_BYTES];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (!buf[0]) snprintf(buf, sizeof(buf), "unknown error");
    exit_message(buf);
}

typedef union Allocation Allocation;
union Allocation {
    struct { Allocation *next; size_t bytes; } h;
    long double align;
};

size_t mul_size(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) {
        sm_set(SM_ERR_RANGE, "size overflow");
        return 0;
    }
    return a * b;
}

size_t add_size(size_t a, size_t b) {
    if (a > SIZE_MAX - b) {
        sm_set(SM_ERR_RANGE, "size overflow");
        return 0;
    }
    return a + b;
}

void *mem_alloc(Memory *m, size_t count, size_t size) {
    size_t bytes;
    Allocation *p;
    if (sm_failed()) return NULL;
    bytes = mul_size(count, size);
    if (sm_failed()) return NULL;
    if (bytes > SIZE_MAX - sizeof(*p)) {
        sm_set(SM_ERR_NOMEM, "allocation overflow");
        return NULL;
    }
    bytes += sizeof(*p);
    if (bytes > m->limit - m->used) {
        sm_set(SM_ERR_NOMEM, "memory budget too small: need %zu more bytes (used %zu of %zu); "
               "reduce --context/--batch or increase --memory-mib", bytes, m->used, m->limit);
        return NULL;
    }
    p = (Allocation *)calloc(1, bytes);
    if (!p) {
        sm_set(SM_ERR_NOMEM, "allocation of %zu bytes failed", bytes);
        return NULL;
    }
    p->h.next = (Allocation *)m->head;
    p->h.bytes = bytes;
    m->head = p;
    m->used += bytes;
    if (m->used > m->peak) m->peak = m->used;
    return p + 1;
}

void mem_release(Memory *m) {
    Allocation *p = (Allocation *)m->head;
    while (p) {
        Allocation *next = p->h.next;
        free(p);
        p = next;
    }
    m->head = NULL;
    m->used = 0;
}

uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

uint32_t parse_u32(const char *s, uint32_t max, const char *what) {
    char *end;
    unsigned long long n;
    if (!*s || *s == '-' || *s == '+') {
        sm_set(SM_ERR_FORMAT, "invalid %s: %s", what, s);
        return 0;
    }
    errno = 0;
    n = strtoull(s, &end, 10);
    if (errno || *end || n > max) {
        sm_set(SM_ERR_FORMAT, "invalid %s: %s", what, s);
        return 0;
    }
    return (uint32_t)n;
}

double parse_real(const char *s, const char *what) {
    char *end;
    double n;
    errno = 0;
    n = strtod(s, &end);
    if (!*s || errno || *end || !sm_finite(n)) {
        sm_set(SM_ERR_FORMAT, "invalid %s: %s", what, s);
        return 0;
    }
    return n;
}

char *trim(char *s) {
    char *end;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') ++s;
    end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) --end;
    *end = 0;
    return s;
}

int br_line(ByteReader *r, char *line, size_t cap, const char *what) {
    size_t n = 0;
    if (br_at_end(r)) return 0;
    while (r->pos < r->size) {
        char c = (char)r->data[r->pos++];
        if (!c) {
            sm_set(SM_ERR_FORMAT, "%s contains a NUL byte", what);
            return -1;
        }
        if (n + 1 >= cap) {
            sm_set(SM_ERR_FORMAT, "%s line too long", what);
            return -1;
        }
        line[n++] = c;
        if (c == '\n') break;
    }
    line[n] = 0;
    return 1;
}
