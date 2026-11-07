#include "util.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#define DIE_MESSAGE_BYTES 1024

static void (*die_handler)(const char *message);

void die_set_handler(void (*handler)(const char *message)) { die_handler = handler; }

void die(const char *fmt, ...) {
    char message[DIE_MESSAGE_BYTES];
    va_list ap;
    fflush(stdout);
    va_start(ap, fmt);
    vsnprintf(message, sizeof(message), fmt, ap);
    va_end(ap);
    fprintf(stderr, "smolmoe: %s\n", message);
    if (die_handler) die_handler(message);
    exit(EXIT_FAILURE);
}

typedef union Allocation Allocation;
union Allocation {
    struct { Allocation *next; size_t bytes; } h;
    long double align;
};

size_t mul_size(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) die("size overflow");
    return a * b;
}

size_t add_size(size_t a, size_t b) {
    if (a > SIZE_MAX - b) die("size overflow");
    return a + b;
}

void *mem_alloc(Memory *m, size_t count, size_t size) {
    size_t bytes = mul_size(count, size);
    Allocation *p;
    if (bytes > SIZE_MAX - sizeof(*p)) die("allocation overflow");
    bytes += sizeof(*p);
    if (bytes > m->limit - m->used)
        die("memory budget too small: need %zu more bytes (used %zu of %zu); "
            "reduce --context/--batch or increase --memory-mib", bytes, m->used, m->limit);
    p = (Allocation *)calloc(1, bytes);
    if (!p) die("allocation of %zu bytes failed", bytes);
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
    if (!*s || *s == '-' || *s == '+') die("invalid %s: %s", what, s);
    errno = 0;
    n = strtoull(s, &end, 10);
    if (errno || *end || n > max) die("invalid %s: %s", what, s);
    return (uint32_t)n;
}

double parse_real(const char *s, const char *what) {
    char *end;
    double n;
    errno = 0;
    n = strtod(s, &end);
    if (!*s || errno || *end || !isfinite(n)) die("invalid %s: %s", what, s);
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
        if (!c) die("%s contains a NUL byte", what);
        if (n + 1 >= cap) die("%s line too long", what);
        line[n++] = c;
        if (c == '\n') break;
    }
    line[n] = 0;
    return 1;
}
