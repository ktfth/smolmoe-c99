#include "bundle.h"
#include "container.h"
#include "util.h"

#include <string.h>

int bundle_find(const unsigned char *data, uint64_t size, uint64_t *offset, uint64_t *length) {
    const unsigned char *t;
    uint64_t off, len;
    if (size < BUNDLE_TRAILER_BYTES) return 0;
    t = data + size - BUNDLE_TRAILER_BYTES;
    if (memcmp(t, BUNDLE_MAGIC, 8) || memcmp(t + 24, BUNDLE_MAGIC, 8)) return 0;
    off = rd_u64(t + 8);
    len = rd_u64(t + 16);
    if (off % BUNDLE_ALIGN || len < 16 || off > size - BUNDLE_TRAILER_BYTES ||
        len != size - BUNDLE_TRAILER_BYTES - off)
        die("corrupt bundle trailer");
    *offset = off;
    *length = len;
    return 1;
}

void bundle_trailer(unsigned char out[BUNDLE_TRAILER_BYTES], uint64_t offset, uint64_t length) {
    memcpy(out, BUNDLE_MAGIC, 8);
    put_u64(out + 8, offset);
    put_u64(out + 16, length);
    memcpy(out + 24, BUNDLE_MAGIC, 8);
}

int bundle_sibling_path(const char *exe, char *out, size_t cap) {
    size_t n, base, stem;
    if (!exe || !out || !exe[0] || cap < 7) return 0;
    n = strlen(exe);
    base = n;
    while (base && exe[base - 1] != '/' && exe[base - 1] != '\\') --base;
    stem = n;
    if (n - base >= 4) {
        const char *e = exe + (n - 4);
        if (e[0] == '.' && (e[1] == 'e' || e[1] == 'E') && (e[2] == 'x' || e[2] == 'X') &&
            (e[3] == 'e' || e[3] == 'E'))
            stem = n - 4;
    }
    if (stem == base || stem > cap - 6) return 0;
    memcpy(out, exe, stem);
    memcpy(out + stem, ".smoe", 6);
    return 1;
}
