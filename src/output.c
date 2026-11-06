#include "output.h"
#include "util.h"

#include <string.h>

Output output_to(FILE *f) {
    Output o;
    memset(&o, 0, sizeof(o));
    o.f = f;
    return o;
}

static uint32_t incomplete_tail(const unsigned char *p, uint32_t n) {
    uint32_t back;
    for (back = 1; back <= 3 && back <= n; ++back) {
        unsigned char c = p[n - back];
        uint32_t need;
        if ((c & 0xc0) == 0x80) continue;
        need = (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : (c & 0xf8) == 0xf0 ? 4 : 1;
        return back < need ? back : 0;
    }
    return 0;
}

static void write_all(FILE *f, const unsigned char *p, size_t n) {
    if (n && fwrite(p, 1, n, f) != n) die("output write failed");
}

void output_bytes(void *ctx, const unsigned char *p, uint32_t n) {
    Output *o = (Output *)ctx;
    unsigned char tail[8];
    uint32_t total = o->n + n, tl = total < 4 ? total : 4, i, hold;
    for (i = 0; i < tl; ++i) {
        uint32_t at = total - tl + i;
        tail[i] = at < o->n ? o->pending[at] : p[at - o->n];
    }
    hold = incomplete_tail(tail, tl);
    if (total - hold <= o->n) {
        write_all(o->f, o->pending, total - hold);
    } else {
        write_all(o->f, o->pending, o->n);
        write_all(o->f, p, total - hold - o->n);
    }
    memmove(o->pending, tail + tl - hold, hold);
    o->n = hold;
}

void output_finish(Output *o) {
    write_all(o->f, o->pending, o->n);
    o->n = 0;
    if (fflush(o->f)) die("output flush failed");
}
