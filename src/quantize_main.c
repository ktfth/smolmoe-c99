/* smolmoe-quantize: rewrites an SMOEQ8 checkpoint with Q5 matrices.
 *
 * The source is an existing Q8 checkpoint, so this is a second lossy step
 * (Q8 -> float -> Q5). Its quality cost is measured, not assumed:
 * see docs/DECISIONS.md. Embedded blobs (config, tokenizer) are copied. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "container.h"
#include "kernels.h"
#include "modelfile.h"
#include "platform.h"
#include "threadpool.h"
#include "util.h"

#define MAX_KEEP 32
#define MAX_COLS 65536u

typedef struct Options {
    const char *in, *out;
    const char *keep[MAX_KEEP];
    int nkeep;
    int simulate_bits; /* 0: real Q5. 3..7: keep Q8 storage, restrict values to an N-bit grid */
} Options;

/* Per-tensor conversion job handed to the payload writer. */
typedef struct Convert {
    const Tensor *t;
    int simulate_bits;
} Convert;

static float row_buffer[MAX_COLS];
static unsigned char packed_buffer[(MAX_COLS / 32) * 36];

static void usage(void) {
    fputs("usage: smolmoe-quantize --in Q8.smoe --out Q5.smoe [--keep NAME ...] [--simulate-bits N]\n"
          "  Converts every Q8 matrix to Q5 (22 bytes per 32 weights) except tensors whose\n"
          "  name contains a --keep substring. Routers are always kept in Q8.\n"
          "  --simulate-bits N  research mode: keep Q8 storage but restrict every block to an\n"
          "                     N-bit grid (3..7) to measure quality before writing kernels.\n", stderr);
    exit(1);
}

static void parse(int argc, char **argv, Options *o) {
    int a;
    memset(o, 0, sizeof(*o));
    for (a = 1; a < argc; ++a) {
        if (a + 1 >= argc) usage();
        if (!strcmp(argv[a], "--in")) o->in = argv[++a];
        else if (!strcmp(argv[a], "--out")) o->out = argv[++a];
        else if (!strcmp(argv[a], "--simulate-bits")) {
            o->simulate_bits = (int)parse_u32(argv[++a], 7, "simulate-bits");
            if (o->simulate_bits < 3) die("--simulate-bits must be in 3..7");
        } else if (!strcmp(argv[a], "--keep")) {
            if (o->nkeep == MAX_KEEP) die("too many --keep patterns");
            o->keep[o->nkeep++] = argv[++a];
        } else usage();
    }
    if (!o->in || !o->out) usage();
    if (same_file(o->in, o->out)) die("--out must differ from --in");
}

static int keep_q8(const Options *o, const char *name) {
    int i;
    if (strstr(name, "router")) return 1;
    for (i = 0; i < o->nkeep; ++i)
        if (strstr(name, o->keep[i])) return 1;
    return 0;
}

static void quantize_grid_row(const float *x, uint32_t cols, int bits, unsigned char *out) {
    const int lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
    uint32_t nb = (cols + QK - 1) / QK, b, j, k;
    for (b = 0; b < nb; ++b, x += QK, out += Q8_BLOCK_BYTES) {
        uint32_t n = cols - b * QK < QK ? cols - b * QK : QK;
        float vmax = 0, best_d = 0;
        double best = -1;
        for (j = 0; j < n; ++j)
            if (fabsf(x[j]) > fabsf(vmax)) vmax = x[j];
        for (k = 0; vmax != 0 && k <= 20; ++k) {
            float d = vmax / -((float)-lo - 1.5f + 0.1f * (float)k), inv = 1.0f / d;
            double err = 0;
            for (j = 0; j < n; ++j) {
                long q = lrintf(x[j] * inv);
                double e = (double)x[j] - (double)(q < lo ? lo : q > hi ? hi : q) * d;
                err += e * e;
            }
            if (best < 0 || err < best) { best = err; best_d = d; }
        }
        {
            /* Q8 scales are non-negative: flip scale and levels together. */
            float stored = fabsf(best_d);
            long sign = best_d < 0 ? -1 : 1;
            memcpy(out, &stored, 4);
            for (j = 0; j < QK; ++j) {
                long q = j < n && best_d != 0 ? lrintf(x[j] / best_d) : 0;
                out[4 + j] = (unsigned char)(int8_t)(sign * (q < lo ? lo : q > hi ? hi : q));
            }
        }
    }
}

static void write_converted(FILE *f, const void *ctx) {
    const Convert *c = (const Convert *)ctx;
    const Tensor *t = c->t;
    uint32_t r, nb = (t->cols + QK - 1) / QK;
    for (r = 0; r < t->rows; ++r) {
        dequantize_q8_row(t->data + (size_t)r * nb * Q8_BLOCK_BYTES, t->cols, row_buffer);
        if (c->simulate_bits) {
            quantize_grid_row(row_buffer, t->cols, c->simulate_bits, packed_buffer);
            write_bytes(f, packed_buffer, (size_t)nb * Q8_BLOCK_BYTES);
        } else {
            quantize_q5_row(row_buffer, t->cols, packed_buffer);
            write_bytes(f, packed_buffer, (size_t)nb * Q5_BLOCK_BYTES);
        }
    }
}

/* Re-reads the written file through the runtime parser and validator. */
static void verify_output(const char *path, Pool *pool, Memory *mem) {
    MappedFile mapped;
    ModelFile mf;
    map_file(path, &mapped);
    sm_check(modelfile_parse(&mf, mapped.data, mapped.size, mem));
    sm_check(modelfile_verify_payload(&mf, pool));
    unmap_file(&mapped);
}

int main(int argc, char **argv) {
    Options o;
    Memory mem;
    MappedFile mapped;
    ModelFile mf;
    Pool *pool;
    OutRecord *records;
    Convert *jobs;
    FILE *f;
    char *tmp;
    uint64_t before = 0, after = 0;
    uint32_t i;

    platform_init(&argc, &argv);
    parse(argc, argv, &o);
    memset(&mem, 0, sizeof(mem));
    mem.limit = (size_t)256 << 20;
    pool = pool_create(cpu_physical_cores(), &mem);
    if (!pool) die("%s", sm_last_error());
    map_file(o.in, &mapped);
    sm_check(modelfile_parse(&mf, mapped.data, mapped.size, &mem));
    sm_check(modelfile_verify_payload(&mf, pool));

    records = (OutRecord *)mem_alloc(&mem, mf.count, sizeof(OutRecord));
    jobs = (Convert *)mem_alloc(&mem, mf.count, sizeof(Convert));
    for (i = 0; i < mf.count; ++i) {
        const Tensor *t = &mf.tensors[i];
        uint32_t nb = (t->cols + QK - 1) / QK;
        records[i] = record_from_tensor(t);
        if (t->dtype == DTYPE_Q8 && !keep_q8(&o, t->name) && !t->alias) {
            jobs[i].t = t;
            jobs[i].simulate_bits = o.simulate_bits;
            records[i].write = write_converted;
            records[i].ctx = &jobs[i];
            if (!o.simulate_bits) {
                records[i].dtype = DTYPE_Q5;
                records[i].bytes = (uint64_t)t->rows * nb * Q5_BLOCK_BYTES;
            }
        }
        if (!t->alias) {
            before += t->bytes;
            after += records[i].bytes;
        }
    }
    /* A tied lm_head follows its embed's new type and size. */
    for (i = 0; i < mf.count; ++i) {
        uint32_t j;
        if (!mf.tensors[i].alias) continue;
        for (j = 0; j < mf.count; ++j)
            if (!strcmp(records[j].name, "embed")) {
                records[i].dtype = records[j].dtype;
                records[i].bytes = records[j].bytes;
            }
    }

    tmp = (char *)mem_alloc(&mem, strlen(o.out) + 5, 1);
    snprintf(tmp, strlen(o.out) + 5, "%s.tmp", o.out);
    f = open_file(tmp, "wb");
    if (!f) die("cannot create %s: %s", tmp, strerror(errno));
    container_write(f, mf.version, records, mf.count);
    if (fclose(f)) die("cannot finish %s: %s", tmp, strerror(errno));
    verify_output(tmp, pool, &mem);
    if (!replace_file(tmp, o.out)) {
        remove(tmp);
        die("cannot replace %s (is it in use?)", o.out);
    }
    fprintf(stderr, "{\"tensors\":%u,\"payload_bytes_before\":%llu,\"payload_bytes_after\":%llu,\"ratio\":%.4f}\n",
            mf.count, (unsigned long long)before, (unsigned long long)after, before ? (double)after / (double)before : 0);
    pool_destroy(pool);
    unmap_file(&mapped);
    mem_release(&mem);
    return 0;
}
