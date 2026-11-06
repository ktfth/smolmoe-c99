/* smolmoe-quantize: rewrites an SMOEQ8 checkpoint with Q5 matrices.
 *
 * The source is an existing Q8 checkpoint, so this is a second lossy step
 * (Q8 -> float -> Q5). Its quality cost is measured, not assumed:
 * see docs/DECISIONS.md. */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernels.h"
#include "modelfile.h"
#include "platform.h"
#include "threadpool.h"
#include "util.h"

#define HEADER_BYTES 16u
#define RECORD_BYTES 72u
#define PAYLOAD_ALIGN 64u
#define MAX_KEEP 32

typedef struct Plan {
    uint32_t dtype;
    int simulate;
    uint64_t offset, bytes;
} Plan;

typedef struct Options {
    const char *in, *out;
    const char *keep[MAX_KEEP];
    int nkeep;
    int simulate_bits; /* 0: real Q5. 3..7: keep Q8 storage, restrict values to an N-bit grid */
} Options;

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
        }
        else if (!strcmp(argv[a], "--keep")) {
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

static uint64_t align_up(uint64_t x) {
    return (x + PAYLOAD_ALIGN - 1) / PAYLOAD_ALIGN * PAYLOAD_ALIGN;
}

static void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24);
}

static void put_u64(unsigned char *p, uint64_t v) {
    put_u32(p, (uint32_t)v);
    put_u32(p + 4, (uint32_t)(v >> 32));
}

static void write_bytes(FILE *f, const void *p, size_t n) {
    if (n && fwrite(p, 1, n, f) != n) die("write failed: %s", strerror(errno));
}

static void write_zeros(FILE *f, uint64_t n) {
    static const unsigned char zeros[PAYLOAD_ALIGN];
    while (n) {
        size_t chunk = n > PAYLOAD_ALIGN ? PAYLOAD_ALIGN : (size_t)n;
        write_bytes(f, zeros, chunk);
        n -= chunk;
    }
}

static void plan_layout(const ModelFile *mf, const Options *o, Plan *plan) {
    uint64_t offset = align_up(HEADER_BYTES + (uint64_t)mf->count * RECORD_BYTES);
    uint32_t i;
    for (i = 0; i < mf->count; ++i) {
        const Tensor *t = &mf->tensors[i];
        uint64_t nb = (t->cols + QK - 1) / QK;
        if (t->alias) continue; /* resolved below, after its target */
        int convert = t->dtype == DTYPE_Q8 && !keep_q8(o, t->name);
        plan[i].dtype = convert && !o->simulate_bits ? DTYPE_Q5 : t->dtype;
        plan[i].simulate = convert && o->simulate_bits;
        plan[i].bytes = plan[i].dtype == DTYPE_Q5 ? t->rows * nb * Q5_BLOCK_BYTES : t->bytes;
        plan[i].offset = offset;
        offset = align_up(offset + plan[i].bytes);
    }
    for (i = 0; i < mf->count; ++i)
        if (mf->tensors[i].alias) plan[i] = plan[0]; /* lm_head aliases embed (index 0) */
}

static void write_directory(FILE *f, const ModelFile *mf, const Plan *plan) {
    unsigned char header[HEADER_BYTES], rec[RECORD_BYTES];
    uint32_t i;
    memset(header, 0, sizeof(header));
    memcpy(header, "SMOEQ8\0", 7);
    header[7] = (unsigned char)mf->version;
    put_u32(header + 8, mf->count);
    write_bytes(f, header, sizeof(header));
    for (i = 0; i < mf->count; ++i) {
        const Tensor *t = &mf->tensors[i];
        memset(rec, 0, sizeof(rec));
        memcpy(rec, t->name, TENSOR_NAME_SIZE);
        put_u32(rec + 40, t->rows);
        put_u32(rec + 44, t->cols);
        put_u32(rec + 48, plan[i].dtype);
        put_u32(rec + 52, t->alias);
        put_u64(rec + 56, plan[i].offset);
        put_u64(rec + 64, plan[i].bytes);
        write_bytes(f, rec, sizeof(rec));
    }
}

/* Q8 block (f32 scale, int8) whose values lie on a signed N-bit grid; the
 * scale search mirrors quantize_q5_row(). */
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

static void write_tensor(FILE *f, const Tensor *t, const Plan *p, int bits, float *row, unsigned char *packed) {
    uint32_t r, nb = (t->cols + QK - 1) / QK;
    if (p->simulate) {
        for (r = 0; r < t->rows; ++r) {
            dequantize_q8_row(t->data + (size_t)r * nb * Q8_BLOCK_BYTES, t->cols, row);
            quantize_grid_row(row, t->cols, bits, packed);
            write_bytes(f, packed, (size_t)nb * Q8_BLOCK_BYTES);
        }
        return;
    }
    if (p->dtype == t->dtype) {
        write_bytes(f, t->data, (size_t)t->bytes);
        return;
    }
    for (r = 0; r < t->rows; ++r) {
        dequantize_q8_row(t->data + (size_t)r * nb * Q8_BLOCK_BYTES, t->cols, row);
        quantize_q5_row(row, t->cols, packed);
        write_bytes(f, packed, (size_t)nb * Q5_BLOCK_BYTES);
    }
}

int main(int argc, char **argv) {
    Options o;
    Memory mem;
    MappedFile mapped;
    ModelFile mf;
    Pool *pool;
    Plan *plan;
    FILE *f;
    char *tmp;
    float *row;
    unsigned char *packed;
    uint64_t position, before = 0, after = 0;
    uint32_t i;

    platform_init(&argc, &argv);
    parse(argc, argv, &o);
    memset(&mem, 0, sizeof(mem));
    mem.limit = (size_t)256 << 20;
    pool = pool_create(cpu_physical_cores(), &mem);
    map_file(o.in, &mapped);
    modelfile_parse(&mf, mapped.data, mapped.size, &mem);
    modelfile_verify_payload(&mf, pool);
    plan = (Plan *)mem_alloc(&mem, mf.count, sizeof(Plan));
    plan_layout(&mf, &o, plan);

    tmp = (char *)mem_alloc(&mem, strlen(o.out) + 5, 1);
    snprintf(tmp, strlen(o.out) + 5, "%s.tmp", o.out);
    f = open_file(tmp, "wb");
    if (!f) die("cannot create %s: %s", tmp, strerror(errno));
    row = (float *)mem_alloc(&mem, 65536, sizeof(float));
    packed = (unsigned char *)mem_alloc(&mem, (65536 / QK) * Q8_BLOCK_BYTES, 1);
    write_directory(f, &mf, plan);
    position = HEADER_BYTES + (uint64_t)mf.count * RECORD_BYTES;
    for (i = 0; i < mf.count; ++i) {
        const Tensor *t = &mf.tensors[i];
        if (t->alias) continue;
        write_zeros(f, plan[i].offset - position);
        write_tensor(f, t, &plan[i], o.simulate_bits, row, packed);
        position = plan[i].offset + plan[i].bytes;
        before += t->bytes;
        after += plan[i].bytes;
    }
    if (fclose(f)) die("cannot finish %s: %s", tmp, strerror(errno));
    remove(o.out);
    if (rename(tmp, o.out)) die("cannot rename %s to %s: %s", tmp, o.out, strerror(errno));
    fprintf(stderr, "{\"tensors\":%u,\"payload_bytes_before\":%llu,\"payload_bytes_after\":%llu,\"ratio\":%.4f}\n",
            mf.count, (unsigned long long)before, (unsigned long long)after, before ? (double)after / (double)before : 0);
    pool_destroy(pool);
    unmap_file(&mapped);
    mem_release(&mem);
    return 0;
}
