#include "modelfile.h"
#include "f16.h"
#include "kernels.h"

#include <math.h>
#include <string.h>

#define HEADER_BYTES 16u
#define RECORD_BYTES 72u
#define MAX_TENSORS 2000000u
#define MAX_TENSOR_COLS 65536u

static uint64_t block_bytes(uint32_t dtype) {
    return dtype == DTYPE_Q8 ? Q8_BLOCK_BYTES : Q5_BLOCK_BYTES;
}

static uint64_t expected_bytes(const Tensor *t) {
    if (t->dtype == DTYPE_BLOB) return t->bytes; /* any length */
    if (t->dtype != DTYPE_F32) return (uint64_t)t->rows * ((t->cols + QK - 1) / QK) * block_bytes(t->dtype);
    return (uint64_t)t->rows * t->cols * 4;
}

/* The only permitted alias: lm_head sharing embed's extent (tied weights). */
static void check_alias(const ModelFile *mf, const Tensor *t, uint32_t index) {
    const Tensor *e = NULL;
    uint32_t j;
    for (j = 0; j < index; ++j)
        if (!strcmp(mf->tensors[j].name, "embed")) e = &mf->tensors[j];
    if (!e || strcmp(t->name, "lm_head") || t->offset != e->offset || t->bytes != e->bytes ||
        t->rows != e->rows || t->cols != e->cols || t->dtype != e->dtype)
        die("invalid tied embedding alias");
}

static void check_metadata(const ModelFile *mf, const Tensor *t) {
    if (t->dtype == DTYPE_BLOB) {
        if (mf->version < 3 || t->rows != 1 || t->cols != 1 || t->alias || strncmp(t->name, "__", 2))
            die("invalid blob record: %s", t->name);
        return;
    }
    if (t->alias > 1 || (t->alias && mf->version < 2) || !t->rows || !t->cols || t->cols > MAX_TENSOR_COLS ||
        t->dtype > DTYPE_Q5 || !strncmp(t->name, "__", 2))
        die("invalid tensor metadata: %s", t->name);
}

void modelfile_parse(ModelFile *mf, const unsigned char *data, uint64_t size, Memory *mem) {
    uint64_t previous;
    uint32_t i;
    memset(mf, 0, sizeof(*mf));
    if (size < HEADER_BYTES) die("invalid checkpoint size");
    mf->base = data;
    mf->size = size;
    mf->version = data[7];
    if (memcmp(data, "SMOEQ8\0", 7) || mf->version < 1 || mf->version > 3 || rd_u32(data + 12))
        die("unsupported checkpoint format (expected SMOEQ8 v1-v3)");
    mf->count = rd_u32(data + 8);
    if (!mf->count || mf->count > MAX_TENSORS) die("invalid tensor count");
    previous = HEADER_BYTES + (uint64_t)mf->count * RECORD_BYTES;
    if (previous > size) die("truncated checkpoint directory");
    mf->tensors = (Tensor *)mem_alloc(mem, mf->count, sizeof(Tensor));
    for (i = 0; i < mf->count; ++i) {
        const unsigned char *rec = data + HEADER_BYTES + (uint64_t)i * RECORD_BYTES;
        Tensor *t = &mf->tensors[i];
        memcpy(t->name, rec, TENSOR_NAME_SIZE);
        if (!memchr(t->name, 0, TENSOR_NAME_SIZE) || !t->name[0]) die("invalid tensor name");
        t->rows = rd_u32(rec + 40);
        t->cols = rd_u32(rec + 44);
        t->dtype = rd_u32(rec + 48);
        t->alias = rd_u32(rec + 52);
        t->offset = rd_u64(rec + 56);
        t->bytes = rd_u64(rec + 64);
        check_metadata(mf, t);
        if (t->bytes != expected_bytes(t) || (!t->alias && t->offset < previous) || t->offset > size ||
            t->bytes > size - t->offset)
            die("invalid tensor extent: %s", t->name);
        if (t->alias) check_alias(mf, t, i);
        else previous = t->offset + t->bytes;
        if (i && strcmp(mf->tensors[i - 1].name, t->name) >= 0)
            die("tensor directory must have unique sorted names");
        if (t->dtype == DTYPE_BLOB) ++mf->blobs;
        t->data = data + t->offset;
    }
}

const Tensor *modelfile_blob(const ModelFile *mf, const char *name) {
    uint32_t i;
    for (i = 0; i < mf->count; ++i)
        if (mf->tensors[i].dtype == DTYPE_BLOB && !strcmp(mf->tensors[i].name, name)) return &mf->tensors[i];
    return NULL;
}

const Tensor *modelfile_find(const ModelFile *mf, const char *name, uint32_t rows, uint32_t cols,
                             uint32_t dtype) {
    uint32_t lo = 0, hi = mf->count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (strcmp(mf->tensors[mid].name, name) < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo == mf->count || strcmp(mf->tensors[lo].name, name)) die("missing tensor: %s", name);
    if (mf->tensors[lo].rows != rows || mf->tensors[lo].cols != cols ||
        (dtype == DTYPE_QUANTIZED ? mf->tensors[lo].dtype == DTYPE_F32 : mf->tensors[lo].dtype != dtype))
        die("wrong tensor shape/type: %s", name);
    return &mf->tensors[lo];
}

typedef struct VerifyTask {
    const ModelFile *mf;
    uint32_t bad; /* lowest bad tensor index + 1, or 0 */
} VerifyTask;

static int verify_units(const Tensor *t, uint64_t begin, uint64_t end) {
    uint64_t u;
    if (t->dtype == DTYPE_Q8) {
        for (u = begin; u < end; ++u) {
            float s = rd_f32(t->data + u * Q8_BLOCK_BYTES);
            if (!isfinite(s) || s < 0) return 0;
        }
    } else if (t->dtype == DTYPE_Q5) {
        /* Q5 scales are signed: the extreme value maps to the -8 side. */
        for (u = begin; u < end; ++u) {
            const unsigned char *p = t->data + u * Q5_BLOCK_BYTES;
            if (!isfinite(f16_to_f32((uint16_t)(p[0] | p[1] << 8)))) return 0;
        }
    } else {
        for (u = begin; u < end; ++u)
            if (!isfinite(rd_f32(t->data + u * 4))) return 0;
    }
    return 1;
}

static void verify_task(void *ctx, uint32_t index, uint32_t count) {
    VerifyTask *v = (VerifyTask *)ctx;
    uint32_t i;
    for (i = 0; i < v->mf->count; ++i) {
        const Tensor *t = &v->mf->tensors[i];
        uint64_t units = t->dtype == DTYPE_F32 ? t->bytes / 4 : t->bytes / block_bytes(t->dtype);
        uint64_t begin = units * index / count, end = units * (index + 1) / count;
        if (t->alias || t->dtype == DTYPE_BLOB) continue;
        if (!verify_units(t, begin, end)) {
            uint32_t expected = __atomic_load_n(&v->bad, __ATOMIC_RELAXED);
            while ((!expected || i + 1 < expected) &&
                   !__atomic_compare_exchange_n(&v->bad, &expected, i + 1, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
            return;
        }
    }
}

void modelfile_verify_payload(const ModelFile *mf, Pool *pool) {
    VerifyTask v;
    v.mf = mf;
    v.bad = 0;
    pool_run(pool, verify_task, &v);
    if (v.bad) die("non-finite value or invalid scale in tensor %s", mf->tensors[v.bad - 1].name);
}
