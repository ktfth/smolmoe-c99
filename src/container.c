#include "container.h"
#include "util.h"

#include <errno.h>
#include <string.h>

#define HEADER_BYTES 16u
#define RECORD_BYTES 72u

void write_bytes(FILE *f, const void *p, size_t n) {
    if (n && fwrite(p, 1, n, f) != n) die("write failed: %s", strerror(errno));
}

void write_zeros(FILE *f, uint64_t n) {
    static const unsigned char zeros[4096];
    while (n) {
        size_t chunk = n > sizeof(zeros) ? sizeof(zeros) : (size_t)n;
        write_bytes(f, zeros, chunk);
        n -= chunk;
    }
}

void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

void put_u64(unsigned char *p, uint64_t v) {
    put_u32(p, (uint32_t)v);
    put_u32(p + 4, (uint32_t)(v >> 32));
}

void write_tensor_raw(FILE *f, const void *ctx) {
    const Tensor *t = (const Tensor *)ctx;
    write_bytes(f, t->data, (size_t)t->bytes);
}

OutRecord record_from_tensor(const Tensor *t) {
    OutRecord r;
    memset(&r, 0, sizeof(r));
    memcpy(r.name, t->name, TENSOR_NAME_SIZE);
    r.rows = t->rows;
    r.cols = t->cols;
    r.dtype = t->dtype;
    r.alias = t->alias;
    r.bytes = t->bytes;
    r.write = write_tensor_raw;
    r.ctx = t;
    return r;
}

static uint64_t align_up(uint64_t x) {
    return (x + CONTAINER_ALIGN - 1) / CONTAINER_ALIGN * CONTAINER_ALIGN;
}

static void assign_offsets(OutRecord *records, uint32_t count) {
    uint64_t offset = align_up(HEADER_BYTES + (uint64_t)count * RECORD_BYTES);
    const OutRecord *embed = NULL;
    uint32_t i;
    for (i = 0; i < count; ++i) {
        if (i && strcmp(records[i - 1].name, records[i].name) >= 0) die("internal error: records not sorted");
        if (records[i].alias) continue;
        records[i].offset = offset;
        offset = align_up(offset + records[i].bytes);
        if (!strcmp(records[i].name, "embed")) embed = &records[i];
    }
    for (i = 0; i < count; ++i) {
        if (!records[i].alias) continue;
        if (!embed || records[i].bytes != embed->bytes) die("internal error: alias without matching embed");
        records[i].offset = embed->offset;
    }
}

uint64_t container_write(FILE *f, uint32_t version, OutRecord *records, uint32_t count) {
    unsigned char header[HEADER_BYTES], rec[RECORD_BYTES];
    uint64_t position;
    uint32_t i;
    assign_offsets(records, count);
    memset(header, 0, sizeof(header));
    memcpy(header, "SMOEQ8\0", 7);
    header[7] = (unsigned char)version;
    put_u32(header + 8, count);
    write_bytes(f, header, sizeof(header));
    for (i = 0; i < count; ++i) {
        const OutRecord *r = &records[i];
        memset(rec, 0, sizeof(rec));
        memcpy(rec, r->name, TENSOR_NAME_SIZE);
        put_u32(rec + 40, r->rows);
        put_u32(rec + 44, r->cols);
        put_u32(rec + 48, r->dtype);
        put_u32(rec + 52, r->alias);
        put_u64(rec + 56, r->offset);
        put_u64(rec + 64, r->bytes);
        write_bytes(f, rec, sizeof(rec));
    }
    position = HEADER_BYTES + (uint64_t)count * RECORD_BYTES;
    for (i = 0; i < count; ++i) {
        const OutRecord *r = &records[i];
        if (r->alias) continue;
        write_zeros(f, r->offset - position);
        r->write(f, r->ctx);
        position = r->offset + r->bytes;
    }
    return position;
}
