/* SMOEQ8 writer shared by smolmoe-quantize and smolmoe-bundle. */
#ifndef SMOLMOE_CONTAINER_H
#define SMOLMOE_CONTAINER_H

#include <stdint.h>
#include <stdio.h>

#include "modelfile.h"

#define CONTAINER_ALIGN 64u

typedef void (*PayloadWriter)(FILE *f, const void *ctx);

typedef struct OutRecord {
    char name[TENSOR_NAME_SIZE];
    uint32_t rows, cols, dtype, alias;
    uint64_t bytes;
    PayloadWriter write; /* must emit exactly `bytes` bytes; unused for aliases */
    const void *ctx;
    uint64_t offset;     /* assigned by container_write, relative to the container start */
} OutRecord;

/* Writes header, directory and 64-byte-aligned payloads. Records must be
 * sorted by name; an alias (lm_head) takes the extent of "embed".
 * Returns the number of bytes written. */
uint64_t container_write(FILE *f, uint32_t version, OutRecord *records, uint32_t count);

void write_bytes(FILE *f, const void *p, size_t n);
void write_zeros(FILE *f, uint64_t n);
void put_u32(unsigned char *p, uint32_t v);
void put_u64(unsigned char *p, uint64_t v);

/* Copies a mapped tensor or blob verbatim. ctx: const Tensor *. */
void write_tensor_raw(FILE *f, const void *ctx);

/* Builds the record of a tensor read from a parsed container. */
OutRecord record_from_tensor(const Tensor *t);

#endif
