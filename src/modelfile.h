/* SMOEQ8 checkpoint container (docs/FORMAT.md), parsed from a mapping. */
#ifndef SMOLMOE_MODELFILE_H
#define SMOLMOE_MODELFILE_H

#include <stdint.h>

#include "threadpool.h"
#include "util.h"

#define TENSOR_NAME_SIZE 40u
#define DTYPE_F32 0u
#define DTYPE_Q8 1u
#define DTYPE_Q5 2u
#define DTYPE_BLOB 3u /* v3: opaque bytes named "__*" (embedded config, tokenizer) */
#define BLOB_CONFIG "__config__"
#define BLOB_TOKENIZER "__tokenizer__"
/* modelfile_find() wildcard: any quantized matrix type (Q8 or Q5). */
#define DTYPE_QUANTIZED 0xffffffffu

typedef struct Tensor {
    char name[TENSOR_NAME_SIZE];
    uint32_t rows, cols, dtype, alias;
    uint64_t offset, bytes;
    const unsigned char *data;
} Tensor;

typedef struct ModelFile {
    const unsigned char *base;
    uint64_t size;
    uint32_t version, count, blobs; /* count includes blobs */
    Tensor *tensors;
} ModelFile;

/* Validates header, sorted directory and extents. */
void modelfile_parse(ModelFile *mf, const unsigned char *data, uint64_t size, Memory *mem);

/* Checks every Q8 scale (finite, >= 0) and every F32 value (finite) once, in
 * parallel, so the hot path never needs per-access validation. */
void modelfile_verify_payload(const ModelFile *mf, Pool *pool);

/* Embedded blob by name, or NULL. */
const Tensor *modelfile_blob(const ModelFile *mf, const char *name);

const Tensor *modelfile_find(const ModelFile *mf, const char *name, uint32_t rows, uint32_t cols,
                             uint32_t dtype);

#endif
