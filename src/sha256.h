/* SHA-256 (FIPS 180-4), for the inventory's change detection. */
#ifndef SMOLMOE_SHA256_H
#define SMOLMOE_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct Sha256 {
    uint32_t h[8];
    uint64_t bytes;
    unsigned char block[64];
    size_t used;
} Sha256;

void sha256_init(Sha256 *s);
void sha256_update(Sha256 *s, const void *data, size_t n);
void sha256_final(Sha256 *s, unsigned char digest[32]);

/* Lowercase hex of the first `nbytes` bytes of the digest of data; out holds 2*nbytes+1. */
void sha256_hex(const void *data, size_t n, size_t nbytes, char *out);

#endif
