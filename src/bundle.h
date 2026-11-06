/* Self-contained executables: an SMOEQ8 v3 container appended to the runtime.
 *
 *   [executable][zero padding to 64 KiB][container][trailer]
 *   trailer (32 bytes) = "SMOEBDL1" | u64 offset | u64 length | "SMOEBDL1"
 *
 * The runtime maps its own executable and parses the container in place. */
#ifndef SMOLMOE_BUNDLE_H
#define SMOLMOE_BUNDLE_H

#include <stdint.h>

#define BUNDLE_MAGIC "SMOEBDL1"
#define BUNDLE_TRAILER_BYTES 32u
#define BUNDLE_ALIGN 65536u

/* Returns 1 and the container extent if the data ends with a valid trailer. */
int bundle_find(const unsigned char *data, uint64_t size, uint64_t *offset, uint64_t *length);
void bundle_trailer(unsigned char out[BUNDLE_TRAILER_BYTES], uint64_t offset, uint64_t length);

#endif
