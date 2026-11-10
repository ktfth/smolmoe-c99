/* Self-contained executables: an SMOEQ8 v3 container appended to the runtime.
 *
 *   [executable][zero padding to 64 KiB][container][trailer]
 *   trailer (32 bytes) = "SMOEBDL1" | u64 offset | u64 length | "SMOEBDL1"
 *
 * The runtime maps its own executable and parses the container in place.
 * Windows will not start an executable of 4 GiB or more, so a model that
 * would cross that line is left as <stem>.smoe beside the executable
 * (the ".exe" suffix, any case, is not part of the stem). */
#ifndef SMOLMOE_BUNDLE_H
#define SMOLMOE_BUNDLE_H

#include <stddef.h>
#include <stdint.h>

#define BUNDLE_MAGIC "SMOEBDL1"
#define BUNDLE_TRAILER_BYTES 32u
#define BUNDLE_ALIGN 65536u
/* Windows refuses to launch a PE image whose size does not fit in 32 bits. */
#define BUNDLE_WIN_EXE_LIMIT 4294967296ull

/* Returns 1 and the container extent if the data ends with a valid trailer. */
int bundle_find(const unsigned char *data, uint64_t size, uint64_t *offset, uint64_t *length);
void bundle_trailer(unsigned char out[BUNDLE_TRAILER_BYTES], uint64_t offset, uint64_t length);

/* Writes "<directory><stem>.smoe" for an executable path. Returns 0 when the
 * path is empty, the stem is empty, or the result does not fit in cap. */
int bundle_sibling_path(const char *exe, char *out, size_t cap);

#endif
