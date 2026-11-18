/* libFuzzer helpers. A rejected file is a normal input: parsers return a status
 * instead of exiting, and the harness releases the arena and continues. */
#ifndef FUZZ_SUPPORT_H
#define FUZZ_SUPPORT_H

#include <string.h>

#include "util.h"

static Memory fuzz_mem;

static void fuzz_prepare(void) {
    memset(&fuzz_mem, 0, sizeof(fuzz_mem));
    fuzz_mem.limit = (size_t)32 << 20;
}

#endif
