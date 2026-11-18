#include <stdint.h>

#include "config.h"
#include "fuzz_support.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    Config c;
    fuzz_prepare();
    if (config_parse(data, size, &c) == SM_OK) (void)c.dim;
    mem_release(&fuzz_mem);
    return 0;
}
