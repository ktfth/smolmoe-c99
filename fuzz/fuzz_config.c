#include <stdint.h>

#include "config.h"
#include "fuzz_support.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    Config c;
    if (setjmp(fuzz_jmp)) return 0;
    fuzz_prepare();
    fuzz_cleanup = NULL;
    c = config_parse(data, size);
    (void)c.dim;
    mem_release(&fuzz_mem);
    return 0;
}
