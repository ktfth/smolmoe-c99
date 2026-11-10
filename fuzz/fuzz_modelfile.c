#include <stdint.h>

#include "fuzz_support.h"
#include "modelfile.h"
#include "threadpool.h"

static Pool *fuzz_pool;

static void drop_pool(void) {
    if (!fuzz_pool) return;
    pool_destroy(fuzz_pool);
    fuzz_pool = NULL;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ModelFile mf;
    if (setjmp(fuzz_jmp)) return 0;
    fuzz_prepare();
    fuzz_pool = NULL;
    fuzz_cleanup = drop_pool;
    modelfile_parse(&mf, data, (uint64_t)size, &fuzz_mem);
    fuzz_pool = pool_create(1, &fuzz_mem);
    modelfile_verify_payload(&mf, fuzz_pool);
    drop_pool();
    mem_release(&fuzz_mem);
    return 0;
}
