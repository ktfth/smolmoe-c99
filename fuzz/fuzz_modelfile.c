#include <stdint.h>

#include "fuzz_support.h"
#include "modelfile.h"
#include "threadpool.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    ModelFile mf;
    Pool *pool;
    fuzz_prepare();
    if (modelfile_parse(&mf, data, (uint64_t)size, &fuzz_mem) == SM_OK) {
        pool = pool_create(1, &fuzz_mem);
        if (pool) {
            (void)modelfile_verify_payload(&mf, pool);
            pool_destroy(pool);
        }
    }
    mem_release(&fuzz_mem);
    return 0;
}
