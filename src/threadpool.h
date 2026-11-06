/* Fixed-size worker pool with static work partitioning.
 * Every task receives (index, count) and splits its own work deterministically,
 * so results never depend on scheduling. The caller thread acts as index 0. */
#ifndef SMOLMOE_THREADPOOL_H
#define SMOLMOE_THREADPOOL_H

#include <stdint.h>

#include "util.h"

typedef void (*PoolTask)(void *ctx, uint32_t index, uint32_t count);
typedef struct Pool Pool;

Pool *pool_create(uint32_t threads, Memory *mem);
void pool_run(Pool *pool, PoolTask task, void *ctx);
uint32_t pool_threads(const Pool *pool);
void pool_destroy(Pool *pool);

static inline void split_range(uint32_t total, uint32_t index, uint32_t count,
                               uint32_t *begin, uint32_t *end) {
    *begin = (uint32_t)((uint64_t)total * index / count);
    *end = (uint32_t)((uint64_t)total * (index + 1) / count);
}

#endif
