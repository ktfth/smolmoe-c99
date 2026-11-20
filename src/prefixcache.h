/* The KV cache of the chat's system prompt (the prefix) on disk, so a new chat
 * skips its prefill. A cache is used only after it proves itself: the last
 * prefix position is computed again over the cached keys and values, and its
 * logits must hash to the value stored when the cache was written. Another
 * model, prompt or a damaged file fails that check, and the prefix is
 * computed again and the cache rewritten. */
#ifndef SMOLMOE_PREFIXCACHE_H
#define SMOLMOE_PREFIXCACHE_H

#include <stdint.h>

#include "model.h"

typedef enum {
    PREFIX_CACHE_OFF,        /* no cache file asked for */
    PREFIX_CACHE_LOADED,     /* read and checked */
    PREFIX_CACHE_WRITTEN,    /* there was none: computed and written */
    PREFIX_CACHE_REBUILT,    /* there was one that failed the check: computed and rewritten */
    PREFIX_CACHE_UNWRITABLE  /* computed, but the file could not be written */
} PrefixCacheState;

/* Processes the n prefix tokens (n >= 1) at position 0: from the cache at path
 * when it passes the check, otherwise by prefill, then writes the cache. With
 * path NULL it is a plain prefill. When a forward pass fails, sm_failed() is
 * set and the state returned means nothing. */
PrefixCacheState prefix_cache_prefill(Model *m, const uint32_t *ids, uint32_t n, const char *path);

/* "carregado", "gravado", ... for the stats; NULL for PREFIX_CACHE_OFF. */
const char *prefix_cache_state_name(PrefixCacheState s);

#endif
