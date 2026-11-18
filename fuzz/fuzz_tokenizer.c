#include <stdint.h>

#include "fuzz_support.h"
#include "tokenizer.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    Tokenizer *t;
    uint32_t cap, n, produced, piece_n;
    uint32_t *ids;
    unsigned char *piece;
    if (setjmp(fuzz_jmp)) return 0;
    fuzz_prepare();
    fuzz_cleanup = NULL;
    t = tokenizer_parse(data, size, 260, &fuzz_mem);
    cap = tokenizer_prepare(t, 64, &fuzz_mem);
    ids = (uint32_t *)mem_alloc(&fuzz_mem, cap, sizeof(uint32_t));
    n = size > 64 ? 64u : (uint32_t)size;
    produced = tokenizer_encode(t, data, n, ids, 0, NULL);
    piece_n = tokenizer_max_piece(t);
    piece = (unsigned char *)mem_alloc(&fuzz_mem, piece_n ? piece_n : 1, 1);
    if (produced) tokenizer_decode(t, ids[0], piece);
    (void)tokenizer_is_control(t, 0);
    mem_release(&fuzz_mem);
    return 0;
}
