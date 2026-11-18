/* libFuzzer helpers. die() exits; a handler longjmps back so a rejected file
 * is a normal input. The jump target has to be the fuzz entry itself: setjmp
 * in a function that then returns is not a valid target. */
#ifndef FUZZ_SUPPORT_H
#define FUZZ_SUPPORT_H

#include <setjmp.h>
#include <string.h>

#include "util.h"

static jmp_buf fuzz_jmp;
static Memory fuzz_mem;
static void (*fuzz_cleanup)(void);

static void fuzz_on_die(const char *message) {
    (void)message;
    if (fuzz_cleanup) fuzz_cleanup();
    mem_release(&fuzz_mem);
    longjmp(fuzz_jmp, 1);
}

static void fuzz_prepare(void) {
    memset(&fuzz_mem, 0, sizeof(fuzz_mem));
    fuzz_mem.limit = (size_t)32 << 20;
    die_set_handler(fuzz_on_die);
}

#endif
