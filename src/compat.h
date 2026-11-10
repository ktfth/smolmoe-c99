/* Compiler extensions the runtime actually uses, in one header.
 *
 * GCC and Clang: __atomic_*, __builtin_cpu_supports, __attribute__((target)).
 * MSVC has none of those. The branches below call Interlocked* and __cpuid so
 * a port can compile this seam; the SIMD kernels themselves are still written
 * for GCC and Clang (per-function AVX2 via SM_TARGET_AVX2) and are not built
 * with MSVC today. The project is C99 plus those extensions, not strict C99.
 *
 * Every atomic here is a 32-bit unsigned word. MSVC's Interlocked* are full
 * barriers, which is stronger than the requested order and still correct.
 * The order argument is the GCC/Clang memory order on those compilers and is
 * ignored on MSVC. */
#ifndef SMOLMOE_COMPAT_H
#define SMOLMOE_COMPAT_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#  define SM_ATOMIC_RELAXED __ATOMIC_RELAXED
#  define SM_ATOMIC_ACQUIRE __ATOMIC_ACQUIRE
#  define SM_ATOMIC_RELEASE __ATOMIC_RELEASE
#  define SM_ATOMIC_ACQ_REL __ATOMIC_ACQ_REL
#  define SM_TARGET_AVX2 __attribute__((target("avx2,f16c")))
#  define sm_load_u32(p, order) __atomic_load_n((p), (order))
#  define sm_store_u32(p, v, order) __atomic_store_n((p), (v), (order))
#  define sm_add_fetch_u32(p, v, order) __atomic_add_fetch((p), (v), (order))
/* Strong compare-and-swap. Stores the observed value in *expected on failure. */
#  define sm_cas_u32(p, expected, desired, order) \
      __atomic_compare_exchange_n((p), (expected), (desired), 0, (order), (order))
#elif defined(_MSC_VER)
#  include <intrin.h>
#  define SM_ATOMIC_RELAXED 0
#  define SM_ATOMIC_ACQUIRE 1
#  define SM_ATOMIC_RELEASE 2
#  define SM_ATOMIC_ACQ_REL 3
/* MSVC has no per-function target attribute. Compile AVX2 units with /arch:AVX2. */
#  define SM_TARGET_AVX2
#  define sm_load_u32(p, order) sm_load_u32_msvc(p, order)
#  define sm_store_u32(p, v, order) sm_store_u32_msvc((p), (v), (order))
#  define sm_add_fetch_u32(p, v, order) sm_add_fetch_u32_msvc((p), (v), (order))
#  define sm_cas_u32(p, expected, desired, order) sm_cas_u32_msvc((p), (expected), (desired), (order))

static inline uint32_t sm_load_u32_msvc(uint32_t *p, int order) {
    (void)order;
    return (uint32_t)_InterlockedCompareExchange((volatile long *)(void *)p, 0, 0);
}

static inline void sm_store_u32_msvc(uint32_t *p, uint32_t v, int order) {
    (void)order;
    (void)_InterlockedExchange((volatile long *)(void *)p, (long)v);
}

static inline uint32_t sm_add_fetch_u32_msvc(uint32_t *p, uint32_t v, int order) {
    (void)order;
    return (uint32_t)_InterlockedExchangeAdd((volatile long *)(void *)p, (long)v) + v;
}

static inline int sm_cas_u32_msvc(uint32_t *p, uint32_t *expected, uint32_t desired, int order) {
    uint32_t saw;
    (void)order;
    saw = (uint32_t)_InterlockedCompareExchange((volatile long *)(void *)p, (long)desired, (long)*expected);
    if (saw == *expected) return 1;
    *expected = saw;
    return 0;
}
#else
#  define SM_ATOMIC_RELAXED 0
#  define SM_ATOMIC_ACQUIRE 1
#  define SM_ATOMIC_RELEASE 2
#  define SM_ATOMIC_ACQ_REL 3
#  define SM_TARGET_AVX2
#  define sm_load_u32(p, order) sm_load_u32_plain(p, order)
#  define sm_store_u32(p, v, order) sm_store_u32_plain((p), (v), (order))
#  define sm_add_fetch_u32(p, v, order) sm_add_fetch_u32_plain((p), (v), (order))
#  define sm_cas_u32(p, expected, desired, order) sm_cas_u32_plain((p), (expected), (desired), (order))

/* Single-threaded fallback so the calls compile. Not safe for a real pool. */
static inline uint32_t sm_load_u32_plain(uint32_t *p, int order) {
    (void)order;
    return *p;
}
static inline void sm_store_u32_plain(uint32_t *p, uint32_t v, int order) {
    (void)order;
    *p = v;
}
static inline uint32_t sm_add_fetch_u32_plain(uint32_t *p, uint32_t v, int order) {
    (void)order;
    *p += v;
    return *p;
}
static inline int sm_cas_u32_plain(uint32_t *p, uint32_t *expected, uint32_t desired, int order) {
    (void)order;
    if (*p == *expected) {
        *p = desired;
        return 1;
    }
    *expected = *p;
    return 0;
}
#endif

/* Runtime AVX2+F16C check. False where the compiler has no such check.
 * GCC's builtin also requires the OS to have enabled the YMM state; the MSVC
 * path checks OSXSAVE and XCR0 the same way, or the first AVX instruction
 * would fault. */
static inline int sm_cpu_supports_avx2_f16c(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int leaf1[4], leaf7[4];
    unsigned int ecx, ebx, xcr0;
    __cpuid(leaf1, 1);
    ecx = (unsigned int)leaf1[2];
    if ((ecx & (1u << 27)) == 0 || (ecx & (1u << 28)) == 0 || (ecx & (1u << 29)) == 0) return 0;
    xcr0 = (unsigned int)_xgetbv(0);
    if ((xcr0 & 6u) != 6u) return 0;
    __cpuidex(leaf7, 7, 0);
    ebx = (unsigned int)leaf7[1];
    return (ebx & (1u << 5)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
#else
    return 0;
#endif
}

#endif
