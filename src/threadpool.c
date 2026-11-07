#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "threadpool.h"

/* WebAssembly without threads (the single-thread WASI build): the caller does
 * all the work. Results are the same for any thread count. */
#if defined(__wasm__) && !defined(__wasm_atomics__)

struct Pool {
    uint32_t threads;
};

Pool *pool_create(uint32_t threads, Memory *mem) {
    Pool *p = (Pool *)mem_alloc(mem, 1, sizeof(*p));
    (void)threads;
    p->threads = 1;
    return p;
}

void pool_run(Pool *p, PoolTask task, void *ctx) {
    (void)p;
    task(ctx, 0, 1);
}

uint32_t pool_threads(const Pool *p) {
    return p->threads;
}

void pool_destroy(Pool *p) {
    (void)p;
}

#else

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  include <process.h>
#  define thread_yield() SwitchToThread()
typedef HANDLE Thread;
typedef SRWLOCK Mutex;
typedef CONDITION_VARIABLE Cond;
#  define mutex_init(m) InitializeSRWLock(m)
#  define mutex_lock(m) AcquireSRWLockExclusive(m)
#  define mutex_unlock(m) ReleaseSRWLockExclusive(m)
#  define mutex_free(m) ((void)(m))
#  define cond_init(c) InitializeConditionVariable(c)
#  define cond_wait(c, m) SleepConditionVariableSRW((c), (m), INFINITE, 0)
#  define cond_broadcast(c) WakeAllConditionVariable(c)
#  define cond_free(c) ((void)(c))
#elif defined(__wasm__)
/* WebAssembly with threads (the browser build): the host starts worker
 * instances of this module on the same shared memory before main() runs.
 * Each enters sm_worker_main() on its own stack and waits in a slot until a
 * pool hands it work. Sleeping uses memory.atomic.wait, which the browser
 * allows in a Web Worker, where the host runs the module. */
#  define thread_yield() ((void)0)
#  define WASM_WORKERS 15u
#  define WASM_WORKER_STACK (128u * 1024u)
typedef int Thread; /* unused: the workers belong to the host */
#else
#  include <pthread.h>
#  include <sched.h>
#  define thread_yield() sched_yield()
typedef pthread_t Thread;
typedef pthread_mutex_t Mutex;
typedef pthread_cond_t Cond;
#  define mutex_init(m) pthread_mutex_init((m), NULL)
#  define mutex_lock(m) pthread_mutex_lock(m)
#  define mutex_unlock(m) pthread_mutex_unlock(m)
#  define mutex_free(m) pthread_mutex_destroy(m)
#  define cond_init(c) pthread_cond_init((c), NULL)
#  define cond_wait(c, m) pthread_cond_wait((c), (m))
#  define cond_broadcast(c) pthread_cond_broadcast(c)
#  define cond_free(c) pthread_cond_destroy(c)
#endif

#if defined(__x86_64__) || defined(__i386__)
#  define cpu_relax() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(__arm__)
#  define cpu_relax() __asm__ __volatile__("yield")
#else
#  define cpu_relax() ((void)0)
#endif

/* Workers spin briefly between dispatches (a forward pass issues hundreds),
 * then sleep when the pool goes idle. */
#define SPIN_LIMIT 20000u

typedef struct Worker {
    struct Pool *pool;
    uint32_t index;
} Worker;

struct Pool {
    uint32_t threads;
    PoolTask task;
    void *ctx;
    uint32_t generation;
    uint32_t done;
    uint32_t stop;
#if !defined(__wasm__)
    Mutex lock;
    Cond wake;
#endif
    Thread *handles;
    Worker *workers;
};

/* ---- sleeping, waking and starting workers: the per-platform part ------- */

#if defined(__wasm__)

/* A host worker's slot: the address of the Worker it runs, 0 while it waits. */
static uint32_t worker_slots[WASM_WORKERS];
static unsigned char worker_stacks[WASM_WORKERS][WASM_WORKER_STACK] __attribute__((aligned(16)));

/* The number of worker instances the host started. */
__attribute__((import_module("smolmoe"), import_name("workers"))) uint32_t host_workers(void);

static void wait_while(uint32_t *word, uint32_t value) {
    while (__atomic_load_n(word, __ATOMIC_ACQUIRE) == value)
        __builtin_wasm_memory_atomic_wait32((int32_t *)(void *)word, (int32_t)value, -1);
}

static void wake_all(uint32_t *word) {
    __builtin_wasm_memory_atomic_notify((int32_t *)(void *)word, UINT32_MAX);
}

static void pool_init_sync(Pool *p) { (void)p; }
static void pool_free_sync(Pool *p) { (void)p; }

static void pool_sleep(Pool *p, uint32_t seen) { wait_while(&p->generation, seen); }

static void pool_wake(Pool *p) {
    __atomic_add_fetch(&p->generation, 1, __ATOMIC_RELEASE);
    wake_all(&p->generation);
}

static uint32_t thread_limit(uint32_t threads) {
    uint32_t host = host_workers();
    uint32_t most = 1 + (host < WASM_WORKERS ? host : WASM_WORKERS);
    return threads < most ? threads : most;
}

static void start_worker(Pool *p, uint32_t i) {
    __atomic_store_n(&worker_slots[i - 1], (uint32_t)(uintptr_t)&p->workers[i], __ATOMIC_RELEASE);
    wake_all(&worker_slots[i - 1]);
}

/* The worker clears its slot when it leaves the pool. */
static void join_worker(Pool *p, uint32_t i) {
    uint32_t w;
    (void)p;
    while ((w = __atomic_load_n(&worker_slots[i - 1], __ATOMIC_ACQUIRE)) != 0)
        __builtin_wasm_memory_atomic_wait32((int32_t *)(void *)&worker_slots[i - 1], (int32_t)w, -1);
}

#else

static void pool_init_sync(Pool *p) {
    mutex_init(&p->lock);
    cond_init(&p->wake);
}

static void pool_free_sync(Pool *p) {
    cond_free(&p->wake);
    mutex_free(&p->lock);
}

static void pool_sleep(Pool *p, uint32_t seen) {
    mutex_lock(&p->lock);
    while (__atomic_load_n(&p->generation, __ATOMIC_ACQUIRE) == seen) cond_wait(&p->wake, &p->lock);
    mutex_unlock(&p->lock);
}

static void pool_wake(Pool *p) {
    mutex_lock(&p->lock);
    __atomic_add_fetch(&p->generation, 1, __ATOMIC_RELEASE);
    cond_broadcast(&p->wake);
    mutex_unlock(&p->lock);
}

static uint32_t thread_limit(uint32_t threads) { return threads; }

#endif

static void worker_loop(Worker *w) {
    Pool *p = w->pool;
    uint32_t seen = 0;
    for (;;) {
        uint32_t generation, spins = 0;
        while ((generation = __atomic_load_n(&p->generation, __ATOMIC_ACQUIRE)) == seen) {
            if (++spins < SPIN_LIMIT) {
                cpu_relax();
                continue;
            }
            pool_sleep(p, seen);
        }
        seen = generation;
        if (__atomic_load_n(&p->stop, __ATOMIC_ACQUIRE)) return;
        p->task(p->ctx, w->index, p->threads);
        __atomic_add_fetch(&p->done, 1, __ATOMIC_ACQ_REL);
    }
}

#if defined(__wasm__)

/* The stack top of host worker `slot`; the host sets the instance's stack
 * pointer to it before calling sm_worker_main. Uses no stack itself. */
__attribute__((export_name("sm_worker_stack"))) uintptr_t sm_worker_stack(uint32_t slot) {
    return slot < WASM_WORKERS ? (uintptr_t)worker_stacks[slot] + WASM_WORKER_STACK : 0;
}

/* A host worker's whole life: wait for a pool, run its tasks until the pool
 * stops, give the slot back, repeat. */
__attribute__((export_name("sm_worker_main"))) void sm_worker_main(uint32_t slot) {
    if (slot >= WASM_WORKERS) return;
    for (;;) {
        wait_while(&worker_slots[slot], 0);
        worker_loop((Worker *)(uintptr_t)__atomic_load_n(&worker_slots[slot], __ATOMIC_ACQUIRE));
        __atomic_store_n(&worker_slots[slot], 0, __ATOMIC_RELEASE);
        wake_all(&worker_slots[slot]);
    }
}

#else

#if defined(_WIN32)
static unsigned __stdcall thread_main(void *arg) {
    worker_loop((Worker *)arg);
    return 0;
}
#else
static void *thread_main(void *arg) {
    worker_loop((Worker *)arg);
    return NULL;
}
#endif

static void start_worker(Pool *p, uint32_t i) {
#if defined(_WIN32)
    p->handles[i] = (HANDLE)_beginthreadex(NULL, 0, thread_main, &p->workers[i], 0, NULL);
    if (!p->handles[i]) die("cannot create worker thread");
#else
    if (pthread_create(&p->handles[i], NULL, thread_main, &p->workers[i])) die("cannot create worker thread");
#endif
}

static void join_worker(Pool *p, uint32_t i) {
#if defined(_WIN32)
    WaitForSingleObject(p->handles[i], INFINITE);
    CloseHandle(p->handles[i]);
#else
    pthread_join(p->handles[i], NULL);
#endif
}

#endif

Pool *pool_create(uint32_t threads, Memory *mem) {
    Pool *p = (Pool *)mem_alloc(mem, 1, sizeof(*p));
    uint32_t i;
    if (!threads) threads = 1;
    threads = thread_limit(threads);
    p->threads = threads;
    pool_init_sync(p);
    p->handles = (Thread *)mem_alloc(mem, threads, sizeof(*p->handles));
    p->workers = (Worker *)mem_alloc(mem, threads, sizeof(*p->workers));
    for (i = 1; i < threads; ++i) {
        p->workers[i].pool = p;
        p->workers[i].index = i;
        start_worker(p, i);
    }
    return p;
}

void pool_run(Pool *p, PoolTask task, void *ctx) {
    uint32_t spins;
    if (p->threads == 1) {
        task(ctx, 0, 1);
        return;
    }
    p->task = task;
    p->ctx = ctx;
    __atomic_store_n(&p->done, 0, __ATOMIC_RELAXED);
    pool_wake(p);
    task(ctx, 0, p->threads);
    for (spins = 0; __atomic_load_n(&p->done, __ATOMIC_ACQUIRE) != p->threads - 1; ++spins) {
        /* Yield once spinning stops paying off (e.g. more threads than cores). */
        if (spins < SPIN_LIMIT) cpu_relax();
        else thread_yield();
    }
}

uint32_t pool_threads(const Pool *p) {
    return p->threads;
}

void pool_destroy(Pool *p) {
    uint32_t i;
    if (!p) return;
    if (p->threads > 1) {
        __atomic_store_n(&p->stop, 1, __ATOMIC_RELEASE);
        pool_wake(p);
        for (i = 1; i < p->threads; ++i) join_worker(p, i);
    }
    pool_free_sync(p);
}

#endif
