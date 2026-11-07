#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#endif

#include "threadpool.h"

/* WebAssembly without threads (the WASI build the tests run): the caller does
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
 * then sleep on a condition variable when the pool goes idle. */
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
    Mutex lock;
    Cond wake;
    Thread *handles;
    Worker *workers;
};

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
            mutex_lock(&p->lock);
            while (__atomic_load_n(&p->generation, __ATOMIC_ACQUIRE) == seen) cond_wait(&p->wake, &p->lock);
            mutex_unlock(&p->lock);
        }
        seen = generation;
        if (__atomic_load_n(&p->stop, __ATOMIC_ACQUIRE)) return;
        p->task(p->ctx, w->index, p->threads);
        __atomic_add_fetch(&p->done, 1, __ATOMIC_ACQ_REL);
    }
}

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

Pool *pool_create(uint32_t threads, Memory *mem) {
    Pool *p = (Pool *)mem_alloc(mem, 1, sizeof(*p));
    uint32_t i;
    if (!threads) threads = 1;
    p->threads = threads;
    mutex_init(&p->lock);
    cond_init(&p->wake);
    p->handles = (Thread *)mem_alloc(mem, threads, sizeof(*p->handles));
    p->workers = (Worker *)mem_alloc(mem, threads, sizeof(*p->workers));
    for (i = 1; i < threads; ++i) {
        p->workers[i].pool = p;
        p->workers[i].index = i;
#if defined(_WIN32)
        p->handles[i] = (HANDLE)_beginthreadex(NULL, 0, thread_main, &p->workers[i], 0, NULL);
        if (!p->handles[i]) die("cannot create worker thread");
#else
        if (pthread_create(&p->handles[i], NULL, thread_main, &p->workers[i]))
            die("cannot create worker thread");
#endif
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
    mutex_lock(&p->lock);
    __atomic_add_fetch(&p->generation, 1, __ATOMIC_RELEASE);
    cond_broadcast(&p->wake);
    mutex_unlock(&p->lock);
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
        mutex_lock(&p->lock);
        __atomic_add_fetch(&p->generation, 1, __ATOMIC_RELEASE);
        cond_broadcast(&p->wake);
        mutex_unlock(&p->lock);
        for (i = 1; i < p->threads; ++i) {
#if defined(_WIN32)
            WaitForSingleObject(p->handles[i], INFINITE);
            CloseHandle(p->handles[i]);
#else
            pthread_join(p->handles[i], NULL);
#endif
        }
    }
    cond_free(&p->wake);
    mutex_free(&p->lock);
}

#endif
