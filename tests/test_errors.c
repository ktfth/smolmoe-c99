/* Library errors must come back as a status and a message. The process keeps
 * running, so a second call after a failure still works. */
#include <stdio.h>
#include <string.h>

#include "config.h"
#include "desktop.h"
#include "modelfile.h"
#include "model.h"
#include "sampler.h"
#include "threadpool.h"
#include "tokenizer.h"

static int failures;

#define CHECK(cond, ...)                                         \
    do {                                                         \
        if (!(cond)) {                                           \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                        \
            fputc('\n', stderr);                                 \
            ++failures;                                          \
        }                                                        \
    } while (0)

static void put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

static void test_modelfile(void) {
    Memory mem;
    ModelFile mf;
    Pool *pool;
    unsigned char tiny[8], bad[16], nanfile[124];
    sm_status st;
    SmError err;
    memset(&mem, 0, sizeof(mem));
    memset(tiny, 0, sizeof(tiny));
    mem.limit = 1048576;

    st = modelfile_parse(&mf, tiny, sizeof(tiny), &mem);
    CHECK(st == SM_ERR_FORMAT, "truncated status %d", (int)st);
    CHECK(strcmp(sm_last_error(), "invalid checkpoint size") == 0, "size message: %s", sm_last_error());
    sm_error_get(&err);
    CHECK(err.status == SM_ERR_FORMAT && strstr(err.message, "invalid checkpoint size"), "SmError copy");

    memset(bad, 'X', sizeof(bad));
    st = modelfile_parse(&mf, bad, sizeof(bad), &mem);
    CHECK(st == SM_ERR_FORMAT, "magic status %d", (int)st);
    CHECK(strstr(sm_last_error(), "unsupported checkpoint format") != NULL, "magic message: %s", sm_last_error());

    /* A directory that names one tensor, then a NaN Q8 scale. Parsing accepts
     * it; the payload check must report the tensor and return. */
    memset(nanfile, 0, sizeof(nanfile));
    memcpy(nanfile, "SMOEQ8", 6);
    nanfile[7] = 1;
    put_u32(nanfile + 8, 1);
    memcpy(nanfile + 16, "embed", 5);
    put_u32(nanfile + 56, 1);
    put_u32(nanfile + 60, 32);
    put_u32(nanfile + 64, DTYPE_Q8);
    put_u32(nanfile + 72, 88);
    put_u32(nanfile + 80, 36);
    nanfile[88] = 0x00;
    nanfile[89] = 0x00;
    nanfile[90] = 0xc0;
    nanfile[91] = 0x7f; /* NaN scale, little-endian */
    st = modelfile_parse(&mf, nanfile, sizeof(nanfile), &mem);
    CHECK(st == SM_OK, "nan file did not parse: %s", sm_last_error());
    pool = pool_create(1, &mem);
    CHECK(pool != NULL, "pool: %s", sm_last_error());
    if (pool) {
        st = modelfile_verify_payload(&mf, pool);
        CHECK(st == SM_ERR_NUMERIC, "scale status %d (%s)", (int)st, sm_last_error());
        CHECK(strstr(sm_last_error(), "invalid scale in tensor embed") != NULL, "scale message: %s", sm_last_error());
        pool_destroy(pool);
    }
    mem_release(&mem);
}

static void test_config_and_tokenizer(void) {
    Memory mem;
    Config c;
    Tokenizer *t;
    const char *text = "architecture=nope\n";
    const unsigned char *bad = (const unsigned char *)"NOTBPE\n";
    sm_status st;
    memset(&mem, 0, sizeof(mem));
    mem.limit = 1048576;
    st = config_parse((const unsigned char *)text, strlen(text), &c);
    CHECK(st == SM_ERR_FORMAT, "config status %d", (int)st);
    CHECK(strstr(sm_last_error(), "unsupported architecture") != NULL, "config message: %s", sm_last_error());
    t = tokenizer_parse(bad, 7, 300, &mem);
    CHECK(t == NULL, "bad tokenizer opened");
    CHECK(strstr(sm_last_error(), "invalid tokenizer magic") != NULL, "tokenizer message: %s", sm_last_error());
    t = tokenizer_parse(bad, 7, 300, &mem);
    CHECK(t == NULL, "second bad tokenizer killed the process or opened");
    mem_release(&mem);
}

static void test_sampler_and_forward(void) {
    Memory mem;
    Sampler s;
    Model m;
    unsigned char banned[4] = {0, 0, 0, 0};
    uint32_t token = 1;
    sm_status st;
    memset(&mem, 0, sizeof(mem));
    mem.limit = 1048576;
    st = sampler_init(&s, 4, banned, 0.0f, 1, 0.9f, 0, &mem);
    CHECK(st == SM_ERR_RANGE, "seed status %d", (int)st);
    CHECK(strcmp(sm_last_error(), "seed must be nonzero") == 0, "seed message: %s", sm_last_error());

    memset(&m, 0, sizeof(m));
    m.batch = 1;
    m.context = 4;
    st = model_forward(&m, &token, 2, 0, LOGITS_NONE);
    CHECK(st == SM_ERR_RANGE, "batch status %d", (int)st);
    CHECK(strstr(sm_last_error(), "token batch exceeds context") != NULL, "batch message: %s", sm_last_error());
    st = model_forward(&m, &token, 1, 5, LOGITS_NONE);
    CHECK(st == SM_ERR_RANGE, "second forward status %d (%s)", (int)st, sm_last_error());
    mem_release(&mem);
}

static void noop_text(void *ctx, const unsigned char *bytes, uint32_t n) {
    (void)ctx;
    (void)bytes;
    (void)n;
}
static void noop_end(void *ctx) { (void)ctx; }
static void noop_notice(void *ctx, const char *line) {
    (void)ctx;
    (void)line;
}

static void test_desktop_malformed(void) {
    const char *path = "bad-checkpoint-test.smoe";
    unsigned char header[16];
    FILE *f;
    char err[DESKTOP_ERROR_BYTES];
    DesktopOptions o;
    Desktop *d;
    memset(header, 0, sizeof(header));
    memcpy(header, "SMOEQ8", 6);
    header[7] = 1;
    header[8] = 1; /* one tensor, no directory */
    f = fopen(path, "wb");
    CHECK(f != NULL, "cannot write %s", path);
    if (!f) return;
    CHECK(fwrite(header, 1, sizeof(header), f) == sizeof(header), "short write");
    fclose(f);

    memset(&o, 0, sizeof(o));
    o.path = path;
    o.sink.text = noop_text;
    o.sink.reply_end = noop_end;
    o.sink.notice = noop_notice;
    d = desktop_open(&o, err);
    CHECK(d == NULL, "malformed model opened");
    CHECK(strstr(err, "truncated checkpoint directory") != NULL, "desktop error: %s", err);
    desktop_close(d);

    d = desktop_open(&o, err);
    CHECK(d == NULL, "second open of the same file did not return");
    CHECK(strstr(err, "truncated checkpoint directory") != NULL, "second desktop error: %s", err);
    desktop_close(d);
    remove(path);
}

int main(void) {
    test_modelfile();
    test_config_and_tokenizer();
    test_sampler_and_forward();
    test_desktop_malformed();
    if (failures) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("ok\n");
    return 0;
}
