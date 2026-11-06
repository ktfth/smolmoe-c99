/* smolmoe-bundle: packs a model into a single file.
 *
 *   smolmoe-bundle --model m.smoe --config c.ini --tokenizer t.qbp --out single.smoe
 *       SMOEQ8 v3 container with the config and tokenizer embedded as blobs.
 *   smolmoe-bundle ... --exe build/smolmoe.exe --out qwen-pt.exe
 *       The runtime executable followed by that container: runs with no arguments.
 *
 * Everything is validated against the runtime parsers before and after writing. */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bundle.h"
#include "config.h"
#include "container.h"
#include "model.h"
#include "modelfile.h"
#include "platform.h"
#include "threadpool.h"
#include "tokenizer.h"
#include "util.h"

typedef struct Options {
    const char *model, *config, *tokenizer, *exe, *out;
} Options;

typedef struct Blob {
    const unsigned char *data;
    uint64_t size;
} Blob;

static void usage(void) {
    fputs("usage: smolmoe-bundle --model M.smoe [--config C.ini] [--tokenizer T] --out OUT [--exe RUNTIME]\n"
          "  Without --exe: writes a single-file SMOEQ8 v3 model (config and tokenizer embedded).\n"
          "  With --exe:    writes RUNTIME followed by that model: a self-contained executable.\n"
          "  --config/--tokenizer default to the blobs already embedded in M.smoe.\n", stderr);
    exit(1);
}

static void parse(int argc, char **argv, Options *o) {
    int a;
    memset(o, 0, sizeof(*o));
    for (a = 1; a < argc; ++a) {
        const char **slot = NULL;
        if (!strcmp(argv[a], "--model")) slot = &o->model;
        else if (!strcmp(argv[a], "--config")) slot = &o->config;
        else if (!strcmp(argv[a], "--tokenizer")) slot = &o->tokenizer;
        else if (!strcmp(argv[a], "--exe")) slot = &o->exe;
        else if (!strcmp(argv[a], "--out")) slot = &o->out;
        if (!slot || a + 1 >= argc) usage();
        *slot = argv[++a];
    }
    if (!o->model || !o->out) usage();
    if (same_file(o->out, o->model) || same_file(o->out, o->config) || same_file(o->out, o->tokenizer) ||
        same_file(o->out, o->exe))
        die("--out must differ from every input");
}

static void write_blob(FILE *f, const void *ctx) {
    const Blob *b = (const Blob *)ctx;
    write_bytes(f, b->data, (size_t)b->size);
}

/* An explicit file wins; otherwise reuse the blob embedded in the model. */
static Blob resolve_blob(const char *path, const ModelFile *mf, const char *name, MappedFile *holder) {
    Blob b;
    if (path) {
        map_file(path, holder);
        b.data = holder->data;
        b.size = holder->size;
    } else {
        const Tensor *t = modelfile_blob(mf, name);
        if (!t) die("%s is not embedded in the model; pass it explicitly", name);
        b.data = t->data;
        b.size = t->bytes;
    }
    return b;
}

static int by_name(const void *a, const void *b) {
    return strcmp(((const OutRecord *)a)->name, ((const OutRecord *)b)->name);
}

static OutRecord blob_record(const char *name, const Blob *b) {
    OutRecord r;
    memset(&r, 0, sizeof(r));
    snprintf(r.name, sizeof(r.name), "%s", name);
    r.rows = 1;
    r.cols = 1;
    r.dtype = DTYPE_BLOB;
    r.bytes = b->size;
    r.write = write_blob;
    r.ctx = b;
    return r;
}

/* Parses config and tokenizer and builds the model once (context 1), so a
 * bundle can never contain an inconsistent set. */
static void validate(const ModelFile *mf, const Blob *config, const Blob *tok, Pool *pool, Memory *mem) {
    Config c = config_parse(config->data, (size_t)config->size);
    Model model;
    tokenizer_parse(tok->data, (size_t)tok->size, c.vocab, mem);
    model_init(&model, &c, mf, 1, 1, 1, pool, mem);
}

static void copy_runtime(FILE *f, const char *exe, uint64_t *written) {
    MappedFile runtime;
    uint64_t offset, length, size;
    map_file(exe, &runtime);
    /* Re-bundling an already bundled runtime keeps only the program part. */
    size = bundle_find(runtime.data, runtime.size, &offset, &length) ? offset : runtime.size;
    write_bytes(f, runtime.data, (size_t)size);
    *written = size;
    unmap_file(&runtime);
}

static void verify_output(const char *path, int is_exe, Pool *pool, Memory *mem) {
    MappedFile out;
    ModelFile mf;
    uint64_t offset = 0, length;
    map_file(path, &out);
    length = out.size;
    if (is_exe && !bundle_find(out.data, out.size, &offset, &length)) die("internal error: bundle trailer missing");
    modelfile_parse(&mf, out.data + offset, length, mem);
    modelfile_verify_payload(&mf, pool);
    if (!modelfile_blob(&mf, BLOB_CONFIG) || !modelfile_blob(&mf, BLOB_TOKENIZER)) die("internal error: blobs missing");
    unmap_file(&out);
}

int main(int argc, char **argv) {
    Options o;
    Memory mem;
    MappedFile model_file, config_file, tok_file;
    ModelFile mf;
    Pool *pool;
    Blob config, tok;
    OutRecord *records;
    FILE *f;
    char *tmp;
    uint32_t i, n = 0;
    uint64_t base = 0, length;

    platform_init(&argc, &argv);
    parse(argc, argv, &o);
    memset(&mem, 0, sizeof(mem));
    memset(&config_file, 0, sizeof(config_file));
    memset(&tok_file, 0, sizeof(tok_file));
    mem.limit = (size_t)512 << 20;
    pool = pool_create(cpu_physical_cores(), &mem);
    map_file(o.model, &model_file);
    modelfile_parse(&mf, model_file.data, model_file.size, &mem);
    modelfile_verify_payload(&mf, pool);
    config = resolve_blob(o.config, &mf, BLOB_CONFIG, &config_file);
    tok = resolve_blob(o.tokenizer, &mf, BLOB_TOKENIZER, &tok_file);
    validate(&mf, &config, &tok, pool, &mem);

    records = (OutRecord *)mem_alloc(&mem, (size_t)mf.count + 2, sizeof(OutRecord));
    for (i = 0; i < mf.count; ++i)
        if (mf.tensors[i].dtype != DTYPE_BLOB) records[n++] = record_from_tensor(&mf.tensors[i]);
    records[n++] = blob_record(BLOB_CONFIG, &config);
    records[n++] = blob_record(BLOB_TOKENIZER, &tok);
    qsort(records, n, sizeof(*records), by_name);

    tmp = (char *)mem_alloc(&mem, strlen(o.out) + 5, 1);
    snprintf(tmp, strlen(o.out) + 5, "%s.tmp", o.out);
    f = open_file(tmp, "wb");
    if (!f) die("cannot create %s: %s", tmp, strerror(errno));
    if (o.exe) {
        uint64_t runtime_bytes;
        copy_runtime(f, o.exe, &runtime_bytes);
        base = (runtime_bytes + BUNDLE_ALIGN - 1) / BUNDLE_ALIGN * BUNDLE_ALIGN;
        write_zeros(f, base - runtime_bytes);
    }
    length = container_write(f, 3, records, n);
    if (o.exe) {
        unsigned char trailer[BUNDLE_TRAILER_BYTES];
        bundle_trailer(trailer, base, length);
        write_bytes(f, trailer, sizeof(trailer));
    }
    if (fclose(f)) die("cannot finish %s: %s", tmp, strerror(errno));
    unmap_file(&model_file);
    if (config_file.data) unmap_file(&config_file);
    if (tok_file.data) unmap_file(&tok_file);
    verify_output(tmp, o.exe != NULL, pool, &mem);
    if (!replace_file(tmp, o.out)) {
        remove(tmp);
        die("cannot replace %s (is it in use?)", o.out);
    }
    if (o.exe) make_executable(o.out);
    fprintf(stderr, "{\"container_bytes\":%llu,\"runtime_bytes\":%llu,\"self_contained\":%s}\n",
            (unsigned long long)length, (unsigned long long)base, o.exe ? "true" : "false");
    pool_destroy(pool);
    mem_release(&mem);
    return 0;
}
