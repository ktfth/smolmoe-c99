/* smolmoe command-line interface. */
#include <float.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "kernels.h"
#include "model.h"
#include "modelfile.h"
#include "platform.h"
#include "sampler.h"
#include "threadpool.h"
#include "tokenizer.h"
#include "util.h"

#define DEFAULT_SYSTEM "Responda somente em português do Brasil. Não misture outros idiomas. Seja breve e direto."
#define DEFAULT_CONTEXT 2048u
#define DEFAULT_BATCH 32u
#define MAX_BATCH 512u
#define MAX_THREADS 256u
#define PROMPT_BYTES_PER_TOKEN 4u

typedef struct Options {
    const char *model, *config, *tokenizer, *prompt, *prompt_file, *system;
    const char *stats, *trace, *token_ids;
    uint32_t tokens, context, memory_mib, seed, threads, batch, top_k;
    float temperature, top_p;
    int scalar, chat, hash_logits, tokenize_only;
} Options;

typedef struct Stats {
    double start, loaded, prefill_end, ended, first;
    uint32_t prompt_tokens, generated, processed;
    uint64_t logits_hash;
} Stats;

/* ---- UTF-8 aware streaming output -------------------------------------- */

/* Holds back an incomplete trailing UTF-8 sequence so consoles never receive
 * half a character. Byte content is unchanged; it is only delayed. */
typedef struct Output {
    FILE *f;
    unsigned char pending[4];
    uint32_t n;
} Output;

static uint32_t incomplete_tail(const unsigned char *p, uint32_t n) {
    uint32_t back;
    for (back = 1; back <= 3 && back <= n; ++back) {
        unsigned char c = p[n - back];
        uint32_t need;
        if ((c & 0xc0) == 0x80) continue;
        need = (c & 0xe0) == 0xc0 ? 2 : (c & 0xf0) == 0xe0 ? 3 : (c & 0xf8) == 0xf0 ? 4 : 1;
        return back < need ? back : 0;
    }
    return 0;
}

static void write_all(FILE *f, const unsigned char *p, size_t n) {
    if (n && fwrite(p, 1, n, f) != n) die("output write failed");
}

static void output_bytes(void *ctx, const unsigned char *p, uint32_t n) {
    Output *o = (Output *)ctx;
    unsigned char tail[8];
    uint32_t total = o->n + n, tl = total < 4 ? total : 4, i, hold;
    for (i = 0; i < tl; ++i) {
        uint32_t at = total - tl + i;
        tail[i] = at < o->n ? o->pending[at] : p[at - o->n];
    }
    hold = incomplete_tail(tail, tl);
    if (total - hold <= o->n) {
        write_all(o->f, o->pending, total - hold);
    } else {
        write_all(o->f, o->pending, o->n);
        write_all(o->f, p, total - hold - o->n);
    }
    memmove(o->pending, tail + tl - hold, hold);
    o->n = hold;
}

static void output_finish(Output *o) {
    write_all(o->f, o->pending, o->n);
    o->n = 0;
    if (fflush(o->f)) die("output flush failed");
}

/* ---- command line ------------------------------------------------------ */

static void usage(FILE *out) {
    fputs("smolmoe " SMOLMOE_VERSION " - CPU inference for SMOEQ8 checkpoints\n\n"
          "usage: smolmoe --model FILE --config FILE --tokenizer FILE --prompt TEXT [options]\n\n"
          "input\n"
          "  --prompt-file FILE   raw UTF-8/byte prompt instead of --prompt\n"
          "  --chat               wrap the prompt in a Qwen2 ChatML single-turn template\n"
          "  --system TEXT        ChatML system message (default: Brazilian Portuguese)\n"
          "generation\n"
          "  --tokens N           max new tokens (default 64)\n"
          "  --temperature X      0 = greedy (default), positive = sample\n"
          "  --top-k N            sample among the N most likely tokens (default 40, 0 = all)\n"
          "  --top-p X            nucleus sampling mass in (0,1] (default 0.95)\n"
          "  --seed N             nonzero sampling seed (default 1)\n"
          "resources\n"
          "  --context N          KV cache positions (default min(config, 2048))\n"
          "  --threads N          worker threads (default: physical cores)\n"
          "  --batch N            prompt tokens per forward pass (default 32)\n"
          "  --memory-mib N       heap ceiling, excluding mapped weights (default 512)\n"
          "  --scalar             disable SIMD kernels\n"
          "diagnostics\n"
          "  --stats FILE         JSON statistics (always printed to stderr)\n"
          "  --trace FILE         exact hexadecimal logits of every processed token\n"
          "  --hash-logits        FNV-1a64 of every raw logit, reported in the stats\n"
          "  --token-ids FILE     prompt and generated token IDs, one per line\n"
          "  --tokenize-only      encode and render the prompt; no model needed\n"
          "  --version, --help\n", out);
}

static void parse_args(int argc, char **argv, Options *o) {
    int a;
    memset(o, 0, sizeof(*o));
    o->system = DEFAULT_SYSTEM;
    o->tokens = 64;
    o->memory_mib = 512;
    o->seed = 1;
    o->batch = DEFAULT_BATCH;
    o->top_k = 40;
    o->top_p = 0.95f;
    for (a = 1; a < argc; ++a) {
        const char *opt = argv[a], *value;
        if (!strcmp(opt, "--help") || !strcmp(opt, "-h")) { usage(stdout); exit(0); }
        if (!strcmp(opt, "--version")) { puts("smolmoe " SMOLMOE_VERSION); exit(0); }
        if (!strcmp(opt, "--scalar")) { o->scalar = 1; continue; }
        if (!strcmp(opt, "--chat")) { o->chat = 1; continue; }
        if (!strcmp(opt, "--hash-logits")) { o->hash_logits = 1; continue; }
        if (!strcmp(opt, "--tokenize-only")) { o->tokenize_only = 1; continue; }
        if (a + 1 >= argc) die("missing argument for %s", opt);
        value = argv[++a];
        if (!strcmp(opt, "--model")) o->model = value;
        else if (!strcmp(opt, "--config")) o->config = value;
        else if (!strcmp(opt, "--tokenizer")) o->tokenizer = value;
        else if (!strcmp(opt, "--prompt")) o->prompt = value;
        else if (!strcmp(opt, "--prompt-file")) o->prompt_file = value;
        else if (!strcmp(opt, "--system")) o->system = value;
        else if (!strcmp(opt, "--stats")) o->stats = value;
        else if (!strcmp(opt, "--trace")) o->trace = value;
        else if (!strcmp(opt, "--token-ids")) o->token_ids = value;
        else if (!strcmp(opt, "--tokens")) o->tokens = parse_u32(value, MAX_CONTEXT, "tokens");
        else if (!strcmp(opt, "--context")) o->context = parse_u32(value, MAX_CONTEXT, "context");
        else if (!strcmp(opt, "--memory-mib")) o->memory_mib = parse_u32(value, 1048576, "memory-mib");
        else if (!strcmp(opt, "--seed")) o->seed = parse_u32(value, UINT32_MAX, "seed");
        else if (!strcmp(opt, "--threads")) o->threads = parse_u32(value, MAX_THREADS, "threads");
        else if (!strcmp(opt, "--batch")) o->batch = parse_u32(value, MAX_BATCH, "batch");
        else if (!strcmp(opt, "--top-k")) o->top_k = parse_u32(value, UINT32_MAX, "top-k");
        else if (!strcmp(opt, "--top-p")) {
            double p = parse_real(value, "top-p");
            if (!(p > 0 && p <= 1)) die("top-p must be in (0, 1]");
            o->top_p = (float)p;
        } else if (!strcmp(opt, "--temperature")) {
            double t = parse_real(value, "temperature");
            if (t < 0 || t > 100 || (t > 0 && t < 1e-6)) die("temperature must be 0 or in [1e-6, 100]");
            o->temperature = (float)t;
        } else {
            die("unknown option: %s", opt);
        }
    }
    if (!o->config || !o->tokenizer || (!o->model && !o->tokenize_only) || (!o->prompt && !o->prompt_file) ||
        (o->prompt && o->prompt_file)) {
        usage(stderr);
        exit(1);
    }
    if (!o->seed || !o->memory_mib || !o->batch) die("seed, memory budget and batch must be nonzero");
}

static void check_output_paths(const Options *o) {
    const char *inputs[4], *outputs[3];
    size_t i, j;
    inputs[0] = o->model; inputs[1] = o->config; inputs[2] = o->tokenizer; inputs[3] = o->prompt_file;
    outputs[0] = o->stats; outputs[1] = o->trace; outputs[2] = o->token_ids;
    for (i = 0; i < 3; ++i) {
        if (!outputs[i]) continue;
        for (j = 0; j < 4; ++j)
            if (same_file(outputs[i], inputs[j])) die("output path aliases an input: %s", outputs[i]);
        for (j = 0; j < i; ++j)
            if (same_file(outputs[i], outputs[j])) die("output paths alias each other: %s", outputs[i]);
    }
}

static uint32_t read_prompt(const Options *o, unsigned char *buf, uint32_t cap) {
    size_t n;
    if (o->prompt_file) {
        FILE *f = open_file(o->prompt_file, "rb");
        if (!f) die("cannot open prompt file %s", o->prompt_file);
        n = fread(buf, 1, cap, f);
        if (ferror(f)) die("prompt read error");
        if (n == cap && fgetc(f) != EOF) die("prompt too long: at most %u bytes for this --context", cap);
        fclose(f);
    } else {
        n = strlen(o->prompt);
        if (n > cap) die("prompt too long: at most %u bytes for this --context", cap);
        memcpy(buf, o->prompt, n);
    }
    return (uint32_t)n;
}

static FILE *open_output(const char *path) {
    FILE *f;
    if (!path) return NULL;
    f = open_file(path, "wb");
    if (!f) die("cannot open output %s", path);
    return f;
}

static void close_output(FILE *f, const char *what) {
    if (f && fclose(f)) die("%s close failed", what);
}

/* ---- diagnostics ------------------------------------------------------- */

static void trace_logits(FILE *f, uint32_t pos, const float *logits, uint32_t vocab) {
    uint32_t i;
    fprintf(f, "%u", pos);
    for (i = 0; i < vocab; ++i) fprintf(f, " %a", (double)logits[i]);
    fputc('\n', f);
    if (ferror(f)) die("logit trace write failed");
}

static uint64_t hash_logits(uint64_t h, const float *logits, uint32_t vocab) {
    uint32_t i, j;
    for (i = 0; i < vocab; ++i) {
        uint32_t bits;
        memcpy(&bits, logits + i, 4);
        for (j = 0; j < 4; ++j) { h ^= (bits >> (j * 8)) & 255u; h *= UINT64_C(1099511628211); }
    }
    return h;
}

static void record_logits(const Options *o, FILE *trace, Stats *st, const float *logits, uint32_t rows,
                          uint32_t pos0, uint32_t vocab) {
    uint32_t r;
    for (r = 0; r < rows; ++r) {
        const float *row = logits + (size_t)r * vocab;
        if (trace) trace_logits(trace, pos0 + r, row, vocab);
        if (o->hash_logits) st->logits_hash = hash_logits(st->logits_hash, row, vocab);
    }
}

static void print_stats(FILE *f, const char *backend, const Options *o, const Model *m, const Memory *mem,
                        uint64_t mapped, const Stats *st) {
    double prefill = st->prefill_end - st->loaded, decode = st->ended - st->prefill_end;
    uint32_t decode_forwards = st->processed - st->prompt_tokens;
    fprintf(f,
            "{\"version\":\"%s\",\"backend\":\"%s\",\"threads\":%u,\"batch\":%u,"
            "\"budget_bytes\":%zu,\"tracked_peak_bytes\":%zu,\"rss_peak_bytes\":%" PRIu64 ","
            "\"mapped_bytes\":%" PRIu64 ",\"kv_bytes\":%zu,"
            "\"prompt_tokens\":%u,\"generated_tokens\":%u,\"processed_tokens\":%u,"
            "\"load_seconds\":%.6f,\"prefill_seconds\":%.6f,\"decode_seconds\":%.6f,\"ttft_seconds\":%.6f,"
            "\"prefill_tokens_per_second\":%.3f,\"decode_tokens_per_second\":%.3f,"
            "\"expert_calls\":%" PRIu64 ",\"logits_hash\":\"%016" PRIx64 "\"}\n",
            SMOLMOE_VERSION, backend, pool_threads(m->pool), m->batch, mem->limit, mem->peak, rss_peak_bytes(),
            mapped, model_kv_bytes(m), st->prompt_tokens, st->generated, st->processed,
            st->loaded - st->start, prefill, decode, st->first > 0 ? st->first - st->start : 0,
            prefill > 0 ? st->prompt_tokens / prefill : 0, decode > 0 ? decode_forwards / decode : 0,
            m->expert_calls, o->hash_logits ? st->logits_hash : 0);
    if (ferror(f)) die("statistics write failed");
}

/* ---- main -------------------------------------------------------------- */

static uint32_t default_threads(void) {
    uint32_t n = cpu_physical_cores();
    return n > MAX_THREADS ? MAX_THREADS : n;
}

int main(int argc, char **argv) {
    Options o;
    Stats st;
    Memory mem;
    Config c;
    Tokenizer *tok;
    MappedFile mapped;
    ModelFile mf;
    Model model;
    Sampler sampler;
    Pool *pool;
    Output out;
    FILE *trace, *ids_out = NULL, *stats_out;
    unsigned char *prompt, *piece, *banned;
    uint32_t *ids, prompt_cap, id_cap, np, pos, i, context;
    const char *backend;

    memset(&st, 0, sizeof(st));
    st.start = now_seconds();
    st.logits_hash = UINT64_C(14695981039346656037);
    platform_init(&argc, &argv);
    parse_args(argc, argv, &o);
    check_output_paths(&o);
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24) die("requires IEEE binary32 floats");

    memset(&mem, 0, sizeof(mem));
    mem.limit = mul_size(o.memory_mib, 1048576);
    c = config_read(o.config);
    context = o.context ? o.context : (c.max_context < DEFAULT_CONTEXT ? c.max_context : DEFAULT_CONTEXT);
    if (context > c.max_context) die("--context exceeds the model's max_context (%u)", c.max_context);

    tok = tokenizer_load(o.tokenizer, c.vocab, &mem);
    if (o.chat && !tokenizer_supports_chat(tok)) die("--chat requires a Qwen tokenizer");
    prompt_cap = (uint32_t)(context < MAX_CONTEXT / PROMPT_BYTES_PER_TOKEN ? context * PROMPT_BYTES_PER_TOKEN : MAX_CONTEXT);
    id_cap = tokenizer_prepare(tok, prompt_cap, &mem);
    prompt = (unsigned char *)mem_alloc(&mem, prompt_cap ? prompt_cap : 1, 1);
    ids = (uint32_t *)mem_alloc(&mem, id_cap, sizeof(uint32_t));
    piece = (unsigned char *)mem_alloc(&mem, tokenizer_max_piece(tok), 1);
    np = tokenizer_encode(tok, prompt, read_prompt(&o, prompt, prompt_cap), ids, o.chat, o.system);
    if (np > context) die("encoded prompt (%u tokens) exceeds --context (%u)", np, context);
    out.f = stdout;
    out.n = 0;

    ids_out = open_output(o.token_ids);
    if (ids_out)
        for (i = 0; i < np; ++i) fprintf(ids_out, "%u\n", ids[i]);
    if (o.tokenize_only) {
        tokenizer_render_prompt(tok, ids, np, piece, output_bytes, &out);
        output_finish(&out);
        close_output(ids_out, "token ID output");
        mem_release(&mem);
        return 0;
    }
    if (o.tokens > context - np) die("prompt + --tokens exceeds --context; increase context or reduce tokens");

    backend = kernels_select(o.scalar);
    pool = pool_create(o.threads ? o.threads : default_threads(), &mem);
    map_file(o.model, &mapped);
    modelfile_parse(&mf, mapped.data, mapped.size, &mem);
    modelfile_verify_payload(&mf, pool);
    {
        uint32_t batch = o.batch < context ? o.batch : context;
        int every_logit = o.trace || o.hash_logits;
        model_init(&model, &c, &mf, context, batch, every_logit ? batch : 1, pool, &mem);
    }
    banned = (unsigned char *)mem_alloc(&mem, c.vocab, 1);
    for (i = 0; i < c.vocab; ++i) banned[i] = (unsigned char)tokenizer_is_control(tok, i);
    sampler_init(&sampler, c.vocab, banned, o.temperature, o.top_k, o.top_p, o.seed, &mem);
    trace = open_output(o.trace);
    st.loaded = now_seconds();

    /* Prefill in batches; logits are only needed for the final prompt token
     * unless every logit is being traced or hashed. */
    for (pos = 0; pos < np;) {
        uint32_t n = np - pos < model.batch ? np - pos : model.batch;
        int last = pos + n == np;
        LogitsMode mode = (trace || o.hash_logits) ? LOGITS_ALL : last ? LOGITS_LAST : LOGITS_NONE;
        model_forward(&model, ids + pos, n, pos, mode);
        if (mode == LOGITS_ALL) record_logits(&o, trace, &st, model.logits, n, pos, c.vocab);
        if (mode == LOGITS_ALL && last) memmove(model.logits, model.logits + (size_t)(n - 1) * c.vocab, c.vocab * sizeof(float));
        pos += n;
    }
    st.processed = st.prompt_tokens = np;
    st.prefill_end = now_seconds();

    for (i = 0; i < o.tokens; ++i) {
        uint32_t token = sampler_next(&sampler, model.logits), len;
        if (ids_out) fprintf(ids_out, "%u\n", token);
        if (token == c.eos) break;
        len = tokenizer_decode(tok, token, piece);
        output_bytes(&out, piece, len);
        if (fflush(stdout)) die("output flush failed");
        ++st.generated;
        if (st.first == 0) st.first = now_seconds();
        if (i + 1 < o.tokens) {
            LogitsMode mode = (trace || o.hash_logits) ? LOGITS_ALL : LOGITS_LAST;
            model_forward(&model, &token, 1, pos, mode);
            if (mode == LOGITS_ALL) record_logits(&o, trace, &st, model.logits, 1, pos, c.vocab);
            ++pos;
            ++st.processed;
        }
    }
    output_finish(&out);
    st.ended = now_seconds();

    close_output(trace, "trace");
    close_output(ids_out, "token ID output");
    print_stats(stderr, backend, &o, &model, &mem, mapped.size, &st);
    stats_out = open_output(o.stats);
    if (stats_out) {
        print_stats(stats_out, backend, &o, &model, &mem, mapped.size, &st);
        close_output(stats_out, "stats");
    }
    pool_destroy(pool);
    unmap_file(&mapped);
    mem_release(&mem);
    return 0;
}
