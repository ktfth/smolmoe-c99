/* smolmoe command-line interface. */
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bundle.h"
#include "chat.h"
#include "config.h"
#include "kernels.h"
#include "model.h"
#include "modelfile.h"
#include "output.h"
#include "platform.h"
#include "sampler.h"
#include "threadpool.h"
#include "tokenizer.h"
#include "tools.h"
#include "util.h"

#define DEFAULT_SYSTEM "Responda somente em português do Brasil. Não misture outros idiomas. Seja breve e direto."
#define DEFAULT_CONTEXT 2048u
#define DEFAULT_BATCH 32u
#define DEFAULT_TOKENS 64u
#define DEFAULT_CHAT_TOKENS 512u
#define MAX_BATCH 512u
#define MAX_THREADS 256u
#define PROMPT_BYTES_PER_TOKEN 4u
#define TOOLS_PROMPT_BYTES 4096u /* the "# Tools" section the chat appends to --system */

typedef struct Options {
    const char *model, *config, *tokenizer, *prompt, *prompt_file, *system;
    const char *stats, *trace, *token_ids;
    uint32_t tokens, context, memory_mib, seed, threads, batch, top_k;
    float temperature, top_p;
    int scalar, chat, hash_logits, tokenize_only, perplexity, interactive, tokens_given, no_tools, fixed_clock, profile;
    int memory_given; /* --memory-mib: otherwise the KV cache is added to the default budget */
    ToolClock clock; /* --now */
} Options;

typedef struct Stats {
    double start, loaded, prefill_end, ended, first;
    uint32_t prompt_tokens, generated, processed, tool_calls;
    uint64_t logits_hash;
    double nll;
    uint32_t nll_tokens;
} Stats;

/* Where the weights (and possibly the embedded config/tokenizer) come from. */
typedef struct Source {
    MappedFile file;
    ModelFile mf;
    int present;
} Source;

/* ---- command line ------------------------------------------------------ */

static void usage(FILE *out) {
    fputs("smolmoe " SMOLMOE_VERSION " - CPU inference for SMOEQ8 checkpoints\n\n"
          "usage: smolmoe --model FILE [--config FILE --tokenizer FILE] [--prompt TEXT] [options]\n"
          "       qwen-pt.exe [--prompt TEXT] [options]     (self-contained bundle: no files needed)\n"
          "  Without a prompt, a chat model starts an interactive conversation.\n"
          "  --config/--tokenizer default to the copies embedded by smolmoe-bundle.\n\n"
          "input\n"
          "  --prompt-file FILE   raw UTF-8/byte prompt instead of --prompt\n"
          "  --chat               wrap the prompt in the model's single-turn chat template\n"
          "  --interactive, -i    multi-turn chat on stdin/stdout (default without a prompt)\n"
          "  --system TEXT        chat system message (default: Brazilian Portuguese)\n"
          "  --no-tools           chat without the built-in tools (date, days until a date)\n"
          "  --now YYYY-MM-DDTHH:MM  fixed local time for the tools (default: the clock)\n"
          "generation\n"
          "  --tokens N           max new tokens (default 64; 512 per reply in chat)\n"
          "  --temperature X      0 = greedy (default), positive = sample\n"
          "  --top-k N            sample among the N most likely tokens (default 40, 0 = all)\n"
          "  --top-p X            nucleus sampling mass in (0,1] (default 0.95)\n"
          "  --seed N             nonzero sampling seed (default 1)\n"
          "resources\n"
          "  --context N          KV cache positions (default min(config, 2048))\n"
          "  --threads N          worker threads (default: physical cores)\n"
          "  --batch N            prompt tokens per forward pass (default 32)\n"
          "  --memory-mib N       heap ceiling, excluding mapped weights (default 512 plus the KV cache of up to 2048 positions)\n"
          "  --scalar             disable SIMD kernels\n"
          "diagnostics\n"
          "  --stats FILE         JSON statistics (always printed to stderr)\n"
          "  --trace FILE         exact hexadecimal logits of every processed token\n"
          "  --hash-logits        FNV-1a64 of every raw logit, reported in the stats\n"
          "  --profile            forward-pass time split by phase, reported in the stats\n"
          "  --token-ids FILE     prompt and generated token IDs, one per line\n"
          "  --perplexity         score the prompt (teacher-forced NLL, perplexity); no generation\n"
          "  --tokenize-only      encode and render the prompt; no weights needed\n"
          "  --version, --help\n", out);
}

static void parse_value(Options *o, const char *opt, const char *value) {
    if (!strcmp(opt, "--model")) o->model = value;
    else if (!strcmp(opt, "--config")) o->config = value;
    else if (!strcmp(opt, "--tokenizer")) o->tokenizer = value;
    else if (!strcmp(opt, "--prompt")) o->prompt = value;
    else if (!strcmp(opt, "--prompt-file")) o->prompt_file = value;
    else if (!strcmp(opt, "--system")) o->system = value;
    else if (!strcmp(opt, "--stats")) o->stats = value;
    else if (!strcmp(opt, "--trace")) o->trace = value;
    else if (!strcmp(opt, "--token-ids")) o->token_ids = value;
    else if (!strcmp(opt, "--now")) {
        if (!tool_clock_parse(value, &o->clock)) die("--now must be a local date and time as YYYY-MM-DDTHH:MM");
        o->fixed_clock = 1;
    }
    else if (!strcmp(opt, "--tokens")) {
        o->tokens = parse_u32(value, MAX_CONTEXT, "tokens");
        o->tokens_given = 1;
    } else if (!strcmp(opt, "--context")) o->context = parse_u32(value, MAX_CONTEXT, "context");
    else if (!strcmp(opt, "--memory-mib")) {
        o->memory_mib = parse_u32(value, 1048576, "memory-mib");
        o->memory_given = 1;
    }
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

static void parse_args(int argc, char **argv, Options *o) {
    int a;
    memset(o, 0, sizeof(*o));
    o->system = DEFAULT_SYSTEM;
    o->tokens = DEFAULT_TOKENS;
    o->memory_mib = 512;
    o->seed = 1;
    o->batch = DEFAULT_BATCH;
    o->top_k = 40;
    o->top_p = 0.95f;
    for (a = 1; a < argc; ++a) {
        const char *opt = argv[a];
        if (!strcmp(opt, "--help") || !strcmp(opt, "-h")) { usage(stdout); exit(0); }
        if (!strcmp(opt, "--version")) { puts("smolmoe " SMOLMOE_VERSION); exit(0); }
        if (!strcmp(opt, "--scalar")) { o->scalar = 1; continue; }
        if (!strcmp(opt, "--chat")) { o->chat = 1; continue; }
        if (!strcmp(opt, "--interactive") || !strcmp(opt, "-i")) { o->interactive = 1; continue; }
        if (!strcmp(opt, "--hash-logits")) { o->hash_logits = 1; continue; }
        if (!strcmp(opt, "--tokenize-only")) { o->tokenize_only = 1; continue; }
        if (!strcmp(opt, "--perplexity")) { o->perplexity = 1; continue; }
        if (!strcmp(opt, "--no-tools")) { o->no_tools = 1; continue; }
        if (!strcmp(opt, "--profile")) { o->profile = 1; continue; }
        if (a + 1 >= argc) die("missing argument for %s", opt);
        parse_value(o, opt, argv[++a]);
    }
    if (o->prompt && o->prompt_file) { usage(stderr); exit(1); }
    if (!o->prompt && !o->prompt_file) {
        if (o->tokenize_only || o->perplexity) die("--tokenize-only and --perplexity need --prompt or --prompt-file");
        o->interactive = 1;
    }
    if (o->interactive && (o->prompt || o->prompt_file || o->trace || o->hash_logits || o->token_ids))
        die("interactive chat cannot be combined with a prompt or with --trace/--hash-logits/--token-ids");
    if (o->interactive && !o->tokens_given) o->tokens = DEFAULT_CHAT_TOKENS;
    if (!o->seed || !o->memory_mib || !o->batch) die("seed, memory budget and batch must be nonzero");
    if (o->perplexity) o->tokens = 0;
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

/* -log softmax(logits)[target], in double. */
static double token_nll(const float *logits, uint32_t vocab, uint32_t target) {
    double high = logits[0], sum = 0;
    uint32_t i;
    for (i = 1; i < vocab; ++i)
        if (logits[i] > high) high = logits[i];
    for (i = 0; i < vocab; ++i) sum += exp((double)logits[i] - high);
    return high + log(sum) - (double)logits[target];
}

/* Traces, hashes and scores rows of logits for positions pos0.. . */
static void record_logits(const Options *o, FILE *trace, Stats *st, const float *logits, uint32_t rows,
                          uint32_t pos0, uint32_t vocab, const uint32_t *ids, uint32_t np) {
    uint32_t r;
    for (r = 0; r < rows; ++r) {
        const float *row = logits + (size_t)r * vocab;
        uint32_t pos = pos0 + r;
        if (trace) trace_logits(trace, pos, row, vocab);
        if (o->hash_logits) st->logits_hash = hash_logits(st->logits_hash, row, vocab);
        if (o->perplexity && pos + 1 < np) {
            st->nll += token_nll(row, vocab, ids[pos + 1]);
            ++st->nll_tokens;
        }
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
            "\"expert_calls\":%" PRIu64 ",\"logits_hash\":\"%016" PRIx64 "\"",
            SMOLMOE_VERSION, backend, pool_threads(m->pool), m->batch, mem->limit, mem->peak, rss_peak_bytes(),
            mapped, model_kv_bytes(m), st->prompt_tokens, st->generated, st->processed,
            st->loaded - st->start, prefill, decode, st->first > 0 ? st->first - st->start : 0,
            prefill > 0 ? st->prompt_tokens / prefill : 0, decode > 0 ? decode_forwards / decode : 0,
            m->expert_calls, o->hash_logits ? st->logits_hash : 0);
    if (o->perplexity) {
        double mean = st->nll_tokens ? st->nll / st->nll_tokens : 0;
        fprintf(f, ",\"perplexity_tokens\":%u,\"nll\":%.6f,\"perplexity\":%.6f", st->nll_tokens, mean, exp(mean));
    }
    if (o->interactive) fprintf(f, ",\"tool_calls\":%u", st->tool_calls);
    if (o->profile) {
        uint32_t p;
        fputs(",\"profile\":{", f);
        for (p = 0; p < MODEL_PHASES; ++p)
            fprintf(f, "%s\"%s\":%.6f", p ? "," : "", model_phase_name((ModelPhase)p), m->phase_seconds[p]);
        fputc('}', f);
    }
    fputs("}\n", f);
    if (ferror(f)) die("statistics write failed");
}

/* ---- model source ------------------------------------------------------ */

/* --model wins; otherwise look for a container appended to this executable. */
static void open_source(const Options *o, Source *s, Memory *mem) {
    const char *self;
    char err[512];
    uint64_t offset, length;
    memset(s, 0, sizeof(*s));
    if (o->model) {
        map_file(o->model, &s->file);
        modelfile_parse(&s->mf, s->file.data, s->file.size, mem);
        s->present = 1;
        return;
    }
    self = self_executable_path();
    /* Not being able to read our own file just means "no bundle". */
    if (!self || !try_map_file(self, &s->file, err, sizeof(err))) return;
    if (!bundle_find(s->file.data, s->file.size, &offset, &length)) {
        unmap_file(&s->file);
        return;
    }
    modelfile_parse(&s->mf, s->file.data + offset, length, mem);
    s->present = 1;
}

static const Tensor *embedded(const Source *s, const char *name) {
    return s->present ? modelfile_blob(&s->mf, name) : NULL;
}

static Config load_config(const Options *o, const Source *s) {
    const Tensor *blob;
    if (o->config) return config_read(o->config);
    blob = embedded(s, BLOB_CONFIG);
    if (!blob) die("no config: pass --config, or embed it with smolmoe-bundle");
    return config_parse(blob->data, (size_t)blob->bytes);
}

static Tokenizer *load_tokenizer(const Options *o, const Source *s, uint32_t vocab, Memory *mem) {
    const Tensor *blob;
    if (o->tokenizer) return tokenizer_load(o->tokenizer, vocab, mem);
    blob = embedded(s, BLOB_TOKENIZER);
    if (!blob) die("no tokenizer: pass --tokenizer, or embed it with smolmoe-bundle");
    return tokenizer_parse(blob->data, (size_t)blob->bytes, vocab, mem);
}

/* ---- generation -------------------------------------------------------- */

typedef struct Run {
    const Options *o;
    Model *model;
    Tokenizer *tok;
    Sampler *sampler;
    uint32_t *ids;
    unsigned char *prompt, *piece;
    uint32_t prompt_cap, np;
    FILE *trace, *ids_out;
    Output *out;
    Stats *st;
    const char *system; /* chat: --system, plus the tool declarations when tools are on */
    int tools;
    uint32_t tool_open, tool_close;
    unsigned char *banned; /* the sampler's; the chat allows the tool markers per message */
} Run;

static uint32_t default_threads(void) {
    uint32_t n = cpu_physical_cores();
    return n > MAX_THREADS ? MAX_THREADS : n;
}

/* Batched prefill; every logit is kept when traced, hashed or scored. */
static uint32_t prefill_prompt(const Run *r) {
    const uint32_t vocab = r->model->c.vocab;
    const int every_logit = r->trace || r->o->hash_logits || r->o->perplexity;
    uint32_t pos = 0;
    while (pos < r->np) {
        uint32_t n = r->np - pos < r->model->batch ? r->np - pos : r->model->batch;
        int last = pos + n == r->np;
        LogitsMode mode = every_logit ? LOGITS_ALL : last ? LOGITS_LAST : LOGITS_NONE;
        model_forward(r->model, r->ids + pos, n, pos, mode);
        if (mode == LOGITS_ALL) record_logits(r->o, r->trace, r->st, r->model->logits, n, pos, vocab, r->ids, r->np);
        if (mode == LOGITS_ALL && last)
            memmove(r->model->logits, r->model->logits + (size_t)(n - 1) * vocab, vocab * sizeof(float));
        pos += n;
    }
    return pos;
}

/* Single prompt: prefill, then sample until EOS or --tokens. */
static void run_prompt(const Run *r) {
    const uint32_t vocab = r->model->c.vocab;
    const int every_logit = r->trace || r->o->hash_logits || r->o->perplexity;
    Stats *st = r->st;
    uint32_t pos, i;
    st->loaded = now_seconds();
    pos = prefill_prompt(r);
    st->processed = st->prompt_tokens = r->np;
    st->prefill_end = now_seconds();
    for (i = 0; i < r->o->tokens; ++i) {
        uint32_t token = sampler_next(r->sampler, r->model->logits), len;
        if (r->ids_out) fprintf(r->ids_out, "%u\n", token);
        if (token == EOS_ID) break;
        len = tokenizer_decode(r->tok, token, r->piece);
        output_bytes(r->out, r->piece, len);
        if (fflush(stdout)) die("output flush failed");
        ++st->generated;
        if (st->first == 0) st->first = now_seconds();
        if (i + 1 < r->o->tokens) {
            LogitsMode mode = every_logit ? LOGITS_ALL : LOGITS_LAST;
            model_forward(r->model, &token, 1, pos, mode);
            if (mode == LOGITS_ALL) record_logits(r->o, r->trace, st, r->model->logits, 1, pos, vocab, r->ids, r->np);
            ++pos;
            ++st->processed;
        }
    }
    output_finish(r->out);
    st->ended = now_seconds();
}

static void run_chat(const Run *r) {
    ChatContext cc;
    ChatStats cs;
    Stats *st = r->st;
    cc.model = r->model;
    cc.tok = r->tok;
    cc.sampler = r->sampler;
    cc.out = r->out;
    cc.ids = r->ids;
    cc.line = (char *)r->prompt;
    cc.line_cap = r->prompt_cap;
    cc.piece = r->piece;
    cc.system = r->system;
    cc.tools = r->tools;
    cc.tool_open = r->tool_open;
    cc.tool_close = r->tool_close;
    cc.fixed_clock = r->o->fixed_clock ? &r->o->clock : NULL;
    cc.banned = r->banned;
    cc.max_reply = r->o->tokens;
    st->loaded = now_seconds();
    chat_run(&cc, &cs);
    /* Stats count time spent computing, not time spent waiting for the user. */
    st->prompt_tokens = cs.prompt_tokens;
    st->generated = cs.generated;
    st->processed = cs.prompt_tokens + cs.generated;
    st->prefill_end = st->loaded + cs.prefill_seconds;
    st->ended = st->prefill_end + cs.decode_seconds;
    st->first = cs.generated ? st->loaded + cs.first_token_seconds : 0;
    st->tool_calls = cs.tool_calls;
}

/* Chat tools are on when the vocabulary has the call markers, --no-tools is
 * absent and the declarations fit in half the context. Otherwise the chat runs
 * with the plain system prompt and the markers stay banned. */
static void enable_tools(Run *r, uint32_t context, Memory *mem) {
    size_t cap = strlen(r->o->system) + TOOLS_PROMPT_BYTES;
    char *text;
    uint32_t n;
    r->system = r->o->system;
    if (r->o->no_tools || !tokenizer_tool_markers(r->tok, &r->tool_open, &r->tool_close)) return;
    text = (char *)mem_alloc(mem, cap, 1);
    if (!tools_system_prompt(r->o->system, text, cap)) die("internal error: tool declarations do not fit");
    n = tokenizer_chat_turn(r->tok, "system", (const unsigned char *)text, (uint32_t)strlen(text), 0, r->ids);
    if (n > context / 2) {
        fprintf(stderr, "(ferramentas desativadas: o contexto é pequeno; use --context %u ou mais)\n", 2 * n);
        return;
    }
    r->system = text;
    r->tools = 1;
}

int main(int argc, char **argv) {
    Options o;
    Stats st;
    Memory mem;
    Source src;
    Config c;
    Model model;
    Sampler sampler;
    Pool *pool;
    Output out;
    Run run;
    FILE *stats_out;
    unsigned char *banned;
    uint32_t id_cap, i, context;
    const char *backend;

    memset(&st, 0, sizeof(st));
    memset(&run, 0, sizeof(run));
    st.start = now_seconds();
    st.logits_hash = UINT64_C(14695981039346656037);
    platform_init(&argc, &argv);
    parse_args(argc, argv, &o);
    check_output_paths(&o);
    if (sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24) die("requires IEEE binary32 floats");

    memset(&mem, 0, sizeof(mem));
    mem.limit = mul_size(o.memory_mib, 1048576);
    open_source(&o, &src, &mem);
    if (!src.present && !o.tokenize_only) {
        usage(stderr);
        return 1;
    }
    c = load_config(&o, &src);
    context = o.context ? o.context : (c.max_context < DEFAULT_CONTEXT ? c.max_context : DEFAULT_CONTEXT);
    if (context > c.max_context) die("--context exceeds the model's max_context (%u)", c.max_context);
    /* The KV cache grows with the model and the context; the default budget
     * covers it up to the default context, so a 4B model opens without
     * --memory-mib while a huge --context still needs an explicit budget. */
    if (!o.memory_given)
        mem.limit = add_size(mem.limit, kv_cache_bytes(&c, context < DEFAULT_CONTEXT ? context : DEFAULT_CONTEXT));

    run.o = &o;
    run.st = &st;
    run.out = &out;
    run.tok = load_tokenizer(&o, &src, c.vocab, &mem);
    if ((o.chat || o.interactive) && !tokenizer_supports_chat(run.tok))
        die("chat needs a chat model (a tokenizer with a chat template); pass --prompt for plain text continuation");
    run.prompt_cap = (uint32_t)(context < MAX_CONTEXT / PROMPT_BYTES_PER_TOKEN ? context * PROMPT_BYTES_PER_TOKEN
                                                                               : MAX_CONTEXT);
    {
        /* The chat's system message (with tool declarations) can exceed a small context's prompt cap. */
        size_t system_bytes = o.interactive ? strlen(o.system) + TOOLS_PROMPT_BYTES : 0;
        uint32_t encode_cap = system_bytes > run.prompt_cap ? (uint32_t)system_bytes : run.prompt_cap;
        if (system_bytes > MAX_CONTEXT * PROMPT_BYTES_PER_TOKEN) die("--system is too long");
        id_cap = tokenizer_prepare(run.tok, encode_cap, &mem);
    }
    run.prompt = (unsigned char *)mem_alloc(&mem, (size_t)run.prompt_cap + 1, 1);
    run.ids = (uint32_t *)mem_alloc(&mem, id_cap, sizeof(uint32_t));
    run.piece = (unsigned char *)mem_alloc(&mem, tokenizer_max_piece(run.tok), 1);
    out = output_to(stdout);
    if (!o.interactive) {
        run.np = tokenizer_encode(run.tok, run.prompt, read_prompt(&o, run.prompt, run.prompt_cap), run.ids, o.chat,
                                  o.system);
        if (run.np > context) die("encoded prompt (%u tokens) exceeds --context (%u)", run.np, context);
    }

    run.ids_out = open_output(o.token_ids);
    if (run.ids_out)
        for (i = 0; i < run.np; ++i) fprintf(run.ids_out, "%u\n", run.ids[i]);
    if (o.tokenize_only) {
        tokenizer_render_prompt(run.tok, run.ids, run.np, run.piece, output_bytes, &out);
        output_finish(&out);
        close_output(run.ids_out, "token ID output");
        if (src.present) unmap_file(&src.file);
        mem_release(&mem);
        return 0;
    }
    if (!o.interactive && o.tokens > context - run.np)
        die("prompt + --tokens exceeds --context; increase context or reduce tokens");

    backend = kernels_select(o.scalar);
    pool = pool_create(o.threads ? o.threads : default_threads(), &mem);
    modelfile_verify_payload(&src.mf, pool);
    {
        uint32_t batch = o.batch < context ? o.batch : context;
        int every_logit = o.trace || o.hash_logits || o.perplexity;
        model_init(&model, &c, &src.mf, context, batch, every_logit ? batch : 1, pool, &mem);
        model.profiling = o.profile;
    }
    banned = (unsigned char *)mem_alloc(&mem, c.vocab, 1);
    for (i = 0; i < c.vocab; ++i) banned[i] = (unsigned char)tokenizer_is_control(run.tok, i);
    run.banned = banned;
    if (o.interactive) enable_tools(&run, context, &mem);
    sampler_init(&sampler, c.vocab, banned, o.temperature, o.top_k, o.top_p, o.seed, &mem);
    run.model = &model;
    run.sampler = &sampler;
    run.trace = open_output(o.trace);

    if (o.interactive) run_chat(&run);
    else run_prompt(&run);

    close_output(run.trace, "trace");
    close_output(run.ids_out, "token ID output");
    print_stats(stderr, backend, &o, &model, &mem, src.mf.size, &st);
    stats_out = open_output(o.stats);
    if (stats_out) {
        print_stats(stats_out, backend, &o, &model, &mem, src.mf.size, &st);
        close_output(stats_out, "stats");
    }
    pool_destroy(pool);
    unmap_file(&src.file);
    mem_release(&mem);
    return 0;
}
