#include "config.h"
#include "platform.h"
#include "util.h"

#include <math.h>
#include <string.h>

#define REQUIRED_KEYS 15u
#define KEY_COUNT 20u /* the required ones, then optional RoPE, head_dim, and sliding_window keys */
#define ALL_REQUIRED ((1u << REQUIRED_KEYS) - 1u)
#define KEY_ROTARY_DIM 15u
#define KEY_ROPE_FACTOR 16u
#define KEY_HEAD_DIM 17u
#define KEY_SLIDING_WINDOW 18u
#define KEY_LOCAL_THETA 19u
#define MAX_ROPE_FACTOR 1000.0
#define LINE_CAPACITY 512u

/* Optional RoPE and attention keys: defaults keep the full-head, unscaled rotation. */
static void rope_defaults(Config *c, unsigned seen) {
    uint32_t head_dim = c->head_dim;
    if (!(seen & (1u << KEY_ROTARY_DIM))) c->rotary_dim = head_dim;
    if (!(seen & (1u << KEY_ROPE_FACTOR))) c->rope_factor = 1.0;
    if (!(seen & (1u << KEY_LOCAL_THETA))) c->local_theta = c->theta;
    if (!c->rotary_dim || c->rotary_dim % 2 || c->rotary_dim > head_dim)
        sm_fail(SM_ERR_FORMAT, "invalid rotary_dim %u: must be even and at most the head size %u", c->rotary_dim, head_dim);
    if (!(c->rope_factor > 0 && c->rope_factor <= MAX_ROPE_FACTOR))
        sm_fail(SM_ERR_FORMAT, "invalid rope_attention_factor: must be in (0, %g]", MAX_ROPE_FACTOR);
    if ((seen & (1u << KEY_SLIDING_WINDOW)) && (!c->sliding_window || c->sliding_window > c->max_context))
        sm_fail(SM_ERR_FORMAT, "invalid sliding_window %u: must be in 1..%u", c->sliding_window, c->max_context);
    if (!(c->local_theta > 1) || !isfinite(c->local_theta))
        sm_fail(SM_ERR_FORMAT, "invalid rope_local_theta: must be finite and > 1");
}

static void config_parse_body(const unsigned char *text, size_t size, Config *out) {
    static const char *keys[KEY_COUNT] = {
        "architecture", "dim", "hidden_dim", "layers", "heads", "kv_heads", "experts", "top_k",
        "vocab_size", "max_context", "shared_experts", "bos_id", "eos_id", "rms_eps", "rope_theta",
        "rotary_dim", "rope_attention_factor", "head_dim", "sliding_window", "rope_local_theta"};
    Config c;
    uint32_t *fields[12];
    char line[LINE_CAPACITY];
    unsigned seen = 0;
    ByteReader r = byte_reader(text, size);
    memset(&c, 0, sizeof(c));
    fields[0] = &c.dim; fields[1] = &c.hidden; fields[2] = &c.layers; fields[3] = &c.heads;
    fields[4] = &c.kv_heads; fields[5] = &c.experts; fields[6] = &c.top_k; fields[7] = &c.vocab;
    fields[8] = &c.max_context; fields[9] = &c.shared; fields[10] = &c.bos; fields[11] = &c.eos;
    while (br_line(&r, line, sizeof(line), "config")) {
        char *s, *eq, *value, *comment;
        unsigned k;
        comment = strchr(line, '#');
        if (comment) *comment = 0;
        s = trim(line);
        if (!*s) continue;
        eq = strchr(s, '=');
        if (!eq) sm_fail(SM_ERR_FORMAT, "config requires key=value lines");
        *eq = 0;
        s = trim(s);
        value = trim(eq + 1);
        for (k = 0; k < KEY_COUNT && strcmp(s, keys[k]); ++k) {}
        if (k == KEY_COUNT) sm_fail(SM_ERR_FORMAT, "unknown config key: %s", s);
        if (seen & (1u << k)) sm_fail(SM_ERR_FORMAT, "duplicate config key: %s", s);
        seen |= 1u << k;
        if (k == 0) {
            if (!strcmp(value, "smolmoe_qwen2")) c.qkv_bias = 1;
            else if (!strcmp(value, "smolmoe_gemma3")) c.gemma3 = 1;
            else if (strcmp(value, "smolmoe_v1")) sm_fail(SM_ERR_FORMAT, "unsupported architecture: %s", value);
        } else if (k < 13) {
            *fields[k - 1] = parse_u32(value, MAX_CONTEXT, s);
        } else if (k == 13) {
            c.eps = (float)parse_real(value, s);
        } else if (k == 14) {
            c.theta = (float)parse_real(value, s);
        } else if (k == KEY_ROTARY_DIM) {
            c.rotary_dim = parse_u32(value, MAX_DIM, s);
        } else if (k == KEY_ROPE_FACTOR) {
            c.rope_factor = parse_real(value, s);
        } else if (k == KEY_HEAD_DIM) {
            c.head_dim = parse_u32(value, MAX_DIM, s);
        } else if (k == KEY_SLIDING_WINDOW) {
            c.sliding_window = parse_u32(value, MAX_CONTEXT, s);
        } else {
            c.local_theta = (float)parse_real(value, s);
        }
    }
    if ((seen & ALL_REQUIRED) != ALL_REQUIRED) sm_fail(SM_ERR_FORMAT, "config is missing required fields");
    if (!(seen & (1u << KEY_HEAD_DIM))) {
        if (!c.heads || c.heads > c.dim || c.dim % c.heads)
            sm_fail(SM_ERR_FORMAT, "invalid config dimensions or special token IDs");
        c.head_dim = c.dim / c.heads;
    }
    if (!c.dim || c.dim > MAX_DIM || !c.hidden || c.hidden > MAX_DIM || !c.layers || c.layers > 1024 ||
        !c.heads || c.heads > MAX_DIM || !c.head_dim || c.head_dim > MAX_DIM || c.head_dim % 2 ||
        (uint64_t)c.heads * c.head_dim > MAX_DIM ||
        !c.kv_heads || c.heads % c.kv_heads || !c.experts || c.experts > 4096 ||
        !c.top_k || c.top_k > c.experts || c.shared > 16 || c.vocab < 258 || c.vocab > 258 + MAX_MERGES ||
        !c.max_context || c.max_context > MAX_CONTEXT || c.bos != BOS_ID || c.eos != EOS_ID ||
        !(c.eps > 0 && c.eps < 1) || !(c.theta > 1) || !isfinite(c.theta))
        sm_fail(SM_ERR_FORMAT, "invalid config dimensions or special token IDs");
    rope_defaults(&c, seen);
    *out = c;
}

typedef struct ConfigArgs {
    const unsigned char *text;
    size_t size;
    Config *out;
} ConfigArgs;

static void config_parse_protected(void *ctx) {
    ConfigArgs *a = (ConfigArgs *)ctx;
    config_parse_body(a->text, a->size, a->out);
}

sm_status config_parse(const unsigned char *text, size_t size, Config *out) {
    ConfigArgs a;
    a.text = text;
    a.size = size;
    a.out = out;
    return sm_protect(config_parse_protected, &a);
}

sm_status config_read(const char *path, Config *out) {
    MappedFile f;
    char err[512];
    sm_status st;
    if (!try_map_file(path, &f, err, sizeof(err))) return sm_set(SM_ERR_IO, "%s", err);
    st = config_parse(f.data, (size_t)f.size, out);
    unmap_file(&f);
    return st;
}
