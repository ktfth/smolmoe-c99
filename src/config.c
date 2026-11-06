#include "config.h"
#include "platform.h"
#include "util.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#define KEY_COUNT 15u
#define ALL_KEYS ((1u << KEY_COUNT) - 1u)

Config config_read(const char *path) {
    static const char *keys[KEY_COUNT] = {
        "architecture", "dim", "hidden_dim", "layers", "heads", "kv_heads", "experts", "top_k",
        "vocab_size", "max_context", "shared_experts", "bos_id", "eos_id", "rms_eps", "rope_theta"};
    Config c;
    uint32_t *fields[12];
    char line[512];
    unsigned seen = 0;
    FILE *fp;
    memset(&c, 0, sizeof(c));
    fields[0] = &c.dim; fields[1] = &c.hidden; fields[2] = &c.layers; fields[3] = &c.heads;
    fields[4] = &c.kv_heads; fields[5] = &c.experts; fields[6] = &c.top_k; fields[7] = &c.vocab;
    fields[8] = &c.max_context; fields[9] = &c.shared; fields[10] = &c.bos; fields[11] = &c.eos;
    fp = open_file(path, "r");
    if (!fp) die("cannot open config %s: %s", path, strerror(errno));
    while (fgets(line, sizeof(line), fp)) {
        char *s, *eq, *value, *comment;
        unsigned k;
        if (!strchr(line, '\n') && !feof(fp)) die("config line too long");
        comment = strchr(line, '#');
        if (comment) *comment = 0;
        s = trim(line);
        if (!*s) continue;
        eq = strchr(s, '=');
        if (!eq) die("config requires key=value lines");
        *eq = 0;
        s = trim(s);
        value = trim(eq + 1);
        for (k = 0; k < KEY_COUNT && strcmp(s, keys[k]); ++k) {}
        if (k == KEY_COUNT) die("unknown config key: %s", s);
        if (seen & (1u << k)) die("duplicate config key: %s", s);
        seen |= 1u << k;
        if (k == 0) {
            if (!strcmp(value, "smolmoe_qwen2")) c.qkv_bias = 1;
            else if (strcmp(value, "smolmoe_v1")) die("unsupported architecture: %s", value);
        } else if (k < 13) {
            *fields[k - 1] = parse_u32(value, MAX_CONTEXT, s);
        } else if (k == 13) {
            c.eps = (float)parse_real(value, s);
        } else {
            c.theta = (float)parse_real(value, s);
        }
    }
    if (ferror(fp)) die("config read error");
    fclose(fp);
    if (seen != ALL_KEYS) die("config is missing required fields");
    if (!c.dim || c.dim > MAX_DIM || !c.hidden || c.hidden > MAX_DIM || !c.layers || c.layers > 1024 ||
        !c.heads || c.heads > c.dim || c.dim % c.heads || (c.dim / c.heads) % 2 ||
        !c.kv_heads || c.heads % c.kv_heads || !c.experts || c.experts > 4096 ||
        !c.top_k || c.top_k > c.experts || c.shared > 16 || c.vocab < 258 || c.vocab > 258 + MAX_MERGES ||
        !c.max_context || c.max_context > MAX_CONTEXT || c.bos != BOS_ID || c.eos != EOS_ID ||
        !(c.eps > 0 && c.eps < 1) || !(c.theta > 1) || !isfinite(c.theta))
        die("invalid config dimensions or special token IDs");
    return c;
}
