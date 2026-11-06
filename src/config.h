/* Architecture description, read from a strict key=value file. */
#ifndef SMOLMOE_CONFIG_H
#define SMOLMOE_CONFIG_H

#include <stdint.h>

#define MAX_CONTEXT 1048576u
#define MAX_DIM 65536u
#define MAX_MERGES 262144u
#define BOS_ID 256u
#define EOS_ID 257u

typedef struct Config {
    uint32_t dim, hidden, layers, heads, kv_heads, experts, top_k, vocab;
    uint32_t max_context, shared, bos, eos, qkv_bias;
    uint32_t rotary_dim; /* leading coordinates of each head that RoPE rotates (default: all) */
    float eps, theta;
    double rope_factor; /* rope_attention_factor: scales RoPE cos/sin (default 1) */
} Config;

#include <stddef.h>

/* Parses config text (from a file or an embedded `__config__` blob). */
Config config_parse(const unsigned char *text, size_t size);
Config config_read(const char *path);

#endif
