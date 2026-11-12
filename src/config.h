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
    uint32_t max_context, shared, bos, eos, qkv_bias, gemma3;
    uint32_t head_dim;       /* per-head dimension (default: dim / heads) */
    uint32_t rotary_dim;     /* leading coordinates of each head that RoPE rotates (default: head_dim) */
    uint32_t sliding_window; /* local attention window for non-global layers (default: 0 = full) */
    float eps, theta, local_theta;
    double rope_factor;      /* rope_attention_factor: scales RoPE cos/sin (default 1) */
    /* Llama 3 RoPE scaling (rope_type "llama3"), all four keys or none: low frequencies
     * (wavelength > original_context / low_freq) are divided by factor, high ones
     * (wavelength < original_context / high_freq) kept, and the band between blended. */
    uint32_t llama3_rope;           /* 1 when the four llama3 keys are present */
    uint32_t llama3_original;       /* rope_llama3_original_context */
    double llama3_factor, llama3_low, llama3_high; /* rope_llama3_factor, _low_freq_factor, _high_freq_factor */
} Config;

#include <stddef.h>

/* Parses config text (from a file or an embedded `__config__` blob). */
Config config_parse(const unsigned char *text, size_t size);
Config config_read(const char *path);

#endif
