/* Decoder forward pass: GQA attention, RoPE, RMSNorm, SwiGLU experts with
 * sigmoid top-k routing (docs/FORMAT.md). One code path serves batched
 * prefill and single-token decode; outputs are identical either way. */
#ifndef SMOLMOE_MODEL_H
#define SMOLMOE_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "config.h"
#include "modelfile.h"
#include "threadpool.h"
#include "util.h"

typedef struct QMat {
    const unsigned char *data;
    uint32_t rows, cols, nb, dtype; /* DTYPE_Q8 or DTYPE_Q4 */
    size_t row_bytes;
} QMat;

typedef struct Expert {
    QMat gate, up, down;
} Expert;

typedef struct Layer {
    const float *attn_norm, *ffn_norm, *post_attn_norm, *post_ffn_norm;
    const float *q_bias, *k_bias, *v_bias, *q_norm, *k_norm;
    QMat q, k, v, o, router;
    Expert *experts, *shared;
} Layer;

typedef enum { LOGITS_NONE, LOGITS_LAST, LOGITS_ALL } LogitsMode;

/* Wall time of the forward pass split by phase (--profile). "norm" includes
 * quantizing its output, "activation" is SwiGLU plus quantization, and
 * "attn_quant" quantizes the attention output; "logits" includes the final
 * norm. */
typedef enum ModelPhase {
    PHASE_EMBED, PHASE_NORM, PHASE_QKV, PHASE_ROPE, PHASE_ATTENTION, PHASE_ATTN_OUT, PHASE_FFN_UP,
    PHASE_ACTIVATION, PHASE_ATTN_QUANT, PHASE_FFN_DOWN, PHASE_ROUTING, PHASE_RESIDUAL, PHASE_LOGITS, MODEL_PHASES
} ModelPhase;

typedef struct Model {
    Config c;
    uint32_t context, batch, head_dim, q_dim, kv_dim, logit_rows, nb_dim, nb_q, nb_hidden;
    QMat embed, head;
    const float *final_norm;
    Layer *layers;
    Pool *pool;
    /* Activations, one row per token in the batch. */
    float *x, *xn, *q, *k, *v, *att, *out, *gate, *up, *logits;
    float *xd, *hd, *gxd;
    int8_t *xq, *hq, *gxq;
    /* Routing state (experts > 1 only). */
    float *route, *weights, *expert_out;
    uint32_t *selected, *members;
    double *inv_freq, *inv_freq_local;
    float *rope_cos, *rope_sin, *rope_cos_local, *rope_sin_local;
    float *keys, *values, *scores;
    uint64_t expert_calls;
    int profiling;
    double phase_seconds[MODEL_PHASES];
    /* Per-thread scratch for the prefill repack (gemm_scratch_bytes). */
    unsigned char *pack;
    size_t pack_stride;
} Model;

const char *model_phase_name(ModelPhase p);

/* logit_rows: 1 for normal generation, or the batch size when every prompt
 * token's logits are needed (--trace, --hash-logits). */
void model_init(Model *m, const Config *c, const ModelFile *f, uint32_t context, uint32_t batch,
                uint32_t logit_rows, Pool *pool, Memory *mem);

/* Runs tokens[0..n) at positions pos0..pos0+n-1. LOGITS_LAST writes the last
 * token's logits to m->logits; LOGITS_ALL writes n rows of vocab_size. */
void model_forward(Model *m, const uint32_t *tokens, uint32_t n, uint32_t pos0, LogitsMode mode);

/* Runs n tokens in batch-sized chunks; with last_logits, m->logits holds the
 * last token's logits afterwards. */
void model_prefill(Model *m, const uint32_t *tokens, uint32_t n, uint32_t pos0, int last_logits);

size_t model_kv_bytes(const Model *m);

/* Bytes of the F32 keys and values for `context` positions. */
size_t kv_cache_bytes(const Config *c, uint32_t context);

#endif
