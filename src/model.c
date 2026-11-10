#include "model.h"
#include "compat.h"
#include "kernels.h"
#include "platform.h"
#include "vecmath.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- weights ----------------------------------------------------------- */

static QMat qmat(const ModelFile *f, const char *name, uint32_t rows, uint32_t cols) {
    const Tensor *t = modelfile_find(f, name, rows, cols, DTYPE_QUANTIZED);
    QMat q;
    q.data = t->data;
    q.rows = rows;
    q.cols = cols;
    q.nb = (cols + QK - 1) / QK;
    q.dtype = t->dtype;
    q.row_bytes = (size_t)q.nb * (t->dtype == DTYPE_Q8 ? Q8_BLOCK_BYTES : Q5_BLOCK_BYTES);
    return q;
}

/* F32 vectors are tiny (norms, biases): copy them to aligned heap memory. */
static const float *fvec(const ModelFile *f, const char *name, uint32_t n, Memory *mem) {
    const Tensor *t = modelfile_find(f, name, 1, n, DTYPE_F32);
    float *v = (float *)mem_alloc(mem, n, sizeof(float));
    uint32_t i;
    for (i = 0; i < n; ++i) v[i] = rd_f32(t->data + (size_t)i * 4);
    return v;
}

static const char *layer_name(char *buf, uint32_t layer, const char *suffix) {
    int n = snprintf(buf, TENSOR_NAME_SIZE, "l.%u.%s", layer, suffix);
    if (n < 0 || n >= (int)TENSOR_NAME_SIZE) die("tensor name too long");
    return buf;
}

static Expert expert_weights(const ModelFile *f, const Config *c, uint32_t layer, const char *kind, uint32_t e) {
    char suffix[32], name[TENSOR_NAME_SIZE];
    Expert x;
    snprintf(suffix, sizeof(suffix), "%s.%u.gate", kind, e);
    x.gate = qmat(f, layer_name(name, layer, suffix), c->hidden, c->dim);
    snprintf(suffix, sizeof(suffix), "%s.%u.up", kind, e);
    x.up = qmat(f, layer_name(name, layer, suffix), c->hidden, c->dim);
    snprintf(suffix, sizeof(suffix), "%s.%u.down", kind, e);
    x.down = qmat(f, layer_name(name, layer, suffix), c->dim, c->hidden);
    return x;
}

static void load_layer(Layer *l, const ModelFile *f, const Config *c, uint32_t i, uint32_t q_dim, uint32_t kv_dim, Memory *mem) {
    char name[TENSOR_NAME_SIZE];
    uint32_t e;
    l->attn_norm = fvec(f, layer_name(name, i, "attn_norm"), c->dim, mem);
    l->ffn_norm = fvec(f, layer_name(name, i, "ffn_norm"), c->dim, mem);
    l->q = qmat(f, layer_name(name, i, "q"), q_dim, c->dim);
    l->k = qmat(f, layer_name(name, i, "k"), kv_dim, c->dim);
    l->v = qmat(f, layer_name(name, i, "v"), kv_dim, c->dim);
    l->o = qmat(f, layer_name(name, i, "o"), c->dim, q_dim);
    l->router = qmat(f, layer_name(name, i, "router"), c->experts, c->dim);
    if (c->qkv_bias) {
        l->q_bias = fvec(f, layer_name(name, i, "q_bias"), q_dim, mem);
        l->k_bias = fvec(f, layer_name(name, i, "k_bias"), kv_dim, mem);
        l->v_bias = fvec(f, layer_name(name, i, "v_bias"), kv_dim, mem);
    }
    if (c->gemma3) {
        l->post_attn_norm = fvec(f, layer_name(name, i, "post_attn_norm"), c->dim, mem);
        l->post_ffn_norm = fvec(f, layer_name(name, i, "post_ffn_norm"), c->dim, mem);
        l->q_norm = fvec(f, layer_name(name, i, "q_norm"), c->head_dim, mem);
        l->k_norm = fvec(f, layer_name(name, i, "k_norm"), c->head_dim, mem);
    }
    l->experts = (Expert *)mem_alloc(mem, c->experts, sizeof(Expert));
    l->shared = (Expert *)mem_alloc(mem, c->shared ? c->shared : 1, sizeof(Expert));
    for (e = 0; e < c->experts; ++e) l->experts[e] = expert_weights(f, c, i, "e", e);
    for (e = 0; e < c->shared; ++e) l->shared[e] = expert_weights(f, c, i, "s", e);
}

static void *buffer(Memory *mem, size_t rows, size_t cols, size_t size) {
    return mem_alloc(mem, mul_size(rows, cols), size);
}

void model_init(Model *m, const Config *c, const ModelFile *f, uint32_t context, uint32_t batch,
                uint32_t logit_rows, Pool *pool, Memory *mem) {
    uint64_t expected;
    uint32_t i, half, nb_max, threads = pool_threads(pool);
    memset(m, 0, sizeof(*m));
    m->c = *c;
    m->context = context;
    m->batch = batch;
    m->logit_rows = logit_rows;
    m->pool = pool;
    m->head_dim = c->head_dim;
    m->q_dim = c->heads * m->head_dim;
    m->kv_dim = c->kv_heads * m->head_dim;
    m->nb_dim = (c->dim + QK - 1) / QK;
    m->nb_q = (m->q_dim + QK - 1) / QK;
    m->nb_hidden = (c->hidden + QK - 1) / QK;
    nb_max = m->nb_dim > m->nb_q ? m->nb_dim : m->nb_q;
    half = c->rotary_dim / 2;
    expected = 3 + (uint64_t)c->layers * (7 + 3 * c->qkv_bias + 4 * c->gemma3 + 3 * (uint64_t)(c->experts + c->shared));
    if (expected != f->count - f->blobs) die("tensor count does not match config");

    m->embed = qmat(f, "embed", c->vocab, c->dim);
    m->head = qmat(f, "lm_head", c->vocab, c->dim);
    m->final_norm = fvec(f, "final_norm", c->dim, mem);
    m->layers = (Layer *)mem_alloc(mem, c->layers, sizeof(Layer));
    for (i = 0; i < c->layers; ++i) load_layer(&m->layers[i], f, c, i, m->q_dim, m->kv_dim, mem);

    m->x = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->xn = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->q = (float *)buffer(mem, batch, m->q_dim, sizeof(float));
    m->att = (float *)buffer(mem, batch, m->q_dim, sizeof(float));
    m->out = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->k = (float *)buffer(mem, batch, m->kv_dim, sizeof(float));
    m->v = (float *)buffer(mem, batch, m->kv_dim, sizeof(float));
    m->gate = (float *)buffer(mem, batch, c->hidden, sizeof(float));
    m->up = (float *)buffer(mem, batch, c->hidden, sizeof(float));
    m->xd = (float *)buffer(mem, batch, nb_max, sizeof(float));
    m->xq = (int8_t *)buffer(mem, batch, (size_t)nb_max * QK, 1);
    m->hd = (float *)buffer(mem, batch, m->nb_hidden, sizeof(float));
    m->hq = (int8_t *)buffer(mem, batch, (size_t)m->nb_hidden * QK, 1);
    if (c->experts > 1) {
        m->gxd = (float *)buffer(mem, batch, m->nb_dim, sizeof(float));
        m->gxq = (int8_t *)buffer(mem, batch, (size_t)m->nb_dim * QK, 1);
        m->route = (float *)buffer(mem, batch, c->experts, sizeof(float));
        m->weights = (float *)buffer(mem, batch, c->top_k, sizeof(float));
        m->selected = (uint32_t *)buffer(mem, batch, c->top_k, sizeof(uint32_t));
        m->members = (uint32_t *)buffer(mem, batch, c->top_k, sizeof(uint32_t));
        m->expert_out = (float *)buffer(mem, mul_size(batch, c->top_k), c->dim, sizeof(float));
    }
    m->logits = (float *)buffer(mem, logit_rows, c->vocab, sizeof(float));
    m->inv_freq = (double *)mem_alloc(mem, half, sizeof(double));
    m->inv_freq_local = (double *)mem_alloc(mem, half, sizeof(double));
    for (i = 0; i < half; ++i) {
        m->inv_freq[i] = pow((double)c->theta, -(double)(2 * i) / (double)c->rotary_dim);
        m->inv_freq_local[i] = pow((double)c->local_theta, -(double)(2 * i) / (double)c->rotary_dim);
    }
    m->rope_cos = (float *)buffer(mem, batch, half, sizeof(float));
    m->rope_sin = (float *)buffer(mem, batch, half, sizeof(float));
    m->rope_cos_local = (float *)buffer(mem, batch, half, sizeof(float));
    m->rope_sin_local = (float *)buffer(mem, batch, half, sizeof(float));
    m->keys = (float *)buffer(mem, mul_size(c->layers, context), m->kv_dim, sizeof(float));
    m->values = (float *)buffer(mem, mul_size(c->layers, context), m->kv_dim, sizeof(float));
    m->scores = (float *)buffer(mem, threads, context, sizeof(float));
}

size_t kv_cache_bytes(const Config *c, uint32_t context) {
    size_t kv_dim = (size_t)c->kv_heads * c->head_dim;
    return mul_size(mul_size(mul_size(c->layers, context), kv_dim), 2 * sizeof(float));
}

size_t model_kv_bytes(const Model *m) { return kv_cache_bytes(&m->c, m->context); }

/* ---- parallel matrix-vector products ----------------------------------- */

typedef struct MatItem {
    const QMat *w;
    const float *bias;
    float *y; /* y[token * w->rows + row] */
} MatItem;

typedef struct MatTask {
    MatItem item[3];
    uint32_t items, total_rows, ntok, nb;
    const float *xd;
    const int8_t *xq;
    uint32_t fault;
} MatTask;

static int store(const MatItem *it, uint32_t token, uint32_t row, float y) {
    if (it->bias) y += it->bias[row];
    it->y[(size_t)token * it->w->rows + row] = y;
    return isfinite(y);
}

static int mat_rows(const MatTask *t, const MatItem *it, uint32_t r0, uint32_t r1) {
    const QMat *w = it->w;
    const DotQ8 dot = w->dtype == DTYPE_Q5 ? dot_q5 : dot_q8;
    const DotQ8x4 dot_x4 = w->dtype == DTYPE_Q5 ? dot_q5_x4 : dot_q8_x4;
    size_t xs = (size_t)t->nb * QK;
    uint32_t r, k, j;
    int ok = 1;
    for (r = r0; r < r1; ++r) {
        const unsigned char *row = w->data + w->row_bytes * r;
        for (k = 0; k + 4 <= t->ntok; k += 4) {
            float y[4];
            dot_x4(row, t->xd + (size_t)k * t->nb, t->xq + k * xs, t->nb, t->nb, y);
            for (j = 0; j < 4; ++j) ok &= store(it, k + j, r, y[j]);
        }
        for (; k < t->ntok; ++k)
            ok &= store(it, k, r, dot(row, t->xd + (size_t)k * t->nb, t->xq + k * xs, t->nb));
    }
    return ok;
}

static void mat_task(void *ctx, uint32_t index, uint32_t count) {
    MatTask *t = (MatTask *)ctx;
    uint32_t begin, end, base = 0, i;
    split_range(t->total_rows, index, count, &begin, &end);
    for (i = 0; i < t->items; base += t->item[i].w->rows, ++i) {
        uint32_t rows = t->item[i].w->rows;
        uint32_t lo = begin > base ? begin : base, hi = end < base + rows ? end : base + rows;
        if (lo < hi && !mat_rows(t, &t->item[i], lo - base, hi - base))
            sm_store_u32(&t->fault, 1, SM_ATOMIC_RELAXED);
    }
}

static MatItem item(const QMat *w, const float *bias, float *y) {
    MatItem it;
    it.w = w;
    it.bias = bias;
    it.y = y;
    return it;
}

/* Computes up to three matrices that share one quantized input, splitting
 * their concatenated rows across the pool. */
static void matmul(Model *m, const MatItem *items, uint32_t n, const float *xd, const int8_t *xq, uint32_t ntok) {
    MatTask t;
    uint32_t i;
    memset(&t, 0, sizeof(t));
    t.items = n;
    t.ntok = ntok;
    t.nb = items[0].w->nb;
    t.xd = xd;
    t.xq = xq;
    for (i = 0; i < n; ++i) {
        if (items[i].w->nb != t.nb) die("internal error: matmul inputs differ in width");
        t.item[i] = items[i];
        t.total_rows += items[i].w->rows;
    }
    pool_run(m->pool, mat_task, &t);
    if (t.fault) die("non-finite activation; the checkpoint is outside the supported numeric range");
}

/* ---- profiling (--profile) --------------------------------------------- */

static const char *const PHASE_NAMES[MODEL_PHASES] = {"embed", "norm", "qkv", "rope", "attention", "attn_out",
    "ffn_up", "activation", "attn_quant", "ffn_down", "routing", "residual", "logits"};

const char *model_phase_name(ModelPhase p) { return (unsigned)p < MODEL_PHASES ? PHASE_NAMES[p] : "?"; }

static double clock_start(const Model *m) { return m->profiling ? now_seconds() : 0; }

/* Charges the time since *t to phase p and restarts the clock. */
static void lap(Model *m, ModelPhase p, double *t) {
    double now;
    if (!m->profiling) return;
    now = now_seconds();
    m->phase_seconds[p] += now - *t;
    *t = now;
}

/* ---- elementwise pieces ------------------------------------------------ */

/* Returns 0, or -1 for a non-finite input; a non-finite output is caught by
 * the quantization that always follows. */
static int rmsnorm(const float *x, const float *w, float *y, uint32_t n, float eps) {
    float sq = 0, inv;
    uint32_t i;
    for (i = 0; i < n; ++i) sq += x[i] * x[i];
    inv = 1.0f / sqrtf(sq / (float)n + eps);
    if (!isfinite(sq) || !isfinite(inv)) return -1;
    for (i = 0; i < n; ++i) y[i] = x[i] * inv * w[i];
    return 0;
}

/* Per-token work that ends in a quantized row: optional RMSNorm (src -> rows)
 * or SwiGLU/GeGLU (rows *= act(gate)), then Q8. One token's row never spans two
 * threads, so the split cannot change any value. */
typedef struct RowTask {
    const float *src, *norm, *gate;
    float *rows, *d, eps;
    int8_t *q;
    uint32_t n, cols, nb, gemma3, fault;
} RowTask;

static int row_work(const RowTask *t, uint32_t r) {
    float *row = t->rows + (size_t)r * t->cols;
    if (t->norm && rmsnorm(t->src + (size_t)r * t->cols, t->norm, row, t->cols, t->eps)) return 0;
    if (t->gate) {
        if (t->gemma3) gelu_mul_scalar(row, t->gate + (size_t)r * t->cols, t->cols);
        else silu_mul(row, t->gate + (size_t)r * t->cols, t->cols);
    }
    return quantize_q8(row, t->cols, t->d + (size_t)r * t->nb, t->q + (size_t)r * t->nb * QK) == 0;
}

static void row_task(void *ctx, uint32_t index, uint32_t count) {
    RowTask *t = (RowTask *)ctx;
    uint32_t begin, end, r;
    split_range(t->n, index, count, &begin, &end);
    for (r = begin; r < end; ++r)
        if (!row_work(t, r)) sm_store_u32(&t->fault, 1, SM_ATOMIC_RELAXED);
}

static RowTask rows_of(float *rows, uint32_t n, uint32_t cols, uint32_t nb, float *d, int8_t *q) {
    RowTask t;
    memset(&t, 0, sizeof(t));
    t.rows = rows;
    t.n = n;
    t.cols = cols;
    t.nb = nb;
    t.d = d;
    t.q = q;
    return t;
}

/* A single token (decode) runs inline: waking the pool would cost more. */
static void run_rows(Model *m, RowTask *t) {
    if (t->n == 1) row_task(t, 0, 1);
    else pool_run(m->pool, row_task, t);
    if (t->fault) die("non-finite activation; the checkpoint is outside the supported numeric range");
}

/* RMSNorm of n rows of src into m->xn, quantized into m->xd/m->xq. */
static void norm_quantize(Model *m, const float *src, const float *weight, uint32_t n) {
    RowTask t = rows_of(m->xn, n, m->c.dim, m->nb_dim, m->xd, m->xq);
    t.src = src;
    t.norm = weight;
    t.eps = m->c.eps;
    run_rows(m, &t);
}

static void quantize_attention(Model *m, uint32_t n) {
    RowTask t = rows_of(m->att, n, m->q_dim, m->nb_q, m->xd, m->xq);
    run_rows(m, &t);
}

/* m->up *= act(m->gate), quantized into m->hd/m->hq. */
static void silu_quantize(Model *m, uint32_t n) {
    RowTask t = rows_of(m->up, n, m->c.hidden, m->nb_hidden, m->hd, m->hq);
    t.gate = m->gate;
    t.gemma3 = m->c.gemma3;
    run_rows(m, &t);
}

static void norm_rows_inplace(float *rows, const float *weight, uint32_t n, uint32_t cols, float eps) {
    uint32_t r;
    for (r = 0; r < n; ++r)
        if (rmsnorm(rows + (size_t)r * cols, weight, rows + (size_t)r * cols, cols, eps))
            die("non-finite activation; the checkpoint is outside the supported numeric range");
}

static void rope_tables(Model *m, uint32_t n, uint32_t pos0) {
    uint32_t t, i, half = m->c.rotary_dim / 2;
    for (t = 0; t < n; ++t)
        for (i = 0; i < half; ++i) {
            double angle = (double)(pos0 + t) * m->inv_freq[i];
            double angle_local = (double)(pos0 + t) * m->inv_freq_local[i];
            m->rope_cos[t * half + i] = (float)(m->c.rope_factor * cos(angle));
            m->rope_sin[t * half + i] = (float)(m->c.rope_factor * sin(angle));
            m->rope_cos_local[t * half + i] = (float)(m->c.rope_factor * cos(angle_local));
            m->rope_sin_local[t * half + i] = (float)(m->c.rope_factor * sin(angle_local));
        }
}

/* Adjacent-pair rotation of the first 2*pairs coordinates of each head; the
 * rest pass through (partial rotary). The importer permutes split-half
 * checkpoints into this order. */
static void rope(float *x, uint32_t heads, uint32_t hd, uint32_t pairs, const float *cs, const float *sn) {
    uint32_t h, i;
    for (h = 0; h < heads; ++h, x += hd)
        for (i = 0; i < pairs; ++i) {
            float a = x[2 * i], b = x[2 * i + 1];
            x[2 * i] = a * cs[i] - b * sn[i];
            x[2 * i + 1] = a * sn[i] + b * cs[i];
        }
}

static void add_rows(float *x, const float *y, size_t n) {
    size_t i;
    for (i = 0; i < n; ++i) x[i] += y[i];
}

/* ---- attention --------------------------------------------------------- */

typedef struct AttnTask {
    Model *m;
    uint32_t layer, ntok, pos0;
} AttnTask;

/* In Gemma 3, every 6th layer (5, 11, 17, ...) is global; the other 5 are local. */
static int is_local_layer(const Config *c, uint32_t layer) {
    return c->sliding_window > 0 && (!c->gemma3 || (layer + 1) % 6 != 0);
}

static void attend(Model *m, uint32_t layer, size_t layer_base, uint32_t t, uint32_t h, uint32_t pos, float *scores) {
    const Config *c = &m->c;
    uint32_t hd = m->head_dim, kh = h / (c->heads / c->kv_heads), p, j;
    uint32_t start = is_local_layer(c, layer) && pos >= c->sliding_window ? pos - c->sliding_window + 1 : 0;
    const float *q = m->q + (size_t)t * m->q_dim + (size_t)h * hd;
    const float *keys = m->keys + layer_base + (size_t)kh * hd;
    const float *values = m->values + layer_base + (size_t)kh * hd;
    float *out = m->att + (size_t)t * m->q_dim + (size_t)h * hd;
    float factor = 1.0f / sqrtf((float)hd), high = -FLT_MAX, total = 0, inv;
    for (p = start; p <= pos; ++p) {
        float s = dot_f32(q, keys + (size_t)p * m->kv_dim, hd) * factor;
        scores[p] = s;
        if (s > high) high = s;
    }
    for (p = start; p <= pos; ++p) {
        scores[p] = exp_det(scores[p] - high);
        total += scores[p];
    }
    inv = 1.0f / total;
    for (j = 0; j < hd; ++j) out[j] = 0;
    for (p = start; p <= pos; ++p) {
        const float *v = values + (size_t)p * m->kv_dim;
        float w = scores[p] * inv;
        for (j = 0; j < hd; ++j) out[j] += w * v[j];
    }
}

static void attention_task(void *ctx, uint32_t index, uint32_t count) {
    AttnTask *a = (AttnTask *)ctx;
    Model *m = a->m;
    uint32_t heads = m->c.heads, begin, end, i;
    size_t layer_base = (size_t)a->layer * m->context * m->kv_dim;
    float *scores = m->scores + (size_t)index * m->context;
    split_range(a->ntok * heads, index, count, &begin, &end);
    for (i = begin; i < end; ++i) attend(m, a->layer, layer_base, i / heads, i % heads, a->pos0 + i / heads, scores);
}

static void attention_block(Model *m, uint32_t l, uint32_t n, uint32_t pos0) {
    const Config *c = &m->c;
    Layer *L = &m->layers[l];
    uint32_t t, half = c->rotary_dim / 2;
    int local = is_local_layer(c, l);
    const float *rcos = local ? m->rope_cos_local : m->rope_cos;
    const float *rsin = local ? m->rope_sin_local : m->rope_sin;
    MatItem qkv[3];
    AttnTask task;
    double clock = clock_start(m);
    norm_quantize(m, m->x, L->attn_norm, n);
    lap(m, PHASE_NORM, &clock);
    qkv[0] = item(&L->q, L->q_bias, m->q);
    qkv[1] = item(&L->k, L->k_bias, m->k);
    qkv[2] = item(&L->v, L->v_bias, m->v);
    matmul(m, qkv, 3, m->xd, m->xq, n);
    lap(m, PHASE_QKV, &clock);
    for (t = 0; t < n; ++t) {
        size_t slot = ((size_t)l * m->context + pos0 + t) * m->kv_dim;
        float *qt = m->q + (size_t)t * m->q_dim, *kt = m->k + (size_t)t * m->kv_dim;
        if (L->q_norm) norm_rows_inplace(qt, L->q_norm, c->heads, m->head_dim, c->eps);
        if (L->k_norm) norm_rows_inplace(kt, L->k_norm, c->kv_heads, m->head_dim, c->eps);
        rope(qt, c->heads, m->head_dim, half, rcos + t * half, rsin + t * half);
        rope(kt, c->kv_heads, m->head_dim, half, rcos + t * half, rsin + t * half);
        memcpy(m->keys + slot, kt, m->kv_dim * sizeof(float));
        memcpy(m->values + slot, m->v + (size_t)t * m->kv_dim, m->kv_dim * sizeof(float));
    }
    lap(m, PHASE_ROPE, &clock);
    task.m = m;
    task.layer = l;
    task.ntok = n;
    task.pos0 = pos0;
    pool_run(m->pool, attention_task, &task);
    lap(m, PHASE_ATTENTION, &clock);
    quantize_attention(m, n);
    lap(m, PHASE_ATTN_QUANT, &clock);
    qkv[0] = item(&L->o, NULL, m->out);
    matmul(m, qkv, 1, m->xd, m->xq, n);
    lap(m, PHASE_ATTN_OUT, &clock);
    if (L->post_attn_norm) norm_rows_inplace(m->out, L->post_attn_norm, n, c->dim, c->eps);
    add_rows(m->x, m->out, (size_t)n * c->dim);
    lap(m, PHASE_RESIDUAL, &clock);
}

/* ---- feed-forward and routing ------------------------------------------ */

/* out[t] = down(act(gate x_t) * up x_t) for ntok quantized inputs. */
static void expert_mlp(Model *m, const Expert *e, const float *xd, const int8_t *xq, uint32_t ntok) {
    MatItem items[2];
    double clock = clock_start(m);
    items[0] = item(&e->gate, NULL, m->gate);
    items[1] = item(&e->up, NULL, m->up);
    matmul(m, items, 2, xd, xq, ntok);
    lap(m, PHASE_FFN_UP, &clock);
    silu_quantize(m, ntok);
    lap(m, PHASE_ACTIVATION, &clock);
    items[0] = item(&e->down, NULL, m->out);
    matmul(m, items, 1, m->hd, m->hq, ntok);
    lap(m, PHASE_FFN_DOWN, &clock);
    m->expert_calls += ntok;
}

/* Exact top-k, lower expert index wins ties; sigmoid weights renormalized
 * over the selected experts only. */
static void route_tokens(Model *m, uint32_t n) {
    const Config *c = &m->c;
    uint32_t t, k, e, j;
    for (t = 0; t < n; ++t) {
        const float *r = m->route + (size_t)t * c->experts;
        uint32_t *sel = m->selected + (size_t)t * c->top_k;
        float *w = m->weights + (size_t)t * c->top_k, total = 0;
        for (k = 0; k < c->top_k; ++k) {
            uint32_t best = UINT32_MAX;
            for (e = 0; e < c->experts; ++e) {
                int taken = 0;
                for (j = 0; j < k; ++j) taken |= sel[j] == e;
                if (!taken && (best == UINT32_MAX || r[e] > r[best])) best = e;
            }
            sel[k] = best;
            w[k] = 1.0f / (1.0f + exp_det(-r[best]));
            total += w[k];
        }
        if (!(total > 0) || !isfinite(total)) die("router underflow/overflow");
        for (k = 0; k < c->top_k; ++k) w[k] /= total;
    }
}

static void routed_experts(Model *m, const Layer *L, uint32_t n) {
    const Config *c = &m->c;
    uint32_t e, t, k, j, dim = c->dim, nbd = m->nb_dim;
    MatItem router = item(&L->router, NULL, m->route);
    double clock = clock_start(m);
    matmul(m, &router, 1, m->xd, m->xq, n);
    route_tokens(m, n);
    for (e = 0; e < c->experts; ++e) {
        uint32_t g = 0;
        for (t = 0; t < n * c->top_k; ++t)
            if (m->selected[t] == e) m->members[g++] = t;
        if (!g) continue;
        for (j = 0; j < g; ++j) {
            uint32_t tok = m->members[j] / c->top_k;
            memcpy(m->gxd + (size_t)j * nbd, m->xd + (size_t)tok * nbd, nbd * sizeof(float));
            memcpy(m->gxq + (size_t)j * nbd * QK, m->xq + (size_t)tok * nbd * QK, (size_t)nbd * QK);
        }
        lap(m, PHASE_ROUTING, &clock);
        expert_mlp(m, &L->experts[e], m->gxd, m->gxq, g); /* times its own phases */
        clock = clock_start(m);
        for (j = 0; j < g; ++j)
            memcpy(m->expert_out + (size_t)m->members[j] * dim, m->out + (size_t)j * dim, dim * sizeof(float));
    }
    lap(m, PHASE_ROUTING, &clock);
    /* Accumulate in selection order so batching never changes the sum. */
    for (t = 0; t < n; ++t)
        for (k = 0; k < c->top_k; ++k) {
            const float *y = m->expert_out + ((size_t)t * c->top_k + k) * dim;
            float w = m->weights[(size_t)t * c->top_k + k];
            float *x = m->x + (size_t)t * dim;
            for (j = 0; j < dim; ++j) x[j] += w * y[j];
        }
    lap(m, PHASE_RESIDUAL, &clock);
}

static void ffn_block(Model *m, uint32_t l, uint32_t n) {
    const Config *c = &m->c;
    const Layer *L = &m->layers[l];
    uint32_t s;
    double clock = clock_start(m);
    norm_quantize(m, m->x, L->ffn_norm, n);
    lap(m, PHASE_NORM, &clock);
    if (c->experts == 1) {
        /* A single expert's normalized router weight is exactly 1. */
        expert_mlp(m, &L->experts[0], m->xd, m->xq, n);
        clock = clock_start(m);
        if (L->post_ffn_norm) norm_rows_inplace(m->out, L->post_ffn_norm, n, c->dim, c->eps);
        add_rows(m->x, m->out, (size_t)n * c->dim);
        lap(m, PHASE_RESIDUAL, &clock);
    } else {
        routed_experts(m, L, n);
    }
    for (s = 0; s < c->shared; ++s) {
        expert_mlp(m, &L->shared[s], m->xd, m->xq, n);
        clock = clock_start(m);
        if (L->post_ffn_norm) norm_rows_inplace(m->out, L->post_ffn_norm, n, c->dim, c->eps);
        add_rows(m->x, m->out, (size_t)n * c->dim);
        lap(m, PHASE_RESIDUAL, &clock);
    }
}

/* ---- forward ----------------------------------------------------------- */

static void output_logits(Model *m, uint32_t n, LogitsMode mode) {
    const Config *c = &m->c;
    uint32_t first = mode == LOGITS_ALL ? 0 : n - 1, rows = n - first;
    MatItem head = item(&m->head, NULL, m->logits);
    if (rows > m->logit_rows) die("internal error: logits buffer too small");
    norm_quantize(m, m->x + (size_t)first * c->dim, m->final_norm, rows);
    matmul(m, &head, 1, m->xd, m->xq, rows);
}

void model_forward(Model *m, const uint32_t *tokens, uint32_t n, uint32_t pos0, LogitsMode mode) {
    const Config *c = &m->c;
    uint32_t t, l, j;
    float embed_scale = c->gemma3 ? sqrtf((float)c->dim) : 1.0f;
    double clock = clock_start(m);
    if (!n || n > m->batch || pos0 > m->context || n > m->context - pos0) die("token batch exceeds context");
    for (t = 0; t < n; ++t) {
        if (tokens[t] >= c->vocab) die("token %u out of vocabulary", tokens[t]);
        const unsigned char *row = m->embed.data + (size_t)tokens[t] * m->embed.row_bytes;
        float *x = m->x + (size_t)t * c->dim;
        if (m->embed.dtype == DTYPE_Q5) dequantize_q5_row(row, c->dim, x);
        else dequantize_q8_row(row, c->dim, x);
        if (c->gemma3)
            for (j = 0; j < c->dim; ++j) x[j] *= embed_scale;
    }
    lap(m, PHASE_EMBED, &clock);
    rope_tables(m, n, pos0);
    lap(m, PHASE_ROPE, &clock);
    for (l = 0; l < c->layers; ++l) {
        attention_block(m, l, n, pos0); /* blocks time their own phases */
        ffn_block(m, l, n);
    }
    clock = clock_start(m);
    if (mode != LOGITS_NONE) output_logits(m, n, mode);
    lap(m, PHASE_LOGITS, &clock);
}

void model_prefill(Model *m, const uint32_t *tokens, uint32_t n, uint32_t pos0, int last_logits) {
    uint32_t done = 0;
    while (done < n) {
        uint32_t chunk = n - done < m->batch ? n - done : m->batch;
        LogitsMode mode = last_logits && done + chunk == n ? LOGITS_LAST : LOGITS_NONE;
        model_forward(m, tokens + done, chunk, pos0 + done, mode);
        done += chunk;
    }
}
