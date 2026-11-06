#include "model.h"
#include "kernels.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ---- weights ----------------------------------------------------------- */

static QMat qmat(const ModelFile *f, const char *name, uint32_t rows, uint32_t cols) {
    const Tensor *t = modelfile_find(f, name, rows, cols, DTYPE_Q8);
    QMat q;
    q.data = t->data;
    q.rows = rows;
    q.cols = cols;
    q.nb = (cols + QK - 1) / QK;
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

static void load_layer(Layer *l, const ModelFile *f, const Config *c, uint32_t i, uint32_t kv_dim, Memory *mem) {
    char name[TENSOR_NAME_SIZE];
    uint32_t e;
    l->attn_norm = fvec(f, layer_name(name, i, "attn_norm"), c->dim, mem);
    l->ffn_norm = fvec(f, layer_name(name, i, "ffn_norm"), c->dim, mem);
    l->q = qmat(f, layer_name(name, i, "q"), c->dim, c->dim);
    l->k = qmat(f, layer_name(name, i, "k"), kv_dim, c->dim);
    l->v = qmat(f, layer_name(name, i, "v"), kv_dim, c->dim);
    l->o = qmat(f, layer_name(name, i, "o"), c->dim, c->dim);
    l->router = qmat(f, layer_name(name, i, "router"), c->experts, c->dim);
    if (c->qkv_bias) {
        l->q_bias = fvec(f, layer_name(name, i, "q_bias"), c->dim, mem);
        l->k_bias = fvec(f, layer_name(name, i, "k_bias"), kv_dim, mem);
        l->v_bias = fvec(f, layer_name(name, i, "v_bias"), kv_dim, mem);
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
    uint32_t i, half, threads = pool_threads(pool);
    memset(m, 0, sizeof(*m));
    m->c = *c;
    m->context = context;
    m->batch = batch;
    m->logit_rows = logit_rows;
    m->pool = pool;
    m->head_dim = c->dim / c->heads;
    m->kv_dim = c->kv_heads * m->head_dim;
    m->nb_dim = (c->dim + QK - 1) / QK;
    m->nb_hidden = (c->hidden + QK - 1) / QK;
    half = m->head_dim / 2;
    expected = 3 + (uint64_t)c->layers * (7 + 3 * c->qkv_bias + 3 * (uint64_t)(c->experts + c->shared));
    if (expected != f->count) die("tensor count does not match config");

    m->embed = qmat(f, "embed", c->vocab, c->dim);
    m->head = qmat(f, "lm_head", c->vocab, c->dim);
    m->final_norm = fvec(f, "final_norm", c->dim, mem);
    m->layers = (Layer *)mem_alloc(mem, c->layers, sizeof(Layer));
    for (i = 0; i < c->layers; ++i) load_layer(&m->layers[i], f, c, i, m->kv_dim, mem);

    m->x = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->xn = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->q = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->att = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->out = (float *)buffer(mem, batch, c->dim, sizeof(float));
    m->k = (float *)buffer(mem, batch, m->kv_dim, sizeof(float));
    m->v = (float *)buffer(mem, batch, m->kv_dim, sizeof(float));
    m->gate = (float *)buffer(mem, batch, c->hidden, sizeof(float));
    m->up = (float *)buffer(mem, batch, c->hidden, sizeof(float));
    m->xd = (float *)buffer(mem, batch, m->nb_dim, sizeof(float));
    m->xq = (int8_t *)buffer(mem, batch, (size_t)m->nb_dim * QK, 1);
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
    for (i = 0; i < half; ++i) m->inv_freq[i] = pow((double)c->theta, -(double)(2 * i) / (double)m->head_dim);
    m->rope_cos = (float *)buffer(mem, batch, half, sizeof(float));
    m->rope_sin = (float *)buffer(mem, batch, half, sizeof(float));
    m->keys = (float *)buffer(mem, mul_size(c->layers, context), m->kv_dim, sizeof(float));
    m->values = (float *)buffer(mem, mul_size(c->layers, context), m->kv_dim, sizeof(float));
    m->scores = (float *)buffer(mem, threads, context, sizeof(float));
}

size_t model_kv_bytes(const Model *m) {
    return mul_size(mul_size(mul_size(m->c.layers, m->context), m->kv_dim), 2 * sizeof(float));
}

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
    size_t row_bytes = (size_t)w->nb * Q8_BLOCK_BYTES, xs = (size_t)t->nb * QK;
    uint32_t r, k, j;
    int ok = 1;
    for (r = r0; r < r1; ++r) {
        const unsigned char *row = w->data + row_bytes * r;
        for (k = 0; k + 4 <= t->ntok; k += 4) {
            float y[4];
            dot_q8_x4(row, t->xd + (size_t)k * t->nb, t->xq + k * xs, t->nb, t->nb, y);
            for (j = 0; j < 4; ++j) ok &= store(it, k + j, r, y[j]);
        }
        for (; k < t->ntok; ++k)
            ok &= store(it, k, r, dot_q8(row, t->xd + (size_t)k * t->nb, t->xq + k * xs, t->nb));
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
            __atomic_store_n(&t->fault, 1, __ATOMIC_RELAXED);
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

/* ---- elementwise pieces ------------------------------------------------ */

static void rmsnorm(const float *x, const float *w, float *y, uint32_t n, float eps) {
    float sq = 0, inv;
    uint32_t i;
    for (i = 0; i < n; ++i) sq += x[i] * x[i];
    if (!isfinite(sq)) die("non-finite RMSNorm input");
    inv = 1.0f / sqrtf(sq / (float)n + eps);
    if (!isfinite(inv)) die("non-finite RMSNorm");
    for (i = 0; i < n; ++i) {
        y[i] = x[i] * inv * w[i];
        if (!isfinite(y[i])) die("non-finite RMSNorm output");
    }
}

static void quantize_rows(const float *x, uint32_t rows, uint32_t cols, uint32_t nb, float *d, int8_t *q) {
    uint32_t r;
    for (r = 0; r < rows; ++r)
        if (quantize_q8(x + (size_t)r * cols, cols, d + (size_t)r * nb, q + (size_t)r * nb * QK))
            die("non-finite activation before quantization");
}

static void norm_quantize(Model *m, const float *weight, uint32_t n) {
    uint32_t t, dim = m->c.dim;
    for (t = 0; t < n; ++t) rmsnorm(m->x + (size_t)t * dim, weight, m->xn + (size_t)t * dim, dim, m->c.eps);
    quantize_rows(m->xn, n, dim, m->nb_dim, m->xd, m->xq);
}

static void rope_tables(Model *m, uint32_t n, uint32_t pos0) {
    uint32_t t, i, half = m->head_dim / 2;
    for (t = 0; t < n; ++t)
        for (i = 0; i < half; ++i) {
            double angle = (double)(pos0 + t) * m->inv_freq[i];
            m->rope_cos[t * half + i] = (float)cos(angle);
            m->rope_sin[t * half + i] = (float)sin(angle);
        }
}

/* Adjacent-pair rotation (the importer permutes split-half checkpoints). */
static void rope(float *x, uint32_t heads, uint32_t hd, const float *cs, const float *sn) {
    uint32_t h, i;
    for (h = 0; h < heads; ++h, x += hd)
        for (i = 0; i < hd / 2; ++i) {
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

static void attend(Model *m, size_t layer_base, uint32_t t, uint32_t h, uint32_t pos, float *scores) {
    const Config *c = &m->c;
    uint32_t hd = m->head_dim, kh = h / (c->heads / c->kv_heads), p, j;
    const float *q = m->q + (size_t)t * c->dim + (size_t)h * hd;
    const float *keys = m->keys + layer_base + (size_t)kh * hd;
    const float *values = m->values + layer_base + (size_t)kh * hd;
    float *out = m->att + (size_t)t * c->dim + (size_t)h * hd;
    float factor = 1.0f / sqrtf((float)hd), high = -FLT_MAX, total = 0, inv;
    for (p = 0; p <= pos; ++p) {
        float s = dot_f32(q, keys + (size_t)p * m->kv_dim, hd) * factor;
        scores[p] = s;
        if (s > high) high = s;
    }
    for (p = 0; p <= pos; ++p) {
        scores[p] = expf(scores[p] - high);
        total += scores[p];
    }
    inv = 1.0f / total;
    for (j = 0; j < hd; ++j) out[j] = 0;
    for (p = 0; p <= pos; ++p) {
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
    for (i = begin; i < end; ++i) attend(m, layer_base, i / heads, i % heads, a->pos0 + i / heads, scores);
}

static void attention_block(Model *m, uint32_t l, uint32_t n, uint32_t pos0) {
    const Config *c = &m->c;
    Layer *L = &m->layers[l];
    uint32_t t, half = m->head_dim / 2;
    MatItem qkv[3];
    AttnTask task;
    norm_quantize(m, L->attn_norm, n);
    qkv[0] = item(&L->q, L->q_bias, m->q);
    qkv[1] = item(&L->k, L->k_bias, m->k);
    qkv[2] = item(&L->v, L->v_bias, m->v);
    matmul(m, qkv, 3, m->xd, m->xq, n);
    for (t = 0; t < n; ++t) {
        size_t slot = ((size_t)l * m->context + pos0 + t) * m->kv_dim;
        rope(m->q + (size_t)t * c->dim, c->heads, m->head_dim, m->rope_cos + t * half, m->rope_sin + t * half);
        rope(m->k + (size_t)t * m->kv_dim, c->kv_heads, m->head_dim, m->rope_cos + t * half, m->rope_sin + t * half);
        memcpy(m->keys + slot, m->k + (size_t)t * m->kv_dim, m->kv_dim * sizeof(float));
        memcpy(m->values + slot, m->v + (size_t)t * m->kv_dim, m->kv_dim * sizeof(float));
    }
    task.m = m;
    task.layer = l;
    task.ntok = n;
    task.pos0 = pos0;
    pool_run(m->pool, attention_task, &task);
    quantize_rows(m->att, n, c->dim, m->nb_dim, m->xd, m->xq);
    qkv[0] = item(&L->o, NULL, m->out);
    matmul(m, qkv, 1, m->xd, m->xq, n);
    add_rows(m->x, m->out, (size_t)n * c->dim);
}

/* ---- feed-forward and routing ------------------------------------------ */

/* out[t] = down(SiLU(gate x_t) * up x_t) for ntok quantized inputs. */
static void expert_mlp(Model *m, const Expert *e, const float *xd, const int8_t *xq, uint32_t ntok) {
    uint32_t hidden = m->c.hidden;
    size_t i, total = (size_t)ntok * hidden;
    MatItem items[2];
    items[0] = item(&e->gate, NULL, m->gate);
    items[1] = item(&e->up, NULL, m->up);
    matmul(m, items, 2, xd, xq, ntok);
    for (i = 0; i < total; ++i) m->up[i] *= m->gate[i] / (1.0f + expf(-m->gate[i]));
    quantize_rows(m->up, ntok, hidden, m->nb_hidden, m->hd, m->hq);
    items[0] = item(&e->down, NULL, m->out);
    matmul(m, items, 1, m->hd, m->hq, ntok);
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
            w[k] = 1.0f / (1.0f + expf(-r[best]));
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
        expert_mlp(m, &L->experts[e], m->gxd, m->gxq, g);
        for (j = 0; j < g; ++j)
            memcpy(m->expert_out + (size_t)m->members[j] * dim, m->out + (size_t)j * dim, dim * sizeof(float));
    }
    /* Accumulate in selection order so batching never changes the sum. */
    for (t = 0; t < n; ++t)
        for (k = 0; k < c->top_k; ++k) {
            const float *y = m->expert_out + ((size_t)t * c->top_k + k) * dim;
            float w = m->weights[(size_t)t * c->top_k + k];
            float *x = m->x + (size_t)t * dim;
            for (j = 0; j < dim; ++j) x[j] += w * y[j];
        }
}

static void ffn_block(Model *m, uint32_t l, uint32_t n) {
    const Config *c = &m->c;
    const Layer *L = &m->layers[l];
    uint32_t s;
    norm_quantize(m, L->ffn_norm, n);
    if (c->experts == 1) {
        /* A single expert's normalized router weight is exactly 1. */
        expert_mlp(m, &L->experts[0], m->xd, m->xq, n);
        add_rows(m->x, m->out, (size_t)n * c->dim);
    } else {
        routed_experts(m, L, n);
    }
    for (s = 0; s < c->shared; ++s) {
        expert_mlp(m, &L->shared[s], m->xd, m->xq, n);
        add_rows(m->x, m->out, (size_t)n * c->dim);
    }
}

/* ---- forward ----------------------------------------------------------- */

static void output_logits(Model *m, uint32_t n, LogitsMode mode) {
    const Config *c = &m->c;
    uint32_t first = mode == LOGITS_ALL ? 0 : n - 1, rows = n - first, t;
    MatItem head = item(&m->head, NULL, m->logits);
    if (rows > m->logit_rows) die("internal error: logits buffer too small");
    for (t = 0; t < rows; ++t)
        rmsnorm(m->x + (size_t)(first + t) * c->dim, m->final_norm, m->xn + (size_t)t * c->dim, c->dim, c->eps);
    quantize_rows(m->xn, rows, c->dim, m->nb_dim, m->xd, m->xq);
    matmul(m, &head, 1, m->xd, m->xq, rows);
}

void model_forward(Model *m, const uint32_t *tokens, uint32_t n, uint32_t pos0, LogitsMode mode) {
    const Config *c = &m->c;
    uint32_t t, l;
    if (!n || n > m->batch || pos0 > m->context || n > m->context - pos0) die("token batch exceeds context");
    for (t = 0; t < n; ++t) {
        if (tokens[t] >= c->vocab) die("token %u out of vocabulary", tokens[t]);
        dequantize_q8_row(m->embed.data + (size_t)tokens[t] * m->embed.nb * Q8_BLOCK_BYTES, c->dim,
                          m->x + (size_t)t * c->dim);
    }
    rope_tables(m, n, pos0);
    for (l = 0; l < c->layers; ++l) {
        attention_block(m, l, n, pos0);
        ffn_block(m, l, n);
    }
    if (mode != LOGITS_NONE) output_logits(m, n, mode);
}
