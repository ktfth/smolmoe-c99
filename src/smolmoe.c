/* smolmoe: a bounded-memory sparse decoder. MIT licensed; see LICENSE.
 * C99 core, POSIX file/timing/stats layer. This is NOT Kimi K3 compatible.
 * File formats and arithmetic contract are specified in docs/FORMAT.md. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#if !defined(SM_NO_AVX2) && (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define SM_AVX2 1
#include <immintrin.h>
#else
#define SM_AVX2 0
#endif

#define TILE 65520u /* 1820 complete Q8 blocks */
#define WAYS 4u
#define MAX_CONTEXT 1048576u
#define MAX_DIM 65536u
#define MAX_TENSORS 2000000u
#define MAX_MERGES 262144u
#define NAME_SIZE 40u
#define REC_SIZE 72u

static void die(const char *fmt, ...) {
    va_list ap;
    fputs("smolmoe: ", stderr);
    va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    fputc('\n', stderr); exit(EXIT_FAILURE);
}

typedef union Allocation Allocation;
union Allocation { struct { Allocation *next; size_t bytes; } h; long double align; };
typedef struct { size_t limit, used, peak; Allocation *head; } Memory;

static size_t mul(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) die("size overflow");
    return a * b;
}
static void *alloc(Memory *m, size_t n, size_t size) {
    size_t bytes = mul(n, size);
    Allocation *p;
    if (bytes > SIZE_MAX - sizeof(*p)) die("allocation overflow");
    bytes += sizeof(*p);
    if (bytes > m->limit - m->used)
        die("memory budget too small: need at least %zu more bytes (used=%zu, budget=%zu); reduce --context or increase --memory-mib", bytes, m->used, m->limit);
    p = (Allocation *)calloc(1, bytes);
    if (!p) die("allocation failed for %zu bytes", bytes);
    p->h.next = m->head; p->h.bytes = bytes; m->head = p;
    m->used += bytes; if (m->used > m->peak) m->peak = m->used;
    return p + 1;
}
static void release(Memory *m) {
    Allocation *p = m->head;
    while (p) { Allocation *next = p->h.next; free(p); p = next; }
    m->head = NULL; m->used = 0;
}
static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24;
}
static uint64_t u64(const unsigned char *p) { return u32(p) | (uint64_t)u32(p+4)<<32; }
static float f32(const unsigned char *p) {
    uint32_t u = u32(p); float f; memcpy(&f, &u, 4); return f;
}
static void read_at(int fd, void *buf, size_t bytes, uint64_t off) {
    unsigned char *p = (unsigned char *)buf;
    if (off > INT64_MAX || bytes > (uint64_t)INT64_MAX - off) die("file offset overflow");
    while (bytes) {
        size_t want = bytes > 1048576 ? 1048576 : bytes;
        ssize_t n = pread(fd, p, want, (off_t)off);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) die("checkpoint read failed at offset %" PRIu64 ": %s", off, n ? strerror(errno) : "unexpected EOF");
        p += n; bytes -= (size_t)n; off += (uint64_t)n;
    }
}
static uint32_t number(const char *s, uint32_t max, const char *what) {
    char *end; unsigned long long n;
    if (!*s || *s == '-' || *s == '+') die("invalid %s: %s", what, s);
    errno = 0; n = strtoull(s, &end, 10);
    if (errno || *end || n > max) die("invalid %s: %s", what, s);
    return (uint32_t)n;
}
static double real_number(const char *s, const char *what) {
    char *end; double n;
    errno = 0; n = strtod(s, &end);
    if (!*s || errno || *end || !isfinite(n)) die("invalid %s: %s", what, s);
    return n;
}
static char *trim(char *s) {
    char *end;
    while (*s == ' ' || *s == '\t' || *s == '\r' || *s == '\n') ++s;
    end = s + strlen(s);
    while (end > s && (end[-1]==' ' || end[-1]=='\t' || end[-1]=='\r' || end[-1]=='\n')) --end;
    *end = 0; return s;
}
typedef struct {
    uint32_t dim, hidden, layers, heads, kv_heads, experts, top_k, vocab;
    uint32_t max_context, shared, bos, eos, qkv_bias;
    float eps, theta;
} Config;

static Config config_read(const char *path) {
    Config c; FILE *fp; char line[512]; unsigned seen = 0;
    const char *keys[] = {"architecture","dim","hidden_dim","layers","heads","kv_heads","experts","top_k","vocab_size","max_context","shared_experts","bos_id","eos_id","rms_eps","rope_theta"};
    uint32_t *fields[12];
    memset(&c, 0, sizeof(c));
    fields[0]=&c.dim; fields[1]=&c.hidden; fields[2]=&c.layers; fields[3]=&c.heads;
    fields[4]=&c.kv_heads; fields[5]=&c.experts; fields[6]=&c.top_k; fields[7]=&c.vocab;
    fields[8]=&c.max_context; fields[9]=&c.shared; fields[10]=&c.bos; fields[11]=&c.eos;
    fp = fopen(path, "r"); if (!fp) die("cannot open config: %s", strerror(errno));
    while (fgets(line, sizeof(line), fp)) {
        char *s, *eq, *value, *comment; unsigned k;
        if (!strchr(line, '\n') && !feof(fp)) die("config line too long");
        comment = strchr(line, '#'); if (comment) *comment=0;
        s=trim(line); if (!*s) continue;
        eq=strchr(s, '='); if (!eq) die("config requires key=value");
        *eq=0; s=trim(s); value=trim(eq+1);
        for (k=0; k<15 && strcmp(s,keys[k]); ++k) {}
        if (k==15) die("unknown config key: %s (only smolmoe_v1 is supported)",s);
        if (seen & (1u<<k)) die("duplicate config key: %s",s);
        seen |= 1u<<k;
        if (!k) {
            if(!strcmp(value,"smolmoe_qwen2")) c.qkv_bias=1;
            else if (strcmp(value,"smolmoe_v1")) die("unsupported architecture: %s; full Kimi K3 is not supported",value);
        }
        else if (k<13) *fields[k-1] = number(value, MAX_CONTEXT, s);
        else if (k==13) c.eps=(float)real_number(value,s);
        else c.theta=(float)real_number(value,s);
    }
    if (ferror(fp)) die("config read error");
    fclose(fp);
    if (seen != 32767u) die("config missing required fields");
    if (!c.dim || c.dim>MAX_DIM || !c.hidden || c.hidden>MAX_DIM || !c.layers || c.layers>1024 ||
        !c.heads || c.heads>c.dim || c.dim%c.heads || (c.dim/c.heads)%2 ||
        !c.kv_heads || c.heads%c.kv_heads || !c.experts || c.experts>4096 ||
        !c.top_k || c.top_k>c.experts || c.shared>16 || c.vocab<258 || c.vocab>258+MAX_MERGES ||
        !c.max_context || c.max_context>MAX_CONTEXT || c.bos!=256 || c.eos!=257 ||
        !(c.eps>0 && c.eps<1) || !(c.theta>1) || !isfinite(c.theta)) die("invalid config dimensions or tokenizer IDs");
    return c;
}

/* Byte-fallback BPE: 0..255 bytes, 256 BOS, 257 EOS, then ranked merges. */
typedef struct { uint32_t a,b,bytes; } Merge;
typedef struct QwenTokenizer QwenTokenizer;
typedef struct {
    Merge *merges; uint32_t count, *hash, hash_size;
    int sentencepiece; uint32_t vocab_size,max_piece;
    unsigned char **pieces,*scratch; uint32_t *lengths; float *piece_scores;
    QwenTokenizer *qwen;
} Tokenizer;
static uint64_t mix(uint64_t x) {
    x ^= x>>30; x *= UINT64_C(0xbf58476d1ce4e5b9); x ^= x>>27;
    x *= UINT64_C(0x94d049bb133111eb); return x^(x>>31);
}
#include "qwen_tokenizer.h"
static uint64_t string_hash(const unsigned char *p,uint32_t n) {
    uint64_t h=UINT64_C(14695981039346656037); uint32_t i;
    for(i=0;i<n;++i) {h^=p[i];h*=UINT64_C(1099511628211);}
    return h;
}
static uint32_t piece_lookup(const Tokenizer *t,const unsigned char *p,uint32_t n) {
    uint32_t slot=(uint32_t)string_hash(p,n)&(t->hash_size-1);
    while(t->hash[slot]) {
        uint32_t id=t->hash[slot]-1;
        if(t->lengths[id]==n && !memcmp(t->pieces[id],p,n)) return id;
        slot=(slot+1)&(t->hash_size-1);
    }
    return UINT32_MAX;
}
static Tokenizer sp_read(const char *path,uint32_t vocab,Memory *mem) {
    Tokenizer t; FILE *f=fopen(path,"rb"); unsigned char header[20],rec[8],*data;
    uint32_t i,total,used=0;
    memset(&t,0,sizeof(t));
    if(!f || fread(header,1,20,f)!=20 || memcmp(header,"SMSPB1\0\1",8)) die("invalid score-BPE header");
    t.sentencepiece=1;t.vocab_size=vocab;t.count=vocab-258;
    t.max_piece=u32(header+12);total=u32(header+16);
    if(u32(header+8)!=vocab || !t.max_piece || t.max_piece>65536 || total>67108864) die("invalid score-BPE dimensions");
    t.pieces=(unsigned char **)alloc(mem,vocab,sizeof(*t.pieces));
    t.lengths=(uint32_t *)alloc(mem,vocab,sizeof(uint32_t));
    t.piece_scores=(float *)alloc(mem,vocab,sizeof(float));
    data=(unsigned char *)alloc(mem,total,1);t.scratch=(unsigned char *)alloc(mem,2*t.max_piece+8,1);
    t.hash_size=8;while(t.hash_size<vocab*2) t.hash_size*=2;
    t.hash=(uint32_t *)alloc(mem,t.hash_size,sizeof(uint32_t));
    for(i=0;i<vocab;++i) {
        uint32_t n,slot;
        if(fread(rec,1,8,f)!=8) die("truncated score-BPE record");
        t.piece_scores[i]=f32(rec);n=u32(rec+4);
        if(!isfinite(t.piece_scores[i]) || !n || n>t.max_piece || n>total-used) die("invalid score-BPE record");
        t.pieces[i]=data+used;t.lengths[i]=n;used+=n;
        if(fread(t.pieces[i],1,n,f)!=n) die("truncated score-BPE piece");
        if(piece_lookup(&t,t.pieces[i],n)!=UINT32_MAX) die("duplicate score-BPE piece");
        slot=(uint32_t)string_hash(t.pieces[i],n)&(t.hash_size-1);
        while(t.hash[slot]) slot=(slot+1)&(t.hash_size-1);
        t.hash[slot]=i+1;
    }
    if(used!=total || fgetc(f)!=EOF || ferror(f)) die("invalid score-BPE payload size");
    fclose(f);
    if(piece_lookup(&t,(const unsigned char *)" ",1)==UINT32_MAX) die("score-BPE missing dummy prefix");
    return t;
}
static Tokenizer tokenizer_read(const char *path, uint32_t vocab, Memory *mem) {
    Tokenizer t; FILE *fp=fopen(path,"r"); char line[128]; uint32_t i;
    memset(&t,0,sizeof(t)); if (!fp) die("cannot open tokenizer: %s",strerror(errno));
    if(fread(line,1,8,fp)==8 && !memcmp(line,"SMQBP1\0\1",8)) {fclose(fp);t.count=vocab-258;t.qwen=qwen_read(path,vocab,mem);return t;}
    rewind(fp);
    if(fread(line,1,8,fp)==8 && !memcmp(line,"SMSPB1\0\1",8)) {fclose(fp);return sp_read(path,vocab,mem);}
    rewind(fp);
    if (!fgets(line,sizeof(line),fp) || strcmp(trim(line),"SMBPE1")) die("invalid tokenizer magic");
    t.count=vocab-258; t.hash_size=8;
    while (t.hash_size<t.count*2) t.hash_size*=2;
    t.merges=(Merge *)alloc(mem,t.count,sizeof(Merge));
    t.hash=(uint32_t *)alloc(mem,t.hash_size,sizeof(uint32_t));
    for(i=0;i<t.count;++i) {
        char a[32], b[32], extra; uint32_t p,h;
        if (!fgets(line,sizeof(line),fp) || sscanf(line,"%31s %31s %c",a,b,&extra)!=2) die("invalid tokenizer merge %u",i);
        t.merges[i].a=number(a,257+i,"merge token"); t.merges[i].b=number(b,257+i,"merge token");
        if(t.merges[i].a==256 || t.merges[i].a==257 || t.merges[i].b==256 || t.merges[i].b==257) die("special token in BPE merge");
        t.merges[i].bytes=(t.merges[i].a<256?1:t.merges[t.merges[i].a-258].bytes)+
                          (t.merges[i].b<256?1:t.merges[t.merges[i].b-258].bytes);
        if(t.merges[i].bytes>65536) die("BPE token expands beyond 65536 bytes");
        h=(uint32_t)mix(((uint64_t)t.merges[i].a<<32)|t.merges[i].b)&(t.hash_size-1);
        for(p=h;t.hash[p];p=(p+1)&(t.hash_size-1)) {
            Merge old=t.merges[t.hash[p]-1];
            if(old.a==t.merges[i].a && old.b==t.merges[i].b) die("duplicate BPE pair");
        }
        t.hash[p]=i+1;
    }
    while (fgets(line,sizeof(line),fp)) if (*trim(line)) die("extra tokenizer merges");
    if(ferror(fp)) die("tokenizer read error");
    fclose(fp); return t;
}
static uint32_t sp_encode(const Tokenizer *t,const unsigned char *s,uint32_t bytes,uint32_t *ids) {
    uint32_t n=1,i=0;
    ids[0]=256;
    if(bytes) ids[n++]=piece_lookup(t,(const unsigned char *)" ",1);
    while(i<bytes) {
        uint32_t j=i+1,id,k;
        while(j<bytes && j-i<4 && (s[j]&0xc0)==0x80) ++j;
        id=piece_lookup(t,s+i,j-i);
        if(id!=UINT32_MAX) ids[n++]=id;
        else for(k=i;k<j;++k) ids[n++]=s[k];
        i=j;
    }
    for(;;) {
        float best=-FLT_MAX;uint32_t best_id=UINT32_MAX,pos=0;
        for(i=1;i+1<n;++i) {
            uint32_t a=ids[i],b=ids[i+1],na=t->lengths[a],nb=t->lengths[b],id;
            if(na+nb>t->max_piece) continue;
            memcpy(t->scratch,t->pieces[a],na);memcpy(t->scratch+na,t->pieces[b],nb);
            id=piece_lookup(t,t->scratch,na+nb);
            if(id!=UINT32_MAX && t->piece_scores[id]>best) {best=t->piece_scores[id];best_id=id;pos=i;}
        }
        if(best_id==UINT32_MAX) break;
        ids[pos]=best_id;memmove(ids+pos+1,ids+pos+2,(n-pos-2)*sizeof(*ids));--n;
    }
    return n;
}
static uint32_t merge_rank(const Tokenizer *t, uint32_t a, uint32_t b) {
    uint32_t p=(uint32_t)mix(((uint64_t)a<<32)|b)&(t->hash_size-1);
    while(t->hash[p]) {
        uint32_t r=t->hash[p]-1;
        if(t->merges[r].a==a && t->merges[r].b==b) return r;
        p=(p+1)&(t->hash_size-1);
    }
    return UINT32_MAX;
}
/* Deterministic leftmost, lowest-rank merging, deliberately small O(n^2).
 * Prompt bytes are capped before encoding, so temporary space is bounded. */
static uint32_t encode(const Tokenizer *t, const unsigned char *s, uint32_t bytes, uint32_t *ids) {
    uint32_t n=bytes+1,i;
    if(t->sentencepiece) return sp_encode(t,s,bytes,ids);
    ids[0]=256; for(i=0;i<bytes;++i) ids[i+1]=s[i];
    for(;;) {
        uint32_t best=UINT32_MAX,pos=0;
        for(i=1;i+1<n;++i) { uint32_t r=merge_rank(t,ids[i],ids[i+1]); if(r<best) {best=r;pos=i;} }
        if(best==UINT32_MAX) break;
        ids[pos]=258+best; memmove(ids+pos+1,ids+pos+2,(n-pos-2)*sizeof(*ids)); --n;
    }
    return n;
}
/* Iterative decoding avoids stack overflow for maliciously deep merge trees. */
static void decode(const Tokenizer *t, uint32_t id, uint32_t *stack, uint32_t cap, FILE *out) {
    uint32_t n=0; stack[n++]=id;
    if(t->qwen){QwenTokenizer *q=t->qwen;if(id>=q->vocab)die("invalid Qwen output token");if(!q->special[id] && fwrite(q->piece[id],1,q->length[id],out)!=q->length[id])die("Qwen output write error");return;}
    if(t->sentencepiece) {
        if(id>=t->vocab_size) die("invalid score-BPE token ID");
        if(id<256) {if(fputc((int)id,out)==EOF) die("output write failed");}
        else if(id!=256 && id!=257 && fwrite(t->pieces[id],1,t->lengths[id],out)!=t->lengths[id]) die("output write failed");
        return;
    }
    while(n) {
        uint32_t x=stack[--n];
        if(x<256) { if(fputc((int)x,out)==EOF) die("output write failed"); }
        else if(x>=258 && x-258<t->count) {
            Merge m=t->merges[x-258];
            if(n+2>cap) die("BPE decode stack overflow");
            stack[n++]=m.b; stack[n++]=m.a;
        } else if(x!=256 && x!=257) die("invalid output token");
    }
}

typedef struct { char name[NAME_SIZE]; uint32_t rows,cols,dtype,expert; uint64_t off,bytes; } Tensor;
typedef struct { uint64_t key, stamp; uint32_t len; } Slot;
typedef struct {
    int fd, drop_cache, dense; Tensor *t; uint32_t nt;
    Slot *slots; unsigned char *data; size_t nslots, core_slots;
    uint64_t tick,hits,misses,read_bytes,expert_reads;
} Store;

static Store store_open(const char *path, Memory *mem) {
    Store s; struct stat st; unsigned char header[16], rec[REC_SIZE]; uint32_t i;
    uint64_t previous; uint32_t version;
    memset(&s,0,sizeof(s)); s.fd=open(path,O_RDONLY); if(s.fd<0) die("cannot open checkpoint: %s",strerror(errno));
    if(fstat(s.fd,&st) || st.st_size<16) die("invalid checkpoint size");
    read_at(s.fd,header,16,0);
    version=header[7];
    if(memcmp(header,"SMOEQ8\0",7) || (version!=1 && version!=2) || u32(header+12)) die("unsupported checkpoint format (expected SMOEQ8 v1/v2)");
    s.nt=u32(header+8); if(!s.nt || s.nt>MAX_TENSORS) die("invalid tensor count");
    previous=16+(uint64_t)s.nt*REC_SIZE;
    if(previous>(uint64_t)st.st_size) die("truncated checkpoint directory");
    s.t=(Tensor *)alloc(mem,s.nt,sizeof(Tensor));
    for(i=0;i<s.nt;++i) {
        Tensor *t=&s.t[i]; uint64_t expected; uint32_t alias;
        read_at(s.fd,rec,REC_SIZE,16+(uint64_t)i*REC_SIZE);
        memcpy(t->name,rec,NAME_SIZE);
        if(!memchr(t->name,0,NAME_SIZE) || !t->name[0]) die("invalid tensor name");
        t->rows=u32(rec+40); t->cols=u32(rec+44); t->dtype=u32(rec+48);
        t->off=u64(rec+56); t->bytes=u64(rec+64); t->expert=strstr(t->name,".e.")!=NULL;
        alias=u32(rec+52);
        if(alias>1 || (alias && version!=2) || !t->rows || !t->cols || t->cols>MAX_DIM || t->dtype>1) die("invalid tensor metadata: %s",t->name);
        expected=t->dtype ? (uint64_t)t->rows*((t->cols+31)/32)*36 : (uint64_t)t->rows*t->cols*4;
        if(t->bytes!=expected || (!alias && t->off<previous) || t->off>(uint64_t)st.st_size || t->bytes>(uint64_t)st.st_size-t->off) die("invalid tensor extent: %s",t->name);
        if(alias) {
            Tensor *e=&s.t[0];
            if(!i || strcmp(t->name,"lm_head") || strcmp(e->name,"embed") || t->off!=e->off || t->bytes!=e->bytes || t->rows!=e->rows || t->cols!=e->cols || t->dtype!=e->dtype) die("invalid tied embedding alias");
        } else previous=t->off+t->bytes;
        if(i && strcmp(s.t[i-1].name,t->name)>=0) die("tensor directory must have unique sorted names");
    }
#ifdef POSIX_FADV_RANDOM
    (void)posix_fadvise(s.fd,0,0,POSIX_FADV_RANDOM);
#endif
    return s;
}
static uint32_t tensor_id(Store *s,const char *name,uint32_t rows,uint32_t cols,uint32_t dtype) {
    uint32_t lo=0,hi=s->nt;
    while(lo<hi) {
        uint32_t mid=lo+(hi-lo)/2; int cmp=strcmp(s->t[mid].name,name);
        if(cmp<0) lo=mid+1; else hi=mid;
    }
    if(lo==s->nt || strcmp(s->t[lo].name,name)) die("missing tensor: %s",name);
    if(s->t[lo].rows!=rows || s->t[lo].cols!=cols || s->t[lo].dtype!=dtype) die("wrong tensor shape/type: %s",name);
    return lo;
}
static void cache_init(Store *s,Memory *mem) {
    size_t free_bytes=mem->limit-mem->used, max_slots, needed=0, i;
    /* Reserve the two allocator headers before deriving cache capacity. */
    if(free_bytes<2*sizeof(Allocation)) die("no room for weight cache");
    max_slots=(free_bytes-2*sizeof(Allocation))/(TILE+sizeof(Slot));
    for(i=0;i<s->nt;++i) {
        size_t tiles=(size_t)((s->t[i].bytes+TILE-1)/TILE);
        if(tiles>SIZE_MAX-needed) die("cache size overflow");
        needed+=tiles;
    }
    /* At most twice the actual tile count plus minimum associativity. */
    if(needed<(SIZE_MAX-8)/2 && max_slots>needed*2+8) max_slots=needed*2+8;
    s->nslots=max_slots/8*8;
    if(s->nslots<8) die("memory budget too small for minimum 8-tile cache (about 513 KiB)");
    s->core_slots=(s->nslots*3/4)/4*4;
    if(s->dense)s->core_slots=s->nslots;
    s->slots=(Slot *)alloc(mem,s->nslots,sizeof(Slot));
    s->data=(unsigned char *)alloc(mem,s->nslots,TILE);
}
static const unsigned char *tile_get(Store *s,Tensor *t,uint64_t rel) {
    uint64_t base=(rel/TILE)*TILE, key=t->off+base;
    int routed=t->expert && !s->dense;
    size_t start=routed?s->core_slots:0, count=routed?s->nslots-s->core_slots:s->core_slots;
    size_t set=start+(size_t)(mix(key)%(count/WAYS))*WAYS, victim=set, i;
    Slot *v;
    ++s->tick;
    for(i=set;i<set+WAYS;++i) {
        Slot *p=&s->slots[i];
        if(p->stamp && p->key==key) { ++s->hits; p->stamp=s->tick; return s->data+i*TILE+(size_t)(rel-base); }
        if(!p->stamp || p->stamp<s->slots[victim].stamp) victim=i;
    }
    v=&s->slots[victim]; v->key=key; v->len=(uint32_t)((t->bytes-base)<TILE?t->bytes-base:TILE);
    read_at(s->fd,s->data+victim*TILE,v->len,key); v->stamp=s->tick;
    ++s->misses; s->read_bytes+=v->len; if(t->expert) s->expert_reads+=v->len;
#ifdef POSIX_FADV_DONTNEED
    if(s->drop_cache) (void)posix_fadvise(s->fd,(off_t)key,(off_t)v->len,POSIX_FADV_DONTNEED);
#endif
    return s->data+victim*TILE+(size_t)(rel-base);
}
static float dot_scalar(const unsigned char *q,const float *x,uint32_t n) {
    float sum=0; uint32_t i;
    for(i=0;i<n;++i) { int v=q[i]<128?q[i]:(int)q[i]-256; sum+=(float)v*x[i]; }
    return sum;
}
#if SM_AVX2
__attribute__((target("avx2")))
static float dot_avx2(const unsigned char *q,const float *x,uint32_t n) {
    /* Vectorize conversion/multiplication, preserve scalar addition order.
     * No FMA or reassociation: budget and backend changes are bit-identical. */
    float sum=0, products[8]; uint32_t i,j;
    for(i=0;i+8<=n;i+=8) {
        __m128i b=_mm_loadl_epi64((const __m128i *)(const void *)(q+i));
        __m256 values=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
        _mm256_storeu_ps(products,_mm256_mul_ps(values,_mm256_loadu_ps(x+i)));
        for(j=0;j<8;++j) sum+=products[j];
    }
    for(;i<n;++i) { int v=q[i]<128?q[i]:(int)q[i]-256; sum+=(float)v*x[i]; }
    return sum;
}
#endif
typedef float (*Dot)(const unsigned char *,const float *,uint32_t);
static Dot dot=dot_scalar;
static const char *backend="scalar";
static void select_backend(int force_scalar) {
#if SM_AVX2
    if(!force_scalar && __builtin_cpu_supports("avx2")) {dot=dot_avx2;backend="avx2";}
#else
    (void)force_scalar;
#endif
}
static void matvec(Store *s,uint32_t id,const float *x,float *y) {
    Tensor *t=&s->t[id]; uint32_t r,b,nb=(t->cols+31)/32;
    for(r=0;r<t->rows;++r) {
        float sum=0;
        for(b=0;b<nb;) {
            uint64_t rel=((uint64_t)r*nb+b)*36;
            uint32_t count=(uint32_t)((TILE-rel%TILE)/36),j;
            const unsigned char *q=tile_get(s,t,rel);
            if(count>nb-b) count=nb-b;
            for(j=0;j<count;++j,++b,q+=36) {
                uint32_t n=t->cols-b*32; float scale=f32(q);
                if(n>32) n=32;
                if(!isfinite(scale) || scale<0) die("invalid Q8 scale in %s",t->name);
                sum+=scale*dot(q+4,x+b*32,n);
            }
        }
        if(!isfinite(sum)) die("non-finite matvec in %s",t->name);
        y[r]=sum;
    }
}
static void embedding(Store *s,uint32_t id,uint32_t token,float *x) {
    Tensor *t=&s->t[id]; uint32_t b,nb=(t->cols+31)/32;
    for(b=0;b<nb;++b) {
        const unsigned char *p=tile_get(s,t,((uint64_t)token*nb+b)*36);
        float scale=f32(p); uint32_t j,n=t->cols-b*32;
        if(n>32) n=32;
        if(!isfinite(scale) || scale<0) die("invalid embedding scale");
        for(j=0;j<n;++j) {
            int v=p[4+j]<128?p[4+j]:(int)p[4+j]-256;
            x[b*32+j]=scale*(float)v;
        }
    }
}
static void norm(Store *s,uint32_t id,const float *x,float *y,uint32_t n,float eps) {
    float sq=0,inv; uint32_t i; Tensor *t=&s->t[id];
    for(i=0;i<n;++i) sq+=x[i]*x[i];
    inv=1.0f/sqrtf(sq/(float)n+eps);
    if(!isfinite(inv)) die("non-finite RMSNorm");
    for(i=0;i<n;++i) {
        float w=f32(tile_get(s,t,(uint64_t)i*4));
        y[i]=x[i]*inv*w; if(!isfinite(y[i])) die("non-finite RMSNorm weight");
    }
}

typedef struct { uint32_t gate,up,down; } Expert;
typedef struct { uint32_t an,fn,q,k,v,o,router,qb,kb,vb; Expert *e,*shared; } Layer;
typedef struct {
    Config c; Layer *l; uint32_t embed,final,head,context,kv_dim;
    float *x,*xn,*q,*k,*v,*att,*out,*h,*gate,*logits,*scores,*route;
    float *keys,*values,*weights; uint32_t *selected;
    uint64_t expert_calls;
} Model;
static uint32_t named(Store *s,const char *name,uint32_t rows,uint32_t cols,uint32_t type) {
    return tensor_id(s,name,rows,cols,type);
}
static uint32_t layer_tensor(Store *s,uint32_t layer,const char *suffix,uint32_t rows,uint32_t cols,uint32_t type) {
    char name[NAME_SIZE]; int n=snprintf(name,sizeof(name),"l.%u.%s",layer,suffix);
    if(n<0 || (size_t)n>=sizeof(name)) die("tensor name too long");
    return named(s,name,rows,cols,type);
}
static Expert expert_ids(Store *s,uint32_t layer,const char *kind,uint32_t e,Config *c) {
    char suffix[32]; Expert x;
    (void)snprintf(suffix,sizeof(suffix),"%s.%u.gate",kind,e); x.gate=layer_tensor(s,layer,suffix,c->hidden,c->dim,1);
    (void)snprintf(suffix,sizeof(suffix),"%s.%u.up",kind,e); x.up=layer_tensor(s,layer,suffix,c->hidden,c->dim,1);
    (void)snprintf(suffix,sizeof(suffix),"%s.%u.down",kind,e); x.down=layer_tensor(s,layer,suffix,c->dim,c->hidden,1);
    return x;
}
static Model model_init(Config c,uint32_t context,Store *s,Memory *mem) {
    Model m; uint32_t i,e; uint64_t expected=3+(uint64_t)c.layers*(7+3*c.qkv_bias+3*(uint64_t)(c.experts+c.shared));
    memset(&m,0,sizeof(m)); m.c=c; m.context=context; m.kv_dim=c.kv_heads*(c.dim/c.heads);
    if(expected!=s->nt) die("tensor count does not match config");
    m.embed=named(s,"embed",c.vocab,c.dim,1); m.final=named(s,"final_norm",1,c.dim,0);
    m.head=named(s,"lm_head",c.vocab,c.dim,1);
    m.l=(Layer *)alloc(mem,c.layers,sizeof(Layer));
    for(i=0;i<c.layers;++i) {
        Layer *l=&m.l[i];
        l->an=layer_tensor(s,i,"attn_norm",1,c.dim,0); l->fn=layer_tensor(s,i,"ffn_norm",1,c.dim,0);
        l->q=layer_tensor(s,i,"q",c.dim,c.dim,1); l->k=layer_tensor(s,i,"k",m.kv_dim,c.dim,1);
        l->v=layer_tensor(s,i,"v",m.kv_dim,c.dim,1); l->o=layer_tensor(s,i,"o",c.dim,c.dim,1);
        l->router=layer_tensor(s,i,"router",c.experts,c.dim,1);
        if(c.qkv_bias) {
            l->qb=layer_tensor(s,i,"q_bias",1,c.dim,0);l->kb=layer_tensor(s,i,"k_bias",1,m.kv_dim,0);l->vb=layer_tensor(s,i,"v_bias",1,m.kv_dim,0);
        }
        l->e=(Expert *)alloc(mem,c.experts,sizeof(Expert)); l->shared=(Expert *)alloc(mem,c.shared,sizeof(Expert));
        for(e=0;e<c.experts;++e) l->e[e]=expert_ids(s,i,"e",e,&c);
        for(e=0;e<c.shared;++e) l->shared[e]=expert_ids(s,i,"s",e,&c);
    }
    m.x=(float *)alloc(mem,c.dim,sizeof(float)); m.xn=(float *)alloc(mem,c.dim,sizeof(float));
    m.q=(float *)alloc(mem,c.dim,sizeof(float)); m.k=(float *)alloc(mem,m.kv_dim,sizeof(float));
    m.v=(float *)alloc(mem,m.kv_dim,sizeof(float)); m.att=(float *)alloc(mem,c.dim,sizeof(float));
    m.out=(float *)alloc(mem,c.dim,sizeof(float)); m.h=(float *)alloc(mem,c.hidden,sizeof(float));
    m.gate=(float *)alloc(mem,c.hidden,sizeof(float)); m.logits=(float *)alloc(mem,c.vocab,sizeof(float));
    m.scores=(float *)alloc(mem,context,sizeof(float)); m.route=(float *)alloc(mem,c.experts,sizeof(float));
    m.weights=(float *)alloc(mem,c.top_k,sizeof(float)); m.selected=(uint32_t *)alloc(mem,c.top_k,sizeof(uint32_t));
    m.keys=(float *)alloc(mem,mul(mul(c.layers,context),m.kv_dim),sizeof(float));
    m.values=(float *)alloc(mem,mul(mul(c.layers,context),m.kv_dim),sizeof(float));
    return m;
}
static void rope(float *x,uint32_t heads,uint32_t hd,uint32_t pos,float theta) {
    uint32_t h,j;
    for(h=0;h<heads;++h) for(j=0;j<hd;j+=2) {
        float angle=(float)pos/powf(theta,(float)j/(float)hd),co=cosf(angle),si=sinf(angle);
        float a=x[h*hd+j],b=x[h*hd+j+1];
        x[h*hd+j]=a*co-b*si; x[h*hd+j+1]=a*si+b*co;
    }
}
static void attention(Model *m,uint32_t layer,uint32_t pos) {
    uint32_t hd=m->c.dim/m->c.heads,h,t,j;
    size_t layer_base=mul(mul(layer,m->context),m->kv_dim);
    float factor=1.0f/sqrtf((float)hd);
    for(h=0;h<m->c.heads;++h) {
        uint32_t kh=h/(m->c.heads/m->c.kv_heads); float high=-FLT_MAX,total=0;
        for(t=0;t<=pos;++t) {
            float score=0; const float *k=m->keys+layer_base+(size_t)t*m->kv_dim+kh*hd;
            for(j=0;j<hd;++j) score+=m->q[h*hd+j]*k[j];
            m->scores[t]=score*factor; if(m->scores[t]>high) high=m->scores[t];
        }
        for(t=0;t<=pos;++t) { m->scores[t]=expf(m->scores[t]-high);total+=m->scores[t]; }
        for(j=0;j<hd;++j) {
            float value=0;
            for(t=0;t<=pos;++t) value+=(m->scores[t]/total)*m->values[layer_base+(size_t)t*m->kv_dim+kh*hd+j];
            m->att[h*hd+j]=value;
        }
    }
}
static void expert_forward(Model *m,Store *s,Expert e) {
    uint32_t i;
    matvec(s,e.gate,m->xn,m->gate); matvec(s,e.up,m->xn,m->h);
    for(i=0;i<m->c.hidden;++i) m->h[i]*=m->gate[i]/(1.0f+expf(-m->gate[i]));
    matvec(s,e.down,m->h,m->out); ++m->expert_calls;
}
static void forward(Model *m,Store *s,uint32_t token,uint32_t pos,int with_logits) {
    Config *c=&m->c; uint32_t l,i,k; float total;
    if(token>=c->vocab || pos>=m->context) die("token/context out of range");
    embedding(s,m->embed,token,m->x);
    for(l=0;l<c->layers;++l) {
        Layer *p=&m->l[l]; size_t base=((size_t)l*m->context+pos)*m->kv_dim;
        norm(s,p->an,m->x,m->xn,c->dim,c->eps);
        matvec(s,p->q,m->xn,m->q); matvec(s,p->k,m->xn,m->k); matvec(s,p->v,m->xn,m->v);
        if(c->qkv_bias) {
            for(i=0;i<c->dim;++i) m->q[i]+=f32(tile_get(s,&s->t[p->qb],(uint64_t)i*4));
            for(i=0;i<m->kv_dim;++i) {
                m->k[i]+=f32(tile_get(s,&s->t[p->kb],(uint64_t)i*4));
                m->v[i]+=f32(tile_get(s,&s->t[p->vb],(uint64_t)i*4));
            }
        }
        rope(m->q,c->heads,c->dim/c->heads,pos,c->theta);
        rope(m->k,c->kv_heads,c->dim/c->heads,pos,c->theta);
        memcpy(m->keys+base,m->k,m->kv_dim*sizeof(float)); memcpy(m->values+base,m->v,m->kv_dim*sizeof(float));
        attention(m,l,pos); matvec(s,p->o,m->att,m->out);
        for(i=0;i<c->dim;++i) m->x[i]+=m->out[i];
        norm(s,p->fn,m->x,m->xn,c->dim,c->eps); matvec(s,p->router,m->xn,m->route);
        /* Exact top-k routing, lower expert index wins ties. Sigmoid weights
         * are renormalized only across selected experts. No experts dropped. */
        total=0;
        for(k=0;k<c->top_k;++k) {
            uint32_t best=0; float value=-FLT_MAX;
            for(i=0;i<c->experts;++i) if(m->route[i]>value) {value=m->route[i];best=i;}
            m->selected[k]=best; m->route[best]=-FLT_MAX;
            m->weights[k]=1.0f/(1.0f+expf(-value)); total+=m->weights[k];
        }
        if(!(total>0) || !isfinite(total)) die("router underflow/overflow");
        for(k=0;k<c->top_k;++k) {
            float weight=m->weights[k]/total;
            expert_forward(m,s,p->e[m->selected[k]]);
            for(i=0;i<c->dim;++i) m->x[i]+=weight*m->out[i];
        }
        for(k=0;k<c->shared;++k) {
            expert_forward(m,s,p->shared[k]); for(i=0;i<c->dim;++i) m->x[i]+=m->out[i];
        }
    }
    if(with_logits){norm(s,m->final,m->x,m->xn,c->dim,c->eps);matvec(s,m->head,m->xn,m->logits);}
}
static uint32_t rng_next(uint32_t *s) { uint32_t x=*s; x^=x<<13;x^=x>>17;x^=x<<5;*s=x;return x; }
static uint32_t sample(Model *m,float temperature,uint32_t *rng) {
    uint32_t i,best=0; float high=-FLT_MAX,total=0,target;
    /* BOS is never generated. EOS remains eligible. */
    for(i=0;i<m->c.vocab;++i) if(i!=256 && m->logits[i]>high) {high=m->logits[i];best=i;}
    if(temperature==0) return best;
    for(i=0;i<m->c.vocab;++i) {
        float p=i==256?0:expf((m->logits[i]-high)/temperature);
        m->logits[i]=p; total+=p;
    }
    target=((float)(rng_next(rng)>>8)/16777216.0f)*total;
    for(i=0;i<m->c.vocab;++i) {target-=m->logits[i];if(target<0) return i;}
    return best;
}
static double now(void) {
    struct timespec ts; if(clock_gettime(CLOCK_MONOTONIC,&ts)) die("clock_gettime failed");
    return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;
}
static int same_path(const char *a,const char *b) {
    struct stat x,y;
    if(!a || !b) return 0;
    if(!strcmp(a,b)) return 1;
    return !stat(a,&x) && !stat(b,&y) && x.st_dev==y.st_dev && x.st_ino==y.st_ino;
}
static void check_output_paths(const char **inputs,size_t ni,const char **outputs,size_t no) {
    size_t i,j;
    for(i=0;i<no;++i) if(outputs[i]) {
        for(j=0;j<ni;++j) if(same_path(outputs[i],inputs[j])) die("output path aliases an input: %s",outputs[i]);
        for(j=0;j<i;++j) if(same_path(outputs[i],outputs[j])) die("output paths alias each other: %s",outputs[i]);
    }
}
static uint64_t rss_peak(void) {
    struct rusage r; if(getrusage(RUSAGE_SELF,&r)) return 0;
#if defined(__APPLE__)
    return (uint64_t)r.ru_maxrss;
#else
    return (uint64_t)r.ru_maxrss*1024;
#endif
}
static void trace_logits(FILE *f,uint32_t pos,Model *m) {
    uint32_t i;
    if(!f) return;
    fprintf(f,"%u",pos);
    for(i=0;i<m->c.vocab;++i) fprintf(f," %a",(double)m->logits[i]);
    fputc('\n',f); if(ferror(f)) die("logit trace write failed");
}
static uint64_t hash_logits(uint64_t hash,Model *m) {
    uint32_t i,j;
    for(i=0;i<m->c.vocab;++i){uint32_t bits;memcpy(&bits,m->logits+i,4);for(j=0;j<4;++j){hash^=(bits>>(j*8))&255u;hash*=UINT64_C(1099511628211);}}
    return hash;
}
static void usage(FILE *out) {
    fputs("smolmoe --model FILE --config FILE --tokenizer FILE --prompt TEXT [options]\n"
          "  --prompt-file FILE   raw UTF-8/byte prompt instead of --prompt\n"
          "  --tokens N           max new tokens (default 64)\n"
          "  --context N          KV allocation (default min(config,2048))\n"
          "  --memory-mib N       tracked allocation ceiling (default 512)\n"
          "  --temperature X      0=greedy, positive=sample (default 0)\n"
          "  --seed N             nonzero xorshift32 seed (default 1)\n"
          "  --scalar             disable runtime AVX2 dispatch\n"
          "  --chat               Qwen2 ChatML single-turn prompt\n"
          "  --system TEXT        ChatML system message (default: Brazilian Portuguese)\n"
          "  --hash-logits        hash all raw logits into JSON stats for validation\n"
          "  --drop-cache         advise OS to release read checkpoint pages\n"
          "  --stats FILE         JSON statistics (also printed to stderr)\n"
          "  --trace FILE         exact hexadecimal logits before sampling\n"
          "  --token-ids FILE     prompt and generated token IDs, one per line\n"
          "  --tokenize-only      encode/decode prompt; no model required\n"
          "  --help               show this message\n"
          "Checkpoint architecture: smolmoe_v1; NOT a full Kimi K3 loader.\n",out);
}

#ifndef SM_NO_MAIN
int main(int argc,char **argv) {
    const char *model_path=NULL,*config_path=NULL,*tok_path=NULL,*prompt=NULL,*prompt_file=NULL;
    const char *stats_path=NULL,*trace_path=NULL,*ids_path=NULL;
    const char *system_prompt="Responda somente em português do Brasil. Não misture outros idiomas. Seja breve e direto.";
    uint32_t tokens=64,context=0,budget_mib=512,seed=1,pos,np,i,generated=0,processed=0;
    int scalar=0,drop_cache=0,tokenize_only=0,chat=0,logit_hashing=0,a; float temperature=0;
    uint64_t logits_digest=UINT64_C(14695981039346656037);
    Memory mem; Config c; Tokenizer tokenizer; Store store; Model model;
    unsigned char *input; uint32_t *ids,*decode_stack;
    size_t input_len; FILE *trace=NULL,*idout=NULL,*stats=NULL;
    double start=now(),loaded,prefill_end,ended,first=0;
    for(a=1;a<argc;++a) {
        const char *opt=argv[a],*value;
        if(!strcmp(opt,"--help")) {usage(stdout);return 0;}
        if(!strcmp(opt,"--scalar")) {scalar=1;continue;}
        if(!strcmp(opt,"--drop-cache")) {drop_cache=1;continue;}
        if(!strcmp(opt,"--tokenize-only")) {tokenize_only=1;continue;}
        if(!strcmp(opt,"--chat")) {chat=1;continue;}
        if(!strcmp(opt,"--hash-logits")) {logit_hashing=1;continue;}
        if(a+1>=argc) die("missing argument for %s",opt);
        value=argv[++a];
        if(!strcmp(opt,"--model")) model_path=value;
        else if(!strcmp(opt,"--config")) config_path=value;
        else if(!strcmp(opt,"--tokenizer")) tok_path=value;
        else if(!strcmp(opt,"--prompt")) prompt=value;
        else if(!strcmp(opt,"--prompt-file")) prompt_file=value;
        else if(!strcmp(opt,"--stats")) stats_path=value;
        else if(!strcmp(opt,"--trace")) trace_path=value;
        else if(!strcmp(opt,"--token-ids")) ids_path=value;
        else if(!strcmp(opt,"--system")) system_prompt=value;
        else if(!strcmp(opt,"--tokens")) tokens=number(value,MAX_CONTEXT,"tokens");
        else if(!strcmp(opt,"--context")) context=number(value,MAX_CONTEXT,"context");
        else if(!strcmp(opt,"--memory-mib")) budget_mib=number(value,1048576,"memory-mib");
        else if(!strcmp(opt,"--seed")) seed=number(value,UINT32_MAX,"seed");
        else if(!strcmp(opt,"--temperature")) {
            double temp=real_number(value,"temperature");
            if(temp<0 || temp>100 || (temp>0 && temp<1e-6)) die("temperature must be 0 or in [1e-6,100]");
            temperature=(float)temp;
        } else die("unknown option: %s",opt);
    }
    if(!config_path || !tok_path || (!model_path && !tokenize_only) || (!prompt && !prompt_file) || (prompt && prompt_file)) {usage(stderr);return 1;}
    {
        const char *inputs[]={model_path,config_path,tok_path,prompt_file};
        const char *outputs[]={stats_path,trace_path,ids_path};
        check_output_paths(inputs,4,outputs,3);
    }
    if(sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24 || sizeof(off_t)<8) die("requires IEEE binary32 and 64-bit POSIX file offsets");
    if(!seed || !budget_mib) die("seed and memory budget must be nonzero");
    memset(&mem,0,sizeof(mem));mem.limit=mul(budget_mib,1048576);
    c=config_read(config_path); if(!context) context=c.max_context<2048?c.max_context:2048;
    if(context>c.max_context) die("--context exceeds model context");
    tokenizer=tokenizer_read(tok_path,c.vocab,&mem);
    if(chat && !tokenizer.qwen)die("--chat requires a Qwen tokenizer");
    if(tokenizer.qwen)qwen_workspace(tokenizer.qwen,context,&mem);
    /* A prompt may be at most context-1 raw bytes, before BPE. This explicit
     * cap keeps tokenization workspace independent of checkpoint size. */
    input=(unsigned char *)alloc(&mem,context,1);
    if(prompt_file) {
        FILE *fp=fopen(prompt_file,"rb"); if(!fp) die("cannot open prompt file");
        input_len=fread(input,1,context,fp);
        if(ferror(fp)) die("prompt read error");
        if(input_len==context) die("prompt too long: at most context-1 raw bytes");
        fclose(fp);
    } else {
        input_len=strlen(prompt);if(input_len>=context) die("prompt too long: at most context-1 raw bytes");
        memcpy(input,prompt,input_len);
    }
    ids=(uint32_t *)alloc(&mem,tokenizer.qwen?tokenizer.qwen->capacity:(size_t)context+1,sizeof(uint32_t));
    decode_stack=(uint32_t *)alloc(&mem,tokenizer.count+2,sizeof(uint32_t));
    np=tokenizer.qwen?qwen_encode(tokenizer.qwen,input,(uint32_t)input_len,ids,chat,system_prompt):encode(&tokenizer,input,(uint32_t)input_len,ids);
    if(np>context) die("encoded prompt exceeds context");
    if(ids_path) {idout=fopen(ids_path,"w");if(!idout) die("cannot open token ID output");}
    if(idout) for(i=0;i<np;++i) fprintf(idout,"%u\n",ids[i]);
    if(tokenize_only) {
        for(i=tokenizer.qwen?0:1;i<np;++i) {
            if(i==1 && tokenizer.sentencepiece && ids[i]>=258 && tokenizer.lengths[ids[i]] && tokenizer.pieces[ids[i]][0]==' ') {
                if(fwrite(tokenizer.pieces[ids[i]]+1,1,tokenizer.lengths[ids[i]]-1,stdout)!=tokenizer.lengths[ids[i]]-1) die("output write failed");
            } else decode(&tokenizer,ids[i],decode_stack,tokenizer.count+2,stdout);
        }
        if(idout && fclose(idout)) die("token ID output close failed");
        if(fflush(stdout)) die("output flush failed");
        release(&mem);return 0;
    }
    if(tokens>context-np) die("prompt + --tokens exceeds --context; increase context or reduce tokens");
    select_backend(scalar); store=store_open(model_path,&mem);store.drop_cache=drop_cache;
    store.dense=c.experts==1;
    model=model_init(c,context,&store,&mem);cache_init(&store,&mem);
    if(trace_path) {trace=fopen(trace_path,"w");if(!trace) die("cannot open logit trace");}
    loaded=now();
    for(pos=0;pos<np;++pos) {forward(&model,&store,ids[pos],pos,trace || logit_hashing || pos+1==np);++processed;trace_logits(trace,pos,&model);if(logit_hashing)logits_digest=hash_logits(logits_digest,&model);}
    prefill_end=now();
    for(i=0;i<tokens;++i) {
        if(tokenizer.qwen){uint32_t j;for(j=0;j<c.vocab;++j)if(j!=257 && tokenizer.qwen->special[j])model.logits[j]=-FLT_MAX;}
        uint32_t token=sample(&model,temperature,&seed);
        if(idout) fprintf(idout,"%u\n",token);
        if(token==c.eos) break;
        decode(&tokenizer,token,decode_stack,tokenizer.count+2,stdout);
        if(fflush(stdout)) die("output flush failed");
        ++generated; if(!first) first=now();
        if(i+1<tokens) {forward(&model,&store,token,pos++,1);++processed;trace_logits(trace,pos-1,&model);if(logit_hashing)logits_digest=hash_logits(logits_digest,&model);}
    }
    ended=now();
    if(trace && fclose(trace)) die("trace close failed");
    if(idout && fclose(idout)) die("token ID output close failed");
    if(stats_path) {stats=fopen(stats_path,"w");if(!stats) die("cannot open stats output");}
    for(a=0;a<(stats?2:1);++a) {
        FILE *f=a?stats:stderr;
        double decode_sec=ended-prefill_end,decode_forwards=(double)(processed-np);
        fprintf(f,"{\"backend\":\"%s\",\"budget_bytes\":%zu,\"tracked_peak_bytes\":%zu,\"rss_peak_bytes\":%" PRIu64 ","
            "\"kv_bytes\":%zu,\"cache_bytes\":%zu,\"prompt_tokens\":%u,\"generated_tokens\":%u,\"processed_tokens\":%u,"
            "\"load_seconds\":%.9f,\"prefill_seconds\":%.9f,\"decode_seconds\":%.9f,\"ttft_seconds\":%.9f,"
            "\"prefill_tokens_per_second\":%.6f,\"decode_tokens_per_second\":%.6f,"
            "\"cache_hits\":%" PRIu64 ",\"cache_misses\":%" PRIu64 ",\"read_bytes\":%" PRIu64 ",\"expert_read_bytes\":%" PRIu64 ",\"expert_calls\":%" PRIu64 ",\"logits_hash\":\"%016" PRIx64 "\"}\n",
            backend,mem.limit,mem.peak,rss_peak(),mul(mul(mul(c.layers,context),model.kv_dim),8),
            mul(store.nslots,TILE+sizeof(Slot)),np,generated,processed,loaded-start,prefill_end-loaded,
            decode_sec,first?first-start:0,(double)np/(prefill_end-loaded),decode_sec>0?decode_forwards/decode_sec:0,
            store.hits,store.misses,store.read_bytes,store.expert_reads,model.expert_calls,logit_hashing?logits_digest:0);
        if(ferror(f)) die("statistics write failed");
    }
    if(stats && fclose(stats)) die("stats close failed");
    close(store.fd);release(&mem);return 0;
}
#endif
