/* Qwen2 byte-level BPE + Unicode NFC + exact pre-tokenization rules.
 * Included after allocator, endian readers and hash mixer in smolmoe.c. */
#include "unicode.h"
typedef struct {uint32_t a,b,out;} QwenMerge;
struct QwenTokenizer {
    uint32_t vocab,nmerge,im_start,hash_size,capacity;
    unsigned char **piece,*special,*normalized;
    uint32_t *length,*hash,*cp,*offset,*symbols;
    QwenMerge *merge;
};
static QwenTokenizer *qwen_read(const char *path,uint32_t vocab,Memory *mem) {
    FILE *f=fopen(path,"rb");unsigned char h[24],r[12],*data;uint32_t i,total,used=0;
    QwenTokenizer *q=(QwenTokenizer *)alloc(mem,1,sizeof(*q));
    if(!f || fread(h,1,24,f)!=24 || memcmp(h,"SMQBP1\0\1",8) || u32(h+8)!=vocab)die("invalid Qwen tokenizer header");
    q->vocab=vocab;q->nmerge=u32(h+12);total=u32(h+16);q->im_start=u32(h+20);
    if(q->nmerge>MAX_MERGES || total>67108864 || q->im_start>=vocab)die("invalid Qwen tokenizer dimensions");
    q->piece=(unsigned char **)alloc(mem,vocab,sizeof(*q->piece));q->length=(uint32_t *)alloc(mem,vocab,4);q->special=(unsigned char *)alloc(mem,vocab,1);
    data=(unsigned char *)alloc(mem,total,1);q->merge=(QwenMerge *)alloc(mem,q->nmerge,sizeof(*q->merge));
    q->hash_size=8;while(q->hash_size<q->nmerge*2)q->hash_size*=2;
    q->hash=(uint32_t *)alloc(mem,q->hash_size,4);
    for(i=0;i<vocab;++i){uint32_t n,flag;if(fread(r,1,8,f)!=8)die("truncated Qwen token");n=u32(r);flag=u32(r+4);
        if(!n || n>65536 || n>total-used || flag>1)die("invalid Qwen piece");
        q->piece[i]=data+used;q->length[i]=n;q->special[i]=(unsigned char)flag;used+=n;
        if(fread(q->piece[i],1,n,f)!=n)die("truncated Qwen piece bytes");
        if(i<256 && (n!=1 || q->piece[i][0]!=i || flag))die("invalid Qwen byte-token permutation");
    }
    if(used!=total)die("Qwen piece byte count mismatch");
    for(i=0;i<q->nmerge;++i){uint32_t slot;QwenMerge *m=&q->merge[i];if(fread(r,1,12,f)!=12)die("truncated Qwen merges");m->a=u32(r);m->b=u32(r+4);m->out=u32(r+8);
        if(m->a>=vocab || m->b>=vocab || m->out>=vocab || q->special[m->a] || q->special[m->b] || q->special[m->out] ||
           q->length[m->a]+q->length[m->b]!=q->length[m->out] || memcmp(q->piece[m->out],q->piece[m->a],q->length[m->a]) || memcmp(q->piece[m->out]+q->length[m->a],q->piece[m->b],q->length[m->b]))die("invalid Qwen merge");
        slot=(uint32_t)mix(((uint64_t)m->a<<32)|m->b)&(q->hash_size-1);
        while(q->hash[slot]){QwenMerge old=q->merge[q->hash[slot]-1];if(old.a==m->a && old.b==m->b)die("duplicate Qwen merge");slot=(slot+1)&(q->hash_size-1);}
        q->hash[slot]=i+1;
    }
    if(fgetc(f)!=EOF || ferror(f))die("extra Qwen tokenizer payload");
    fclose(f);return q;
}
static void qwen_workspace(QwenTokenizer *q,uint32_t context,Memory *mem) {
    q->capacity=context*4+512;
    q->cp=(uint32_t *)alloc(mem,q->capacity,4);q->offset=(uint32_t *)alloc(mem,q->capacity+1,4);
    q->symbols=(uint32_t *)alloc(mem,q->capacity,4);q->normalized=(unsigned char *)alloc(mem,q->capacity*4,1);
}
static uint32_t qwen_rank(const QwenTokenizer *q,uint32_t a,uint32_t b) {
    uint32_t slot=(uint32_t)mix(((uint64_t)a<<32)|b)&(q->hash_size-1);
    while(q->hash[slot]){uint32_t i=q->hash[slot]-1;if(q->merge[i].a==a && q->merge[i].b==b)return i;slot=(slot+1)&(q->hash_size-1);}
    return UINT32_MAX;
}
static int unicode_space(uint32_t c) {return (c>=9 && c<=13) || c==32 || c==0x85 || c==0xa0 || c==0x1680 || (c>=0x2000 && c<=0x200a) || c==0x2028 || c==0x2029 || c==0x202f || c==0x205f || c==0x3000;}
static uint32_t ascii_lower(uint32_t c) {return c>='A' && c<='Z'?c+32:c;}
static int punctuation(uint32_t c) {return !unicode_space(c) && !unicode_kind(c);}
static uint32_t qwen_boundary(const uint32_t *c,uint32_t n,uint32_t i) {
    uint32_t j=i,k;
    if(c[i]=='\'' && i+1<n){uint32_t a=ascii_lower(c[i+1]);
        if(a=='s' || a=='t' || a=='m' || a=='d')return i+2;
        if(i+2<n){uint32_t b=ascii_lower(c[i+2]);if((a=='r' && b=='e') || (a=='v' && b=='e') || (a=='l' && b=='l'))return i+3;}
    }
    if(unicode_kind(c[j])!=1 && c[j]!='\r' && c[j]!='\n' && !unicode_kind(c[j]) && j+1<n && unicode_kind(c[j+1])==1)++j;
    if(unicode_kind(c[j])==1){do{++j;}while(j<n && unicode_kind(c[j])==1);return j;}
    if(unicode_kind(c[i])==2)return i+1;
    j=i;if(c[j]==32 && j+1<n && punctuation(c[j+1]))++j;
    if(punctuation(c[j])){do{++j;}while(j<n && punctuation(c[j]));while(j<n && (c[j]=='\r' || c[j]=='\n'))++j;return j;}
    j=i;while(j<n && unicode_space(c[j]))++j;
    if(j>i){uint32_t last=UINT32_MAX;for(k=i;k<j;++k)if(c[k]=='\r' || c[k]=='\n')last=k;
        if(last!=UINT32_MAX)return last+1;
        if(j==n)return j;
        if(j-i>1)return j-1;
        return j;
    }
    die("Qwen pre-tokenizer made no progress");return i+1;
}
static void qwen_push(uint32_t *ids,uint32_t *n,uint32_t cap,uint32_t id) {if(*n>=cap)die("Qwen prompt workspace exceeded");ids[(*n)++]=id;}
static void qwen_text(QwenTokenizer *q,const unsigned char *s,uint32_t bytes,uint32_t *ids,uint32_t *n) {
    uint32_t nc=normalize_nfc(s,bytes,q->cp,q->capacity),i,j,total=0;
    for(i=0;i<nc;++i){q->offset[i]=total;total+=write_utf8(q->cp[i],q->normalized+total);}q->offset[nc]=total;
    for(i=0;i<nc;i=j){uint32_t a,count,k;
        j=qwen_boundary(q->cp,nc,i);a=q->offset[i];count=q->offset[j]-a;
        if(count>q->capacity)die("Qwen byte workspace exceeded");
        for(k=0;k<count;++k)q->symbols[k]=q->normalized[a+k];
        for(;;){uint32_t best=UINT32_MAX,pos=0;
            for(k=0;k+1<count;++k){uint32_t r=qwen_rank(q,q->symbols[k],q->symbols[k+1]);if(r<best){best=r;pos=k;}}
            if(best==UINT32_MAX)break;
            q->symbols[pos]=q->merge[best].out;memmove(q->symbols+pos+1,q->symbols+pos+2,(count-pos-2)*4);--count;
        }
        for(k=0;k<count;++k)qwen_push(ids,n,q->capacity,q->symbols[k]);
    }
}
static uint32_t qwen_encode(QwenTokenizer *q,const unsigned char *s,uint32_t bytes,uint32_t *ids,int chat,const char *system) {
    uint32_t n=0;
    if(chat){qwen_push(ids,&n,q->capacity,q->im_start);qwen_text(q,(const unsigned char *)"system\n",7,ids,&n);qwen_text(q,(const unsigned char *)system,(uint32_t)strlen(system),ids,&n);qwen_push(ids,&n,q->capacity,257);qwen_text(q,(const unsigned char *)"\n",1,ids,&n);qwen_push(ids,&n,q->capacity,q->im_start);qwen_text(q,(const unsigned char *)"user\n",5,ids,&n);}
    qwen_text(q,s,bytes,ids,&n);
    if(chat){qwen_push(ids,&n,q->capacity,257);qwen_text(q,(const unsigned char *)"\n",1,ids,&n);qwen_push(ids,&n,q->capacity,q->im_start);qwen_text(q,(const unsigned char *)"assistant\n",10,ids,&n);}
    if(!n)qwen_push(ids,&n,q->capacity,256);
    return n;
}
