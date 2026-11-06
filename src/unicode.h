/* UTF-8 validation and canonical NFC, using generated Unicode 15 tables. */
#include "unicode_data.h"
static uint32_t unicode_lookup(const UnicodeRange *a,size_t n,uint32_t cp) {
    size_t lo=0,hi=n;
    while(lo<hi) {size_t m=lo+(hi-lo)/2;if(a[m].b<cp)lo=m+1;else hi=m;}
    return lo<n && a[lo].a<=cp?a[lo].value:0;
}
static uint32_t unicode_kind(uint32_t cp) {return unicode_lookup(unicode_class,sizeof(unicode_class)/sizeof(*unicode_class),cp);}
static uint32_t combining_class(uint32_t cp) {
    size_t lo=0,hi=sizeof(unicode_ccc)/sizeof(*unicode_ccc);
    while(lo<hi){size_t m=lo+(hi-lo)/2;if(unicode_ccc[m].cp<cp)lo=m+1;else hi=m;}
    return lo<sizeof(unicode_ccc)/sizeof(*unicode_ccc) && unicode_ccc[lo].cp==cp?unicode_ccc[lo].value:0;
}
static uint32_t read_utf8(const unsigned char *s,uint32_t bytes,uint32_t *pos) {
    uint32_t c,n,k,minimum;
    if(*pos>=bytes)die("invalid UTF-8 position");
    c=s[(*pos)++];if(c<128)return c;
    if(c>=194 && c<=223){n=1;c&=31;minimum=128;}
    else if(c>=224 && c<=239){n=2;c&=15;minimum=2048;}
    else if(c>=240 && c<=244){n=3;c&=7;minimum=65536;}
    else {die("Qwen tokenizer requires valid UTF-8");return 0;}
    if(bytes-*pos<n)die("truncated UTF-8");
    for(k=0;k<n;++k){uint32_t b=s[(*pos)++];if((b&192)!=128)die("invalid UTF-8 continuation");c=(c<<6)|(b&63);}
    if(c<minimum || c>0x10ffff || (c>=0xd800 && c<=0xdfff))die("invalid UTF-8 code point");
    return c;
}
static uint32_t write_utf8(uint32_t cp,unsigned char *p) {
    if(cp<128){p[0]=(unsigned char)cp;return 1;}
    if(cp<2048){p[0]=(unsigned char)(192|(cp>>6));p[1]=(unsigned char)(128|(cp&63));return 2;}
    if(cp<65536){p[0]=(unsigned char)(224|(cp>>12));p[1]=(unsigned char)(128|((cp>>6)&63));p[2]=(unsigned char)(128|(cp&63));return 3;}
    p[0]=(unsigned char)(240|(cp>>18));p[1]=(unsigned char)(128|((cp>>12)&63));p[2]=(unsigned char)(128|((cp>>6)&63));p[3]=(unsigned char)(128|(cp&63));return 4;
}
static void canonical_decompose(uint32_t cp,uint32_t *out,uint32_t *n,uint32_t cap) {
    size_t lo=0,hi=sizeof(unicode_decomp)/sizeof(*unicode_decomp);
    if(cp>=0xac00 && cp<0xd7a4){uint32_t s=cp-0xac00;canonical_decompose(0x1100+s/588,out,n,cap);canonical_decompose(0x1161+(s%588)/28,out,n,cap);if(s%28)canonical_decompose(0x11a7+s%28,out,n,cap);return;}
    while(lo<hi){size_t m=lo+(hi-lo)/2;if(unicode_decomp[m].a<cp)lo=m+1;else hi=m;}
    if(lo<sizeof(unicode_decomp)/sizeof(*unicode_decomp) && unicode_decomp[lo].a==cp){canonical_decompose(unicode_decomp[lo].b,out,n,cap);if(unicode_decomp[lo].value)canonical_decompose(unicode_decomp[lo].value,out,n,cap);return;}
    if(*n>=cap)die("normalization workspace exceeded");
    out[(*n)++]=cp;
    {uint32_t j=*n-1,cc=combining_class(cp);while(cc && j && combining_class(out[j-1])>cc){out[j]=out[j-1];--j;}out[j]=cp;}
}
static uint32_t canonical_compose(uint32_t a,uint32_t b) {
    size_t lo=0,hi=sizeof(unicode_compose)/sizeof(*unicode_compose);
    if(a>=0x1100 && a<=0x1112 && b>=0x1161 && b<=0x1175)return 0xac00+((a-0x1100)*21+b-0x1161)*28;
    if(a>=0xac00 && a<0xd7a4 && (a-0xac00)%28==0 && b>0x11a7 && b<=0x11c2)return a+b-0x11a7;
    while(lo<hi){size_t m=lo+(hi-lo)/2;const UnicodeRange *p=&unicode_compose[m];if(p->a<a || (p->a==a && p->b<b))lo=m+1;else hi=m;}
    return lo<sizeof(unicode_compose)/sizeof(*unicode_compose) && unicode_compose[lo].a==a && unicode_compose[lo].b==b?unicode_compose[lo].value:0;
}
static uint32_t normalize_nfc(const unsigned char *s,uint32_t bytes,uint32_t *cp,uint32_t cap) {
    uint32_t pos=0,n=0,i,out=0,starter=0,last_cc=0;
    while(pos<bytes)canonical_decompose(read_utf8(s,bytes,&pos),cp,&n,cap);
    for(i=0;i<n;++i){uint32_t cc=combining_class(cp[i]),composed=out?canonical_compose(cp[starter],cp[i]):0;
        if(composed && (last_cc<cc || !last_cc)){cp[starter]=composed;continue;}
        if(!cc)starter=out;
        cp[out++]=cp[i];last_cc=cc;
    }
    return out;
}
