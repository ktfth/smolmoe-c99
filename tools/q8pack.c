/* Row-bounded Safetensors payload converter. No ML/runtime dependencies. */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

static void fail(const char *s) {fprintf(stderr,"q8pack: %s\n",s);exit(1);}
static uint32_t le32(const unsigned char *b) {return b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24;}
static void put32(uint32_t u) {unsigned char b[4];b[0]=(unsigned char)u;b[1]=(unsigned char)(u>>8);b[2]=(unsigned char)(u>>16);b[3]=(unsigned char)(u>>24);if(fwrite(b,1,4,stdout)!=4)fail("write error");}
static void putfloat(float x) {uint32_t u;memcpy(&u,&x,4);put32(u);}
int main(int argc,char **argv) {
    int fd,bf16,outq;uint64_t base;uint32_t rows,cols,r,i,*order=NULL;
    unsigned char *raw;float *values;
    if(argc!=8 && argc!=9) fail("usage: q8pack SOURCE OFFSET ROWS COLS BF16 Q8 PERMUTE_HEAD_DIM [ROW_MAP]");
    fd=open(argv[1],O_RDONLY);base=strtoull(argv[2],NULL,10);rows=(uint32_t)strtoul(argv[3],NULL,10);cols=(uint32_t)strtoul(argv[4],NULL,10);
    bf16=atoi(argv[5]);outq=atoi(argv[6]);
    if(fd<0 || !rows || !cols || cols>65536 || sizeof(float)!=4)fail("invalid source/dimensions");
    raw=(unsigned char *)malloc((size_t)cols*4);values=(float *)malloc((size_t)cols*4);
    if(!raw || !values) fail("allocation failed");
    if(argc==9) {
        FILE *f=fopen(argv[8],"rb");order=(uint32_t *)malloc((size_t)rows*4);
        if(!f || !order)fail("row map open/allocation");
        for(r=0;r<rows;++r){unsigned char b[4];if(fread(b,1,4,f)!=4)fail("truncated row map");order[r]=le32(b);if(order[r]>=rows)fail("row map index");}
        fclose(f);
    }
    for(r=0;r<rows;++r) {
        uint32_t source_row=order?order[r]:r,hd=(uint32_t)strtoul(argv[7],NULL,10);
        size_t bytes=(size_t)cols*(bf16?2:4),done=0;
        if(hd) {uint32_t local=r%hd;if(hd%2 || rows%hd)fail("head permutation");source_row=r-local+(local%2)*(hd/2)+local/2;}
        while(done<bytes) {ssize_t n=pread(fd,raw+done,bytes-done,(off_t)(base+(uint64_t)source_row*bytes+done));if(n<=0)fail("source read error");done+=(size_t)n;}
        for(i=0;i<cols;++i) {uint32_t u=bf16?((uint32_t)raw[i*2]|(uint32_t)raw[i*2+1]<<8)<<16:le32(raw+i*4);memcpy(values+i,&u,4);if(!isfinite(values[i]))fail("non-finite source");}
        if(!outq) {for(i=0;i<cols;++i)putfloat(values[i]);continue;}
        for(i=0;i<cols;i+=32) {
            uint32_t j,n=cols-i;float high=0,scale;unsigned char q[32];memset(q,0,sizeof(q));if(n>32)n=32;
            for(j=0;j<n;++j)if(fabsf(values[i+j])>high)high=fabsf(values[i+j]);
            scale=high/127.0f;putfloat(scale);
            for(j=0;j<n;++j) {int v=scale?(int)nearbyint((double)values[i+j]/(double)scale):0;if(v>127)v=127;if(v< -127)v= -127;q[j]=(unsigned char)v;}
            if(fwrite(q,1,32,stdout)!=32)fail("write error");
        }
    }
    if(fflush(stdout))fail("flush error");
    free(raw);free(values);free(order);close(fd);return 0;
}
