/* Include the implementation to test the exact production kernel. */
#define main smolmoe_cli_main
#include "../src/smolmoe.c"
#undef main
#include <assert.h>

int main(void) {
    unsigned char q[32]; float x[32],a,b; uint32_t i,n;
    for(i=0;i<32;++i) {q[i]=(unsigned char)(i*11);x[i]=(float)((int)i-17)/13.0f;}
    select_backend(0);
    for(n=1;n<=32;++n) {
        a=dot_scalar(q,x,n);b=dot(q,x,n);
        assert(memcmp(&a,&b,sizeof(a))==0);
    }
    assert(u32((const unsigned char *)"\x12\x34\x56\x78")==UINT32_C(0x78563412));
    { unsigned char f[]={0,0,128,63}; assert(f32(f)==1.0f); }
    { uint32_t state=1;assert(rng_next(&state)==270369); }
    printf("kernel checks passed (%s), exact dot products for all 1..32 tails\n",backend);
    return 0;
}
