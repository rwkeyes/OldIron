// OldIron W1c: does the W4A8 GEMV keep its per-core rate when the weights come from DRAM and
// 16 cores are all asking? Projects the engine ceiling from measured aggregate GMAC/s.
//
// gcc -O3 -march=sandybridge -mavx -msse4.2 -fopenmp -o oiw4mt oiw4_gemv_mt.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define QK 32
#define QB 16

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

static inline __m128 block_acc(const uint8_t *qs, const int8_t *a, float scale, int32_t corr, __m128 acc){
    __m128i p  = _mm_loadu_si128((const __m128i*)qs);
    __m128i lo = _mm_and_si128(p, _mm_set1_epi8(0x0f));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(p,4), _mm_set1_epi8(0x0f));
    const __m128i ones=_mm_set1_epi16(1);
    __m128i s=_mm_add_epi32(
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(lo,hi),_mm_loadu_si128((const __m128i*)a)),ones),
        _mm_madd_epi16(_mm_maddubs_epi16(_mm_unpackhi_epi8(lo,hi),_mm_loadu_si128((const __m128i*)(a+16))),ones));
    s=_mm_sub_epi32(s,_mm_cvtsi32_si128(corr));
    return _mm_add_ps(acc,_mm_mul_ps(_mm_cvtepi32_ps(s),_mm_set1_ps(scale)));
}
static inline float hsum(__m128 v){ v=_mm_add_ps(v,_mm_movehl_ps(v,v)); v=_mm_add_ss(v,_mm_shuffle_ps(v,v,1)); return _mm_cvtss_f32(v); }

static float row_dot(const uint8_t *qs, const float *d, const int8_t *a, const int32_t *bsum, int nb){
    __m128 a0=_mm_setzero_ps(),a1=a0,a2=a0,a3=a0; int b=0;
    for(;b+4<=nb;b+=4){
        a0=block_acc(qs+b*QB,     a+b*QK,     d[b  ],8*bsum[b  ],a0);
        a1=block_acc(qs+(b+1)*QB, a+(b+1)*QK, d[b+1],8*bsum[b+1],a1);
        a2=block_acc(qs+(b+2)*QB, a+(b+2)*QK, d[b+2],8*bsum[b+2],a2);
        a3=block_acc(qs+(b+3)*QB, a+(b+3)*QK, d[b+3],8*bsum[b+3],a3);
    }
    for(;b<nb;b++) a0=block_acc(qs+b*QB,a+b*QK,d[b],8*bsum[b],a0);
    return hsum(_mm_add_ps(_mm_add_ps(a0,a1),_mm_add_ps(a2,a3)));
}

int main(int argc,char**argv){
    int K=2048, N=65536, nb=K/QK;
    size_t wbytes=(size_t)N*nb*QB, sbytes=(size_t)N*nb*sizeof(float);
    uint8_t *qs=aligned_alloc(64,wbytes);
    float   *d =aligned_alloc(64,sbytes);
    printf("matrix: %d rows x %d = %.0f MB packed + %.0f MB scales (does not fit L3)\n",
           N,K,wbytes/1048576.0,sbytes/1048576.0);
    srand(7);
    for(size_t i=0;i<wbytes;i++) qs[i]=(uint8_t)(rand()&0xff);
    for(size_t i=0;i<(size_t)N*nb;i++) d[i]=0.0005f+0.0009f*((rand()%100)/100.f);
    int8_t *a=aligned_alloc(64,K); for(int i=0;i<K;i++) a[i]=(int8_t)((i*37)%121-60);
    int32_t *bsum=aligned_alloc(64,nb*sizeof(int32_t));
    for(int b=0;b<nb;b++){ int32_t s=0; for(int j=0;j<QK;j++) s+=a[b*QK+j]; bsum[b]=s; }

    for(int th=1;th<=16;th*=2){
        double best=1e9; float sink=0;
        for(int rep=0;rep<3;rep++){
            double t0=now();
            #pragma omp parallel for num_threads(th) schedule(static) reduction(+:sink)
            for(int r=0;r<N;r++) sink+=row_dot(qs+(size_t)r*nb*QB, d+(size_t)r*nb, a, bsum, nb);
            double dt=now()-t0; if(dt<best) best=dt;
        }
        double gmac=(double)N*K/best/1e9;
        double gbs=(wbytes+sbytes)/best/1e9;                 /* bytes of weights touched */
        printf("  %2d threads: %7.2f GMAC/s aggregate (%5.2f per core)  %6.2f GB/s of weights  [%g]\n",
               th,gmac,gmac/th,gbs,(double)sink);
    }
    printf("\nprojection for the curated model (3B active params, 4.25 bits => 1.60 GB/token):\n");
    printf("  compute ceiling = GMAC/s / 3.0 GMAC per token ; memory ceiling = 74.9 GB/s / 1.60 GB per token = %.1f t/s\n",
           74.9/1.60);
    return 0;
}
