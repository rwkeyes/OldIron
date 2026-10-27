// OldIron W4 prototype: int8 GEMM for AVX1 - weight reuse across tokens.
//
// The GEMV kernel (oiw4_gemv_v1.c) re-reads and re-unpacks the weight block for every token. Prefill feeds
// many tokens through the same weights, so the win is to unpack a weight block ONCE and use it against NR
// activation vectors. That is the whole idea; this measures how much it is actually worth on Sandy Bridge,
// where only 16 xmm registers exist and NR accumulators compete with the weight/activation temporaries.
//
// Layouts match ggml exactly:
//   block_q4_0: fp16 d; uint8 qs[16]   -> byte j = element j in the low nibble, element j+16 in the high
//   block_q8_0: fp16 d; int8  qs[32]
// (fp32 scales are used here; the real kernel converts fp16 per block, worth a few percent.)
//
//   gcc -O3 -march=sandybridge -mavx -msse4.2 -DNR=4 -o oiw4_gemm oiw4_gemm.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

#ifndef NR
#define NR 4            /* tokens batched per weight-block unpack */
#endif
#define QK 32
#define QB 16

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }

typedef struct { int K, N; uint8_t *qs; float *d; } wmat;      /* N rows x K */

static void init_w(wmat *w, int N, int K, unsigned seed){
    w->N=N; w->K=K; int nb=K/QK;
    w->qs = aligned_alloc(64,(size_t)N*nb*QB);
    w->d  = aligned_alloc(64,(size_t)N*nb*sizeof(float));
    srand(seed);
    for(size_t i=0;i<(size_t)N*nb*QB;i++) w->qs[i]=(uint8_t)(rand()&0xff);
    for(size_t i=0;i<(size_t)N*nb;i++)    w->d[i]=0.0005f+0.0009f*((rand()%100)/100.f);
}

/* scalar reference: one row against one activation vector */
static float ref_row(const wmat *w, const float *a, int row, int K){
    const uint8_t *qs=w->qs+(size_t)row*(K/QK)*QB;
    const float   *d =w->d +(size_t)row*(K/QK);
    float acc=0.f;
    for(int b=0;b<K/QK;b++){
        int s=0;
        for(int j=0;j<QK;j++){
            uint8_t by=qs[b*QB+j/2];
            int n=(j&1)?(by>>4):(by&0x0f);
            s+=(n-8);
        }
        /* activation block is a float vector here; weights carry the quantisation */
        float f=0.f;
        for(int j=0;j<QK;j++){
            uint8_t by=qs[b*QB+j/2];
            int n=(j&1)?(by>>4):(by&0x0f);
            f += (float)(n-8)*a[b*QK+j];
        }
        acc += f*d[b];
        (void)s;
    }
    return acc;
}

/* NR tokens at once: unpack the weight block once, use it NR times.
 * Activations are quantised to int8 with a per-token per-block scale (q8_0 shape). */
typedef struct { int8_t *q; float *d; int nb; } act;   /* nb blocks of 32 */

static void init_act(act *a, int K, unsigned seed){
    a->nb=K/QK;
    a->q=aligned_alloc(64,(size_t)a->nb*QK);
    a->d=aligned_alloc(64,(size_t)a->nb*sizeof(float));
    srand(seed);
    for(int i=0;i<a->nb*QK;i++) a->q[i]=(int8_t)((rand()%121)-60);
    for(int i=0;i<a->nb;i++)    a->d[i]=0.001f+0.002f*((rand()%100)/100.f);
}

/* Per-token per-block sums of the int8 activations. The q4_0 zero-point correction is 8*sum(a_block),
 * which depends ONLY on the activation vector - so it can be hoisted out of the row loop entirely and
 * computed once per (token, block). That removes ~4 of the ~16 instructions in the inner block loop. */
static void act_sums(const act *A, int nb, int32_t *out){
    for(int b=0;b<nb;b++){
        int32_t s=0;
        for(int j=0;j<QK;j++) s+=A->q[b*QK+j];
        out[b]=8*s;
    }
}

static void gemm_rows(const wmat *w, const act *A, int rows, float *out){
    int K=w->K, nb=K/QK;
#if defined(PRECOMP_SUMS)
    int32_t corr[NR][64];      /* nb <= 64 for K <= 2048 */
    for(int t=0;t<NR;t++) act_sums(&A[t], nb, corr[t]);
#endif
    for(int r=0;r<rows;r++){
        const uint8_t *qs=w->qs+(size_t)r*nb*QB;
        const float   *d =w->d +(size_t)r*nb;
        __m128 acc[NR];
        for(int t=0;t<NR;t++) acc[t]=_mm_setzero_ps();
        const __m128i m4=_mm_set1_epi8(0x0f), ones=_mm_set1_epi16(1), ones8=_mm_set1_epi8(1);
        for(int b=0;b<nb;b++){
            /* unpack ONCE ... */
            __m128i p =_mm_loadu_si128((const __m128i*)(qs+b*QB));
            __m128i lo=_mm_and_si128(p,m4);
            __m128i hi=_mm_and_si128(_mm_srli_epi16(p,4),m4);
            /* ... then reuse against every token */
            for(int t=0;t<NR;t++){
                const int8_t *aq=A[t].q+b*QK;
                __m128i a0=_mm_loadu_si128((const __m128i*)aq);
                __m128i a1=_mm_loadu_si128((const __m128i*)(aq+16));
                __m128i s=_mm_add_epi32(
                    _mm_madd_epi16(_mm_maddubs_epi16(lo,a0),ones),
                    _mm_madd_epi16(_mm_maddubs_epi16(hi,a1),ones));
#if defined(PRECOMP_SUMS)
                s=_mm_sub_epi32(s,_mm_cvtsi32_si128(corr[t][b]));   /* 1 instr instead of 7 */
#else
                __m128i sy=_mm_add_epi32(
                    _mm_madd_epi16(_mm_maddubs_epi16(ones8,a0),ones),
                    _mm_madd_epi16(_mm_maddubs_epi16(ones8,a1),ones));
                s=_mm_sub_epi32(s,_mm_slli_epi32(sy,3));
#endif
                acc[t]=_mm_add_ps(acc[t],_mm_mul_ps(_mm_cvtepi32_ps(s),
                          _mm_set1_ps(d[b]*A[t].d[b])));
            }
        }
        for(int t=0;t<NR;t++){
            __m128 v=acc[t];
            v=_mm_add_ps(v,_mm_movehl_ps(v,v)); v=_mm_add_ss(v,_mm_shuffle_ps(v,v,1));
            out[r*NR+t]=_mm_cvtss_f32(v);
        }
    }
}

/* float reference for verification (weights dequantised, activations float) */
/* A is ONE token's activations here, not an array - indexing A[t] here was a stack overflow for NR>1 */
static float ref_full(const wmat *w, const act *A, int row, int t){
    (void)t;
    int K=w->K, nb=K/QK;
    const uint8_t *qs=w->qs+(size_t)row*nb*QB;
    const float   *d =w->d +(size_t)row*nb;
    float acc=0.f;
    for(int b=0;b<nb;b++){
        float f=0.f;
        for(int j=0;j<QK;j++){
            /* ggml q4_0 layout: byte j holds element j in the LOW nibble and element j+16 in the HIGH,
               NOT element j / j+1 interleaved. Getting this wrong makes the reference disagree with the
               kernel by construction (this cost a debug cycle). */
            uint8_t by=qs[b*QB + (j & 15)];
            int n=(j<16)?(by&0x0f):(by>>4);
            f+=(float)(n-8)*((float)A->q[b*QK+j]*A->d[b]);
        }
        acc+=f*d[b];
    }
    return acc;
}

int main(void){
    int K=2048, rows=512;
    wmat w; init_w(&w,rows,K,7);
    act A[NR]; for(int t=0;t<NR;t++) init_act(&A[t],K,100+t);
    float *out=malloc(sizeof(float)*rows*NR);

    gemm_rows(&w,A,rows,out);
    int bad=0;
    for(int r=0;r<rows && bad<4;r++) for(int t=0;t<NR;t++){
        float ref=ref_full(&w,&A[t],r,t);
        float tol=2e-3f*(1.f+(ref<0?-ref:ref));
        if(ref-out[r*NR+t]>tol||out[r*NR+t]-ref>tol){ printf("  MISMATCH r=%d t=%d ref=%f got=%f\n",r,t,ref,out[r*NR+t]); bad++; }
    }

    double best=1e9;
    for(int rep=0;rep<5;rep++){
        double t0=now(); gemm_rows(&w,A,rows,out); double dt=now()-t0;
        if(dt<best) best=dt;
    }
    double macs=(double)rows*NR*K;
    printf("NR=%d  %6.2f GMAC/s  (%d rows x %d k x %d tokens)  verify: %s\n",
           NR, macs/best/1e9, rows, K, NR, bad?"FAIL":"match");
    return 0;
}
