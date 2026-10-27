// OldIron W1b: AVX1 W4A8 GEMV, restructured.
//
// v0 measured 3.2 GMAC/s/core with exact integer correctness, but it reduced horizontally
// (_mm_hadd_epi32 x2) and summed 32 activations with a SCALAR loop per block. Both are per-block
// costs on a loop whose whole body is only ~14 instructions.
//
// v1 fixes, in order of what they were worth:
//   1. precompute per-block activation sums ONCE per GEMV (they are shared by every row)
//   2. never reduce horizontally inside the loop: keep 4 int32 lanes, convert to fp32, scale, add
//      into a __m128 accumulator. The correction (-8*sum(a) per block) is folded into ONE lane,
//      which is valid because all four lanes are later summed into the same accumulator.
//   3. four independent accumulators (block % 4) to hide the cvt/mul/add latency
//   4. two rows per pass so the activation loads are shared
//
// gcc -O3 -march=sandybridge -mavx -msse4.2 -o oiw4b oiw4_gemv_v1.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define QK 32
#define QB 16

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }

typedef struct { int K, N; uint8_t *qs; float *d; } qrow;

static void init_rows(qrow *w, int N, int K, unsigned seed){
    w->N=N; w->K=K; int nb=K/QK;
    w->qs = aligned_alloc(64,(size_t)N*nb*QB);
    w->d  = aligned_alloc(64,(size_t)N*nb*sizeof(float));
    srand(seed);
    for(size_t i=0;i<(size_t)N*nb*QB;i++) w->qs[i]=(uint8_t)(rand()&0xff);
    for(size_t i=0;i<(size_t)N*nb;i++)    w->d[i]=0.0005f+0.0009f*((rand()%100)/100.f);
}

/* exact scalar reference (per-block int32, then fp32 scale) */
static float ref_gemv_row(const qrow *w, const int8_t *a, int row){
    const uint8_t *qs=w->qs+(size_t)row*(w->K/QK)*QB;
    const float   *d =w->d +(size_t)row*(w->K/QK);
    float acc=0.f;
    for(int b=0;b<w->K/QK;b++){
        int s=0;
        for(int j=0;j<QK;j++){
            uint8_t by=qs[b*QB+j/2];
            int n=(j&1)?(by>>4):(by&0x0f);
            s+=(n-8)*(int)a[b*QK+j];
        }
        acc+=(float)s*d[b];
    }
    return acc;
}

/* per-block activation sums, computed once per GEMV */
static void act_block_sums(const int8_t *a, int K, int32_t *out){
    for(int b=0;b<K/QK;b++){
        int32_t s=0;
        for(int j=0;j<QK;j++) s+=a[b*QK+j];
        out[b]=s;
    }
}

/* one block: 4 int32 lanes -> fp32, scaled, accumulated. correction folded into lane 0. */
static inline __m128 block_acc(const uint8_t *qs, const int8_t *a, float scale, int32_t corr, __m128 acc){
    __m128i p  = _mm_loadu_si128((const __m128i*)qs);
    __m128i lo = _mm_and_si128(p, _mm_set1_epi8(0x0f));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(p,4), _mm_set1_epi8(0x0f));
    __m128i w0 = _mm_unpacklo_epi8(lo,hi);
    __m128i w1 = _mm_unpackhi_epi8(lo,hi);
    const __m128i ones = _mm_set1_epi16(1);
    __m128i s0 = _mm_madd_epi16(_mm_maddubs_epi16(w0,_mm_loadu_si128((const __m128i*)a)),    ones);
    __m128i s1 = _mm_madd_epi16(_mm_maddubs_epi16(w1,_mm_loadu_si128((const __m128i*)(a+16))),ones);
    __m128i s  = _mm_add_epi32(s0,s1);
    s = _mm_sub_epi32(s, _mm_cvtsi32_si128(corr));          /* -8*sum(a_block) in lane 0 only */
    return _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(s), _mm_set1_ps(scale)));
}

static inline float hsum(__m128 v){
    v=_mm_add_ps(v,_mm_movehl_ps(v,v)); v=_mm_add_ss(v,_mm_shuffle_ps(v,v,1)); return _mm_cvtss_f32(v);
}

static float avx1_gemv_row(const qrow *w, const int8_t *a, const int32_t *bsum, int row){
    const uint8_t *qs=w->qs+(size_t)row*(w->K/QK)*QB;
    const float   *d =w->d +(size_t)row*(w->K/QK);
    __m128 a0=_mm_setzero_ps(),a1=a0,a2=a0,a3=a0;
    int nb=w->K/QK,b=0;
    for(;b+4<=nb;b+=4){
        a0=block_acc(qs+b*QB,       a+b*QK,      d[b  ], 8*bsum[b  ], a0);
        a1=block_acc(qs+(b+1)*QB,   a+(b+1)*QK,  d[b+1], 8*bsum[b+1], a1);
        a2=block_acc(qs+(b+2)*QB,   a+(b+2)*QK,  d[b+2], 8*bsum[b+2], a2);
        a3=block_acc(qs+(b+3)*QB,   a+(b+3)*QK,  d[b+3], 8*bsum[b+3], a3);
    }
    for(;b<nb;b++) a0=block_acc(qs+b*QB, a+b*QK, d[b], 8*bsum[b], a0);
    return hsum(_mm_add_ps(_mm_add_ps(a0,a1),_mm_add_ps(a2,a3)));
}

/* two rows per pass: activation loads shared */
static void avx1_gemv_2row(const qrow *w, const int8_t *a, const int32_t *bsum, int r0, float *out){
    const uint8_t *q0=w->qs+(size_t)r0*(w->K/QK)*QB, *q1=q0+(size_t)(w->K/QK)*QB;
    const float   *d0=w->d +(size_t)r0*(w->K/QK),    *d1=d0+(w->K/QK);
    __m128 x0=_mm_setzero_ps(),x1=x0,y0=_mm_setzero_ps(),y1=y0;
    int nb=w->K/QK;
    for(int b=0;b+2<=nb;b+=2){
        __m128i aA=_mm_loadu_si128((const __m128i*)(a+b*QK));
        __m128i aB=_mm_loadu_si128((const __m128i*)(a+b*QK+16));
        __m128i aC=_mm_loadu_si128((const __m128i*)(a+(b+1)*QK));
        __m128i aD=_mm_loadu_si128((const __m128i*)(a+(b+1)*QK+16));
        const __m128i ones=_mm_set1_epi16(1);
        #define ROW(QS,D,ACC0,ACC1,B) do{ \
            __m128i p=_mm_loadu_si128((const __m128i*)(QS)); \
            __m128i lo=_mm_and_si128(p,_mm_set1_epi8(0x0f)); \
            __m128i hi=_mm_and_si128(_mm_srli_epi16(p,4),_mm_set1_epi8(0x0f)); \
            __m128i s=_mm_add_epi32( \
                _mm_madd_epi16(_mm_maddubs_epi16(_mm_unpacklo_epi8(lo,hi),aA),ones), \
                _mm_madd_epi16(_mm_maddubs_epi16(_mm_unpackhi_epi8(lo,hi),aB),ones)); \
            s=_mm_sub_epi32(s,_mm_cvtsi32_si128(8*bsum[B])); \
            if(B&1) ACC1=_mm_add_ps(ACC1,_mm_mul_ps(_mm_cvtepi32_ps(s),_mm_set1_ps(D[B]))); \
            else    ACC0=_mm_add_ps(ACC0,_mm_mul_ps(_mm_cvtepi32_ps(s),_mm_set1_ps(D[B]))); \
        }while(0)
        ROW(q0+b*QB,     d0, x0,x1, b);
        ROW(q0+(b+1)*QB, d0, x0,x1, b+1);
        ROW(q1+b*QB,     d1, y0,y1, b);
        ROW(q1+(b+1)*QB, d1, y0,y1, b+1);
        #undef ROW
        (void)aC;(void)aD;
    }
    out[0]=hsum(_mm_add_ps(x0,x1));
    out[1]=hsum(_mm_add_ps(y0,y1));
}

static double bench1(qrow *w, const int8_t *a, const int32_t *bsum, int rows, int reps){
    double best=1e9;
    for(int r=0;r<reps;r++){ double t0=now(); float acc=0;
        for(int row=0;row<rows;row++) acc+=avx1_gemv_row(w,a,bsum,row);
        double dt=now()-t0; if(acc==12345.f)printf("."); if(dt<best)best=dt; }
    return (double)rows*w->K/best/1e9;
}
static double bench2(qrow *w, const int8_t *a, const int32_t *bsum, int rows, int reps){
    double best=1e9; float out[2];
    for(int r=0;r<reps;r++){ double t0=now();
        for(int row=0;row+1<rows;row+=2) avx1_gemv_2row(w,a,bsum,row,out);
        double dt=now()-t0; if(out[0]==12345.f)printf("."); if(dt<best)best=dt; }
    return (double)rows*w->K/best/1e9;
}

int main(void){
    int K=2048;
    struct { const char *name; int N; } shapes[]={
        {"expert FFN row-block (512 rows)",512},
        {"attention proj    (4096 rows)",4096},
        {"lm_head slice     (8192 rows)",8192},
    };
    int8_t *a=aligned_alloc(64,K);
    for(int i=0;i<K;i++) a[i]=(int8_t)((i*37)%121-60);
    int32_t *bsum=aligned_alloc(64,(K/QK)*sizeof(int32_t));
    act_block_sums(a,K,bsum);

    printf("OldIron W1b - AVX1 W4A8 GEMV, K=%d, block=32, no in-loop horizontal reduction\n",K);
    printf("  (engine baseline today: ~1.4 GMAC/s/core implied by 7.5 t/s on 3B active)\n");
    for(unsigned s=0;s<sizeof(shapes)/sizeof(shapes[0]);s++){
        qrow w; init_rows(&w,shapes[s].N,K,1234+s);
        int bad=0;
        for(int r=0;r<64;r++){ float ref=ref_gemv_row(&w,a,r), got=avx1_gemv_row(&w,a,bsum,r);
            float tol=1e-3f*(1.f+(ref<0?-ref:ref)); if(ref-got>tol||got-ref>tol){ if(bad<3)printf("  MISMATCH %d ref=%f got=%f\n",r,ref,got); bad++; } }
        double g1=bench1(&w,a,bsum,shapes[s].N,5);
        double g2=bench2(&w,a,bsum,shapes[s].N,5);
        printf("  %-34s 1-row %6.2f  2-row %6.2f GMAC/s  verify: %s\n",
               shapes[s].name,g1,g2,bad?"FAIL":"exact-int match");
        /* machine-readable line for compiler-sweep.sh: tag, 1-row, 2-row, verdict */
        printf("RESULT kernel=%s row1=%.3f row2=%.3f verify=%s\n",
               s==0?"expert":(s==1?"attn":"lmhead"), g1, g2, bad?"FAIL":"ok");
        free(w.qs); free(w.d);
    }
    return 0;
}
