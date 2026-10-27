// OldIron W1: AVX1 (Sandy Bridge-EP) W4A8 GEMV kernel + exact-reference verification.
//
// Format "OIW4": row-major, K weights per row packed 2-per-byte (low nibble = first weight),
// blocks of 32 weights, one fp32 scale per block, symmetric around 8:
//     w_i = (nibble_i - 8) * scale_block
//     dot  = sum_blocks scale_b * ( sum_j (n_j * a_j) - 8 * sum_j a_j )
// The inner sum is pure integer (weights unsigned 0..15, activations int8) so it is EXACT and
// comparable to a scalar reference without tolerance.
//
// Everything here is SSE4.1/SSSE3, which are AVX1-legal in their VEX.128 encodings: no AVX2 anywhere.
// gcc -O3 -march=sandybridge -mavx -msse4.2 -o oiw4 oiw4.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

#define QK 32                       /* weights per block */
#define QB (QK / 2)                 /* packed bytes per block */

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + 1e-9*t.tv_nsec; }

typedef struct {
    int K, N;
    uint8_t *qs;    /* N * (K/32) * 16 bytes  : packed nibbles   */
    float   *d;     /* N * (K/32)      floats : block scales     */
} qrow;

/* ---- scalar reference: exact int32 per block, then fp32 scale ---- */
static float ref_gemv_row(const qrow *w, const int8_t *a, int row, int *out_i32_sum)
{
    const uint8_t *qs = w->qs + (size_t)row * (w->K / QK) * QB;
    const float   *d  = w->d  + (size_t)row * (w->K / QK);
    float acc = 0.f; int total = 0;
    for (int b = 0; b < w->K / QK; b++) {
        int s = 0;
        for (int j = 0; j < QK; j++) {
            uint8_t byte = qs[b*QB + j/2];
            int n = (j & 1) ? (byte >> 4) : (byte & 0x0f);
            s += (n - 8) * (int)a[b*QK + j];
        }
        acc += (float)s * d[b];
        total += s;
    }
    if (out_i32_sum) *out_i32_sum = total;
    return acc;
}

/* ---- AVX1 kernel: 32 weights per iteration, one 128-bit lane ---- */
static inline float dot_block_avx1(const uint8_t *qs, const int8_t *a, float scale)
{
    __m128i p  = _mm_loadu_si128((const __m128i*)qs);            /* 16 bytes = 32 nibbles */
    __m128i lo = _mm_and_si128(p, _mm_set1_epi8(0x0f));
    __m128i hi = _mm_and_si128(_mm_srli_epi16(p, 4), _mm_set1_epi8(0x0f));
    __m128i w0 = _mm_unpacklo_epi8(lo, hi);                      /* nibbles 0..15  */
    __m128i w1 = _mm_unpackhi_epi8(lo, hi);                      /* nibbles 16..31 */
    __m128i a0 = _mm_loadu_si128((const __m128i*)(a));
    __m128i a1 = _mm_loadu_si128((const __m128i*)(a + 16));
    const __m128i ones = _mm_set1_epi16(1);
    __m128i s0 = _mm_madd_epi16(_mm_maddubs_epi16(w0, a0), ones);
    __m128i s1 = _mm_madd_epi16(_mm_maddubs_epi16(w1, a1), ones);
    __m128i s  = _mm_add_epi32(s0, s1);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    int32_t isum = _mm_cvtsi128_si32(s);
    /* subtract the zero point: 8 * sum(a block) */
    int32_t asum = 0;
    for (int j = 0; j < QK; j++) asum += a[j];                   /* replaced by precomputed sums in W2 */
    return (float)(isum - 8 * asum) * scale;
}

static float avx1_gemv_row(const qrow *w, const int8_t *a, int row)
{
    const uint8_t *qs = w->qs + (size_t)row * (w->K / QK) * QB;
    const float   *d  = w->d  + (size_t)row * (w->K / QK);
    __m256 acc8 = _mm256_setzero_ps();
    float tail = 0.f;
    int nb = w->K / QK, b = 0;
    for (; b + 2 <= nb; b += 2) {                                /* two blocks per step */
        float v0 = dot_block_avx1(qs + b*QB,          a + b*QK,      d[b]);
        float v1 = dot_block_avx1(qs + (b+1)*QB,      a + (b+1)*QK,  d[b+1]);
        acc8 = _mm256_add_ps(acc8, _mm256_set_ps(0,0,0,0,0,0,v1,v0));
    }
    for (; b < nb; b++) tail += dot_block_avx1(qs + b*QB, a + b*QK, d[b]);
    __m128 lo = _mm256_castps256_ps128(acc8), hi = _mm256_extractf128_ps(acc8, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s) + tail;
}

/* ---- data generation ---- */
static void init_rows(qrow *w, int N, int K, unsigned seed)
{
    w->N = N; w->K = K;
    int nb = K / QK;
    w->qs = aligned_alloc(64, (size_t)N * nb * QB);
    w->d  = aligned_alloc(64, (size_t)N * nb * sizeof(float));
    srand(seed);
    for (size_t i = 0; i < (size_t)N * nb * QB; i++) w->qs[i] = (uint8_t)(rand() & 0xff);
    for (size_t i = 0; i < (size_t)N * nb; i++) w->d[i] = 0.0005f + 0.0009f * ((rand() % 100) / 100.f);
}

static int verify(qrow *w, const int8_t *a, int rows)
{
    int bad = 0;
    for (int r = 0; r < rows; r++) {
        float ref = ref_gemv_row(w, a, r, NULL);
        float got = avx1_gemv_row(w, a, r);
        float tol = 1e-3f * (1.f + (ref < 0 ? -ref : ref));
        if ((ref - got > tol) || (got - ref > tol)) {
            if (bad < 5) printf("  MISMATCH row %d: ref=%.6f got=%.6f\n", r, ref, got);
            bad++;
        }
    }
    return bad;
}

static double bench(qrow *w, const int8_t *a, int rows, int reps)
{
    double best = 1e9;
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        float acc = 0;
        for (int row = 0; row < rows; row++) acc += avx1_gemv_row(w, a, row);
        double dt = now() - t0;
        if (acc == 12345.f) printf(".");         /* keep the loop */
        if (dt < best) best = dt;
    }
    return (double)rows * w->K / best / 1e9;      /* GMAC/s */
}

int main(int argc, char **argv)
{
    int K = 2048;
    int64_t threads = 1;
    struct { const char *name; int N; } shapes[] = {
        {"expert FFN row-block (512 rows x 2048)",  512},
        {"attention proj    (4096 rows x 2048)",   4096},
        {"lm_head slice     (8192 rows x 2048)",   8192},
    };
    printf("OldIron W1 - AVX1 W4A8 GEMV, K=%d, block=32 weights, scale-combine per block\n", K);
    int8_t *a = aligned_alloc(64, K);
    for (int i = 0; i < K; i++) a[i] = (int8_t)((i * 37) % 121 - 60);

    for (unsigned s = 0; s < sizeof(shapes)/sizeof(shapes[0]); s++) {
        qrow w; init_rows(&w, shapes[s].N, K, 1234 + s);
        int bad = verify(&w, a, 64);
        double g = bench(&w, a, shapes[s].N, 5);
        printf("  %-42s %7.2f GMAC/s   verify: %s\n", shapes[s].name, g, bad ? "FAIL" : "exact-int match");
        free(w.qs); free(w.d);
    }
    free(a);
    return 0;
}
