# OldIron kernel waypoints (AVX1 / Z820)

Engine baseline today (measured, HP Z820, 35B-A3B IQ3_S): **7.45 t/s CPU decode**, which is
**11.96 GB/s of weight traffic = 16% of the box's 74.9 GB/s** — and 22.5 GMAC/s aggregate, i.e.
**~1.4 GMAC/s per core**. Neither the memory wall (46.7 t/s) nor the compute wall is being reached: the
engine's quantised matvec is the limiter.

| # | waypoint | state | measured effect |
|---|---|---|---|
| **W1** | **AVX1 W4A8 GEMV kernel** (`oiw4_gemv_v1.c`) | **done** | **4.4 GMAC/s/core streaming from DRAM; 7.5 (1 row) / 17.6–18.1 (2 rows) L1-resident; exact-integer match vs scalar reference** |
| W2 | NUMA placement (interleave / per-socket shard) | measured, not integrated | **+59% aggregate**: 44.4 → **70.8 GMAC/s**, 27.8 → 44.2 GB/s |
| W3 | GGUF → OIW4 converter + engine integration | next | target **~20–24 t/s** (3× today) at the measured kernel rate |
| W4 | prefill GEMM (int8, register-blocked) | not started | today 28.5 t/s prefill on the CPU lane |
| W5 | per-layer bit-width / vocab pruning on top of W3 | not started | lm_head is ~14% of per-token bytes |
| W6 | batch decode (GEMV → GEMM amortisation) | not started | MoE: bytes/token fall with batch |

## W1 detail — what made it fast (and what didn't)

v0 (per-block horizontal reduction + scalar activation sums): **3.2 GMAC/s/core**, exact.
v1 (fixed): **8.9 GMAC/s/core** 1-row, **18.1** with 2 rows sharing activation loads.
The two fixes that mattered, in order:

1. **No horizontal reduction in the loop.** Keep 4 int32 lanes, convert to fp32, multiply by the block
   scale, add into a `__m128` accumulator; reduce once per row. The zero-point correction
   (`-8·sum(a_block)`) is folded into a single lane — legal because all lanes land in the same accumulator.
2. **Precompute per-block activation sums once per GEMV** (they are shared by every row) instead of a
   32-iteration scalar loop per block.

ISA check on the built kernel: `vfmadd = 0`, integer-on-`%ymm` = 0 → AVX1-clean, as required.

## W1c — the DRAM/NUMA wall (this is what gates the engine)

65536×2048 matrix (64 MB packed + 16 MB scales, deliberately larger than L3):

| placement | 8 threads | 16 threads |
|---|---|---|
| default malloc (all pages on node 0), threads on both sockets | 40.3 GMAC/s · 25.2 GB/s | 44.4 GMAC/s · 27.8 GB/s |
| `numactl --interleave=all` | 54.0 · 33.8 | **70.8 GMAC/s · 44.2 GB/s** |
| one socket, fully local | 32.1 · 20.1 | 52.6 · 32.9 (8 cores) |

**Interleaving is +59% for one flag.** It is also the reason the 16-thread default arm is *worse per core*
than 8 threads: the second socket reads everything over QPI at ~13 GB/s.

Projection at the measured 70.8 GMAC/s and 44.2 GB/s (3.0 GMAC and 1.60 GB per token):
compute ceiling ≈ **23.6 t/s**, memory ≈ **27.6 t/s** → a realistic **~20–24 t/s**, i.e. **~3× the 7.45 t/s**
the engine does today, before any prefill work.


## W3 — first engine integration (patched `ggml_vec_dot_q4_0_q8_0`)

Replaced ggml's AVX1 branch with the W1 structure (unsigned nibbles straight into `maddubs`, zero point
removed **lane-wise** so there is no in-loop horizontal reduction, 4 accumulators, one reduce per call).
Built kernel still ISA-clean. Same model, same arm (`-ngl 0 -nopo 1 -t 24`, CPU lane):

| weights | engine | decode | note |
|---|---|---|---|
| IQ3_S 3.66 bpw | stock | **7.45 t/s** | |
| Q4_0 4.50 bpw | stock | 6.99 ± 0.47 t/s | simpler format is *slower* — 23% more bytes |
| Q4_0 4.50 bpw | **patched kernel** | **7.93 ± 0.02 t/s** | **+13.5% over stock Q4_0**, coherent |

So the kernel change is real but worth 13.5%, not the ~3× the standalone kernel promised. The op profile
below says why.

### Op profile (instrumented ggml; `OLDIRON_OPPROF=1`, 32 tokens, 24 threads)

| op | ms | % | calls |
|---|---|---|---|
| MUL_MAT | 2021.4 | **52.9%** | 13685 |
| MUL_MAT_ID (routed experts) | 915.8 | **24.0%** | 4200 |
| GET_ROWS | 429.0 | **11.2%** | 3605 |
| CONCAT | 119.2 | 3.1% | 1050 |
| GATED_DELTA_NET | 89.0 | 2.3% | 1050 |
| UNARY / SCALE / RMS_NORM / ADD / MUL | 151.0 | 4.0% | ~40k |
| FLASH_ATTN_EXT | 23.1 | 0.6% | 350 |
| CPY / ARGSORT / SSM_CONV / GLU / SOFT_MAX | 58.0 | 1.5% | ~9k |

Matvecs are **77%** of decode → the kernel is the right target. Everything else (DeltaNet, attention,
norms) is ~10%, so the "attention overhead" hypothesis is dead.

### Why the patch only bought 13.5%: two tensors are off the fast path

Tensor-type audit of the Q4_0 file (`kernels/gguf_types.py`):

| tensor | type | params |
|---|---|---|
| everything else | Q4_0 (34.1 B params) | on the patched kernel |
| **`output.weight` (lm_head)** | **Q6_K** | **508.6 M — 417 MB read every token, on the old q6_K path** |
| `ffn_gate_inp` routers | F32 | 22.2 M — 89 MB/token on the fp32 path |

Per-token weight traffic ≈ **2.19 GB** (1.69 GB MoE active + 0.42 GB lm_head + 0.09 GB routers).
At 7.93 t/s that is 17.4 GB/s — still only 23% of the 74.9 GB/s ceiling, so traffic is not the wall:
the *kernel* is, and two of the three biggest matvecs are still running on unpatched, slower paths.

**Next:** put `output.weight` on Q4_0 (one quantize flag) and re-measure; then decide between patching
`q6_K` or keeping the lm_head at 4 bits.


## Compiler back ends, measured on the Z820 (W1 kernel, identical source and flags)

`-O3 -march=sandybridge -mavx -msse4.2`, all three verified AVX1-clean, all three producing exact-integer matches:

| back end | 1-row | 2-row (the variant that matters) |
|---|---|---|
| **gcc 13.3.0** | 8.93 | **18.15 / 17.92 GMAC/s** |
| clang 18.1.3 | 8.93 | 16.52 / 16.36 (−9%) |
| Intel icx 2026.2 | 7.74 | 15.33 / 15.19 (−15.5%) |
| gcc `-O2` / `-O3 -funroll-loops` | same | 70.2 / 69.5 GMAC/s *on a Zen4 host* — i.e. flag choice is worth <1% |

**gcc is the right compiler and is already in use.** The two LLVM back ends unroll harder and lose on a 16-register
ISA where the 2-row loop is register-starved. clang also has no `libomp` on the Z820 (needs `-fopenmp=libgomp`), and
icx exists only on the build host. Compiler lever: closed — the alternatives are 0 to −15%, never better.

## NUMA: +6% in the engine, but not reachable

| placement / load mode | decode |
|---|---|
| mmap + default (what ships) | **7.83–7.93 t/s** |
| `--load-mode none` + default | 6.80 ± 0.03 |
| `--load-mode none` + `numactl --interleave=all` | 7.21 ± 0.19 (**+6.0%** vs the same mode) |

Interleave genuinely helps (**+6.0%**, same load mode), but the anonymous load mode costs ~13%, so the only way to
collect it is interleaved placement of the **mmap'd page cache** — which needs a cold cache (`drop_caches`, root) or
a fresh copy of the file. The microbenchmark's +59% does not transfer, because in the engine the matvec is 77% of
the time and is not purely bandwidth-bound.

Note the earlier `--interleave=all` run that showed no effect was **invalid**: the file was already in page cache,
and `numactl` cannot re-place pages that were faulted under a different policy.


## NUMA placement: definitively no effect (and the "74.9 GB/s ceiling" framing was wrong)

Materialised a second copy of the weights **while running under `numactl --interleave=all`**, so the copy's
page-cache pages inherit the interleave policy — the only honest way to test placement, since `numactl` cannot
re-place pages faulted earlier. Same file contents, same engine, same thread count, two interleaved rounds:

| arm | round 1 | round 2 |
|---|---|---|
| default placement | 3.13 | 3.13 |
| interleaved copy | 3.13 | 3.15 |

Identical to within 0.02 t/s. `numastat` also shows `FilePages` already split ~50/50 (13241 / 13292 MiB), so the
premise "all weights live on node 0" was never true for a cached file. **NUMA is eliminated.**

### Correction to the headroom estimate

I had been quoting "16% of the 74.9 GB/s ceiling" and therefore "3-5x headroom". That 74.9 GB/s figure is a
**sequential** streaming measurement. Measured against what this access pattern can actually reach:

| arm | threads | weight traffic |
|---|---|---|
| engine, Qwen3-14B dense | 1 | 3.7 GB/s |
| engine | 8 | 24 GB/s |
| engine | 16 | **28 GB/s** |
| pure W4A8 kernel, sequential 64 MB matrix | 16 | **44 GB/s** |

The engine is within **1.6x** of the pure kernel on the same silicon, and its 77% matvec share means closing that
gap entirely would buy ~1.35x end-to-end, not 3-5x. Thread scaling is also respectable rather than broken
(7.7x from 16 threads vs the kernel's 9.4x) — the collapse past 8 threads is partly the machine, not the engine.

**Consequence for the plan:** the kernel path is close to exhausted. Multiples now have to come from doing *fewer
bytes of work per token*, not from making the matvec faster:
1. **batch decode** — for an MoE, one weight read serves several tokens: the only lever with a >2x ceiling here
2. lower bits on the tensors that dominate traffic
3. vocabulary pruning (the 248k lm_head is ~20% of per-token bytes)


## Speculative decoding: net loss on this box (7.3 -> 6.8 t/s)

Draft = Qwen3.5-0.8B Q4_K_M (532 MB, same 248k vocab family as the target), target = Qwen3.6-35B-A3B,
same prompt, temp 0, t=24, `-n 32`:

| arm | generation |
|---|---|
| baseline | **7.3 t/s** |
| with `-md` draft | **6.8 t/s** (-7%) |

The draft reads ~0.5 GB/token (18 ms against a 137 ms target step), and this model emits a long thinking
preamble whose tokens are not predictable enough to accept drafts at a rate that repays that cost. Speculation
is *mechanically* available (`-md`, `--spec-draft-*`) and may pay on highly repetitive output, but it does not
pay on this workload. Do not ship it as a default.

## Where the headroom actually is (revised, after five eliminations)

| lever | result |
|---|---|
| kernel (W4A8, patched `vec_dot_q4_0_q8_0`) | **+13.5%** at fixed format, **+6.4%** net over the best prior config — the only proven gain |
| compiler choice (gcc vs clang vs icx) | gcc already best; alternatives -9% / -15.5% |
| NUMA interleave / placement | 0% (3.13 vs 3.13); page cache already balanced across nodes |
| lm_head Q6_K -> Q4_0 | 0% (7.83 vs 7.93, within noise) |
| speculative decoding | -7% |
| engine vs pure kernel | engine at 28 GB/s, kernel at 44 GB/s -> closing it entirely is ~1.35x on a 77% matvec share |

Decode is within ~1.4x of what this silicon does for this access pattern; prefill (pp128 29-30 t/s = ~87 GMAC/s)
is at roughly half the measured int8 compute floor, so ~2x is the ceiling there. Multiples beyond ~1.5x are not
available from the engine: they require changing the bytes per token (batch serving, lower bits, vocabulary), or a
different model.

## Operational trap: `llama-cli` under tmux writes nothing

Every `llama-cli` run redirected to a file inside a tmux session produced a **0-byte log** and no timing line,
while the identical command over plain ssh works (verified: "Prompt 22.3 t/s | Generation 7.3 t/s"). tmux is fine
for `llama-bench` and for long shells; capture CLI *output* over a plain ssh pipe.


## The hoisted-correction GEMM: big in isolation, a 40% loss in situ (reverted)

The isolated prototype was unambiguous, on the Z820 itself, core-pinned and verified:

| kernel shape | GMAC/s |
|---|---|
| 1 row, correction inline (= what ggml's vec_dot did) | 3.28 |
| 1 row, correction hoisted | 4.05 (+23%) |
| 4 rows, correction hoisted | 6.99 (+113%) |

Implemented for real: mul_mat precomputes `8*sum(a_block)` once per activation column into thread-local
scratch, arms a pointer around the row loop only (so `mul_mat_id` can never read stale sums), and the kernel
reads it with one `movd`. Correctness held (coherent output). Speed did not:

| build | pp128 | tg48 |
|---|---|---|
| W3 kernel (shipped) | 29.50 | **7.93** |
| W3 + hoist + 4-row batching | 29.46 | **4.67** |

A 40% regression, reproduced three times (4.62 / 4.67 / 4.68), with the machine verified idle (load 0.28,
42 C, full clock) and the weights verified cache-warm (5.8 GB/s). Profile moved the wrong way everywhere:
MUL_MAT 63 -> 91 ms/token, MUL_MAT_ID 28.6 -> 80 ms/token. Reverted; 7.70-7.93 restored.

**Why the isolated number did not transfer.** The prototype amortised the hoist over 512 contiguous rows per
call. The engine makes ~1.9M `vec_dot` calls per token across 428 small matvecs, so the per-column precompute
is paid per tiny call, and the extra pointer + branch cost the 4-accumulator schedule its registers. Lesson:
a kernel win measured on one big contiguous matrix does not imply a win at 428 small ones - gate on the
engine, not the microbenchmark.

## Instrumentation traps found the hard way

- `OLDIRON_OPPROF=1` costs ~1.6x (19.0 s of op time for 96 tokens vs 12.3 s). Never quote a rate from a
  profiler run, and never compare a profiler run against a clean one.
- **CMake does not rebuild on a compile-definition change.** Flipping `GGML_CPU_REPACK` rewrote flags.make but
  left every object in place, silently producing a build that did not match its cache. Force it: delete the
  target's `*.o` and touch the sources. This produced a fake "repack off" result that cost a debug cycle.
- `llama-cli` under tmux writes a 0-byte log and no timing line; the same command over a plain ssh pipe works.
  `llama-bench` is fine under tmux.


## Shape census: where the MACs actually live, and the corrected prefill ceiling

Counts only, no clocks, so the rates stay honest (verified: 7.87 tg / 29.93 pp with the census on, i.e. free).

| workload | batching width | share of its MACs |
|---|---|---|
| MUL_MAT (dense projections) | 257+ tokens, 770 calls | **93.8%** |
| MUL_MAT (dense) | 1 token (decode), 19171 calls | 6.2% |
| MUL_MAT_ID (MoE experts) | **5-16 tokens, 4.08M calls** | **84.8%** |
| MUL_MAT_ID | 17-64 tokens | 6.9% |
| MUL_MAT_ID | 1 token | 5.4% |

Two things follow. First, the dense path is genuinely GEMM-shaped (512 tokens per call) - a blocked GEMM
would apply. Second, the MoE path is not: 4.08M separate calls at ~8 tokens per expert. Any GEMM has to be
tuned for 8-16 columns, not 512, and the per-call overhead is paid 4.08M times.

**Corrected prefill ceiling.** Total MACs in a pp512 run are ~2.5 TMAC, because the *dense* attention and
projection matrices (not the 3B-active MoE) dominate prefill. At 29.9 t/s that is ~145 GMAC/s against the
measured 170 GMAC/s int8 floor = **85-90%**. The earlier "52% of ceiling, so 2x available" was wrong: it
assumed prefill cost scales with active parameters. It does not. A GEMM rewrite is worth ~10-15%, not 2x,
which is why it is not worth days on this hardware.

## Long context barely costs prefill throughput

| op | pp512 | pp4096 |
|---|---|---|
| MUL_MAT | 43.1% | 41.6% |
| MUL_MAT_ID | 36.8% | 36.3% |
| CONCAT | 10.0% | 9.5% |
| FLASH_ATTN_EXT | 1.1% | **6.9%** |
| GATED_DELTA_NET | 2.9% | 2.8% |

pp4096 = 28.05 t/s vs pp512 = 29.93 t/s: **6% slower per token at 8x the context**. Attention grows 6x in
share but matmul still dominates, so the hybrid DeltaNet keeps long-context prefill cheap. For a 2k-token
prompt that is ~70 s, and it does not degrade much at 32k.

## The one clearly removable cost left: CONCAT (10% of prefill)

25 s of 266 s at pp4096, 480 calls, pure data movement. Implementation copies one row at a time
(`memcpy` of `row_size`) and parallelises only over `i2`: `for (i2 = ith; i2 < ne02; i2 += nth)`, with the
token loop serial inside each thread. If `ne02` is small (heads), most threads idle while each does thousands
of small strided copies - consistent with ~52 ms per call. Fixing this needs the real tensor shapes, which is
the next measurement, not a guess.


## CONCAT was a profiler artifact; the real cost is cross-expert imbalance

The CONCAT shape is `out[515, 8192]` from `a[3, 8192]` + `b[512, 8192]`, dim 0, with **ne02 = ne03 = 1**. The
upstream loop `for (i2 = ith; i2 < ne02; i2 += nth)` therefore ran entirely on thread 0 while 23 threads idled.
Parallelising the copy across (i1, i2) made it correct and 24x more parallel - and changed nothing:

| | before | after |
|---|---|---|
| CONCAT in profile | 3297 ms, 10.0% | **3035 ms, 10.0%** |
| pp512 | 29.97 | 29.92 |
| pp4096 | 28.05 | 28.00 |

So the copy was never the cost. The op profile brackets each node including the barrier wait at its start, so
the straggler time from the *previous* node (the MoE matmul) is attributed to whichever node follows - CONCAT
in this graph. The 10% is cross-expert load imbalance: experts carry wildly unequal token counts, and the
loop over experts is the scheduling unit, so chunk-level work stealing inside one expert cannot fix it.
Raising chunk granularity by dropping the `ggml_is_numa()` term made prefill unchanged (29.91/28.07) and
**decode 7% worse** (7.87 -> 7.31) - upstream's NUMA preference is correct here. Reverted.

Capturing that last 10% needs a flattened (expert, row-chunk) work queue, i.e. a real scheduling rewrite in
`mul_mat_id`, with a ceiling of ~10% and no guarantee. Not attempted.

## Final shipped state (HP Z820, 24 threads, 35B weights)

| test | value |
|---|---|
| tg48 | **7.86 +- 0.08** t/s |
| pp128 / pp512 / pp4096 | 29.70 / 29.9 / 28.0 t/s |

Shipped kernel is the W3 AVX1 q4_0 dot (+13.5% at fixed format, +6.4% net over the best prior config).
Everything else tried today was measured and rejected: hoisted correction (-40%), 4-token GEMM batching
(inseparable from the hoist that broke it), speculative decoding (-7%), NUMA interleave (0%), lm_head
requantisation (0%), alternate compilers (-9%/-15.5%), NUMA chunking (-7% decode). Long-context prefill is at
85-90% of the measured int8 ceiling, so the platform's compute headroom is genuinely near spent.


## Best configuration for long context on the HP Z820 (GPU + CPU split)

The CPU lane is at its ISA ceiling; the 16 GB card is not. Sweeping `-ncmoe` (MoE layers kept on the CPU),
all at `-t 24`, Q4_0 re-quant weights:

| -ncmoe | pp4096 (bench) | tg48 (short ctx) | peak VRAM @32k ctx | verdict |
|---|---|---|---|---|
| 8 | 358.6 | 68.3 | 15795 @4k -> **no KV room** | 32k OOM |
| **10** | 291.8 | 60.9 | **15339** | **fastest that holds 32k** |
| 12 | 248.1 | 58.4 | 14465 | safe, 1.4 GB spare |
| 14 | 214.8 | 53.6 | 13601 | conservative |
| CPU only | 29.9 | 7.86 | - | fallback |

Measured at a real 32k context with a 140 KB prompt (not llama-bench's short context):

| config (@32k) | prompt | generation |
|---|---|---|
| **ncmoe=10, t=24, f16 KV** | **832.9 t/s** | **52.5 t/s** |
| ncmoe=10 + KV q8_0 | 839.2 | 48.3 (-8%) |
| ncmoe=10, t=16 | 810.9 | 46.0 (-12%) |
| ncmoe=12, t=24 | 731.7 | 51.6 |

**Recommended:** `-ngl 99 -ncmoe 10 -t 24 -c 32768` -> 52.5 tok/s generation, 833 tok/s prompt at 32k
context, 15339 MiB peak. That is **6.7x the CPU lane's decode and 29x its prefill**. Only 508 MiB spare, so
32k is the ceiling with this split; drop to `-ncmoe 12` (51.6 tok/s, 1.4 GB spare) if headroom matters more
than 2%. KV cache quantisation is a net loss here (-8%): the hybrid DeltaNet KV is only ~12 KB/token, so it is
already small enough that q8 mainly adds dequant work in attention.

Left on the table deliberately: ncmoe 8 is 30% faster in prefill but cannot allocate a 32k KV, and ncmoe 6/4
cannot even load the weights (16665 / 15801 MiB against a 15847 MiB card).


## MoE scheduler investigation: closed, no win available on AVX1

### The scheduler is already balanced

`ggml_compute_forward_mul_mat_id` iterates experts in the outer loop and has **every thread visit every
expert**, splitting each expert's rows across threads. Because the expert loop is shared, uneven token counts
per expert do not imbalance anything - the row split is proportional to work, and all threads do the same
experts in the same order. The residual skew is `ceil(nr0/nth)` rounding plus SMT siblings being slower than
full cores on this 24-thread-on-16-core setup: a ~5% ceiling, not the 10% the CONCAT profile implied. The one
lever that looked reachable (`ggml_is_numa()` forcing one chunk per thread) was measured: prefill flat, decode
7% worse. Nothing to win here.

### The real finding: Q4_0 silently disables ggml's tiled GEMM - and that is CORRECT here

`ggml_tiled_supported()` (enabled by default via `GGML_CPU_TILED_MM`) accepts only K-quants:
Q6_K, Q5_K, Q4_K, Q3_K, Q2_K, IQ4_XS, IQ2_*, IQ3_XXS, **IQ3_S**, IQ1_*. **Q4_0 is absent from the list**, so
every Q4_0 matmul - dense included, and dense is 93.8% of prefill MACs at 257+ columns - falls through to
`vec_dot`. Since re-quantising to Q4_0 was what enabled the W3 kernel, this looked like a large opportunity:
switch to a K-quant and get a hand-blocked GEMM for the prefill case.

Measured, same engine, same box, idle (load 1.03), weights warm:

| weights | path | pp512 | pp4096 | tg48 |
|---|---|---|---|---|
| **Q4_0 (W3)** | vec_dot | **29.9** | **28.0** | **7.86** |
| IQ3_M (IQ3_S/IQ3_XXS) | **tiled GEMM** | 19.00 | **15.74** | 7.24 |

The tiled path is **44% slower at prefill**. The reason is in the source: the kernel is built for
`__AVX2__`/`__AVX512VNNI__`, and on an AVX1-only build it compiles the generic path - so the "better" GEMM is
the worse kernel on Sandy Bridge. Q4_0 + vec_dot + the W3 kernel wins on both decode and prefill, and the
quantisation choice is now justified by prefill measurement as well as decode.

### Measurement hazard worth recording

A stale `llama-bench` survived an ssh timeout and kept running for 23 minutes, silently competing with three
subsequent measurements (one produced a nonsense 1.31 t/s). Always confirm `pgrep` is empty between arms, and
re-measure anything surprising on a verified-idle box.

## Honest limits of this result

- The kernel is a **GEMV** (batch 1). Prefill is a different kernel (GEMM) and is not addressed by W1.
- 4.25 bpw for this format vs 3.66 for IQ3_S: the weight file grows from 15.7 GiB to ~19 GiB. Fits 32 GB,
  but it is a size regression that buys speed.
- Numbers are one socket's worth of silicon under `nice -n 19`; treat ±5% as noise and do not compare
  against arms measured on other days without re-running.
