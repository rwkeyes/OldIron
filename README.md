# OldIron

An inference engine plus **one named weight file**, built for hardware that current engines have
written off. OldIron is a product line: each variant names the *lowest* machine it must serve, and nothing in it
is allowed to be slower than that floor needs. The weights are third-party and are not shipped here — see
*The weights* below.

| variant | lowest supported CPU | lowest supported GPU | status |
|---|---|---|---|
| **OldIron-AVX** | Xeon E5-2665 (Sandy Bridge-EP, **AVX1**: no AVX2/FMA/F16C/BMI2) | RTX 3060 8 GB (**sm_86**) | built, staged, **measured on two HP Z820s**; CPU kernel work finished (+13.5% at fixed format, +6.4% net) |
| OldIron-Core2 (planned) | SSE4.1/SSSE3 (2007) | none | not started |

## Results, plainly

**One measured step, and no more.** The AVX1 kernel work shipped and it is worth **+13.5% at fixed format and
+6.4% net** (7.45 → 7.86-7.93 t/s). Every other CPU-lane lever was measured and rejected. The "~3x" projection in
the first version of this README **did not survive the engine tests and is retracted below** — the large multiple
on this hardware is on the GPU lane, not in the matvec.

All numbers measured on an HP Z820 (2× Xeon E5-2665, 32 GB DDR3, RTX 5060 Ti) with the 35B-A3B weight file named in
each run, coherent output throughout, `nice -n 19`, box verified idle between arms.

### CPU lane — shipped state

| what | measured | note |
|---|---|---|
| stock engine, IQ3_S 3.66 bpw | **7.45 ± 0.04 t/s** | `-ngl 0 --no-op-offload -t 24`, the best prior config |
| stock engine, Q4_0 4.50 bpw | 6.99 ± 0.47 t/s | the wider format is *slower* on stock kernels (23% more bytes) |
| **shipped W3 AVX1 kernel, Q4_0** | **7.86 ± 0.08 t/s** (bench) · 7.93 ± 0.02 (harness) | **+13.5% vs stock Q4_0, +6.4% net over stock IQ3_S**; coherent |
| prefill pp128 / pp512 / pp4096 | 29.70 / 29.9 / 28.0 t/s | 8× the context costs 6% per token |
| W1 GEMV kernel alone | 4.4 GMAC/s/core from DRAM; 8.9 / 18.1 L1-resident (1 / 2 rows) | exact-integer vs the scalar reference, AVX1-clean |
| gcc vs clang 18 vs icx 2026 | gcc best; clang −9%, icx −15.5% | compiler lever closed |

The trade the kernel buys: the Q4_0 weight file is ~19 GiB against 15.7 GiB for IQ3_S. It still fits a 32 GB box,
but it is a size regression in exchange for the speed.

### Retractions — what this README claimed, and what the tests returned

| claim in the first version | measured result |
|---|---|
| "projection when W3 lands ~20-24 t/s, about 3x today" | **retracted.** W3 landed and bought 13.5% at fixed format. The engine moves ~28 GB/s where the pure kernel reaches 44 GB/s, so closing that gap entirely is **~1.35× end-to-end** on a 77% matvec share — not 3×. The kernel path is close to exhausted |
| "NUMA interleave, one flag, +59%" | **0% in the engine** — 3.13 vs 3.13 t/s, two interleaved rounds, on a copy materialised under the interleave policy. `numastat` also shows the page cache already split 50/50 across nodes. The +59% was a 64 MB streaming microbenchmark; the engine's matvec is not purely bandwidth-bound, so it did not transfer |
| "11.96 GB/s = 16% of the 74.9 GB/s ceiling, so 3-5× headroom" | the 74.9 GB/s figure is a **sequential** streaming measurement that this access pattern never reaches. Like-for-like the engine is at 28 GB/s against the kernel's 44 GB/s — 1.6×, worth ~1.35× end-to-end |
| lm_head Q6_K → Q4_0 (417 MB/token on the old path) | **0%** — 7.83 vs 7.93 t/s, inside noise |

### Also measured and rejected (detail in [`kernels/RESULTS.md`](kernels/RESULTS.md))

- **hoisted-correction GEMM: −40% in situ** (7.93 → 4.67 t/s) despite +113% on the isolated matrix — the engine
  makes ~1.9M small `vec_dot` calls per token, so the per-call precompute is never amortised. Reverted.
- **4-row GEMM batching:** inseparable from the hoist that broke it; not shipped.
- **speculative decoding:** 7.3 → 6.8 t/s (**−7%**) with a 0.8B draft on this thinking-heavy output.
- **NUMA chunk granularity** (dropping the `ggml_is_numa()` term): prefill flat, **decode −7%**. Reverted.
- **K-quant tiled GEMM instead of Q4_0 + `vec_dot`:** the tiled path is **44% slower at prefill** on AVX1
  (19.00/15.74 t/s vs 29.9/28.0), because it compiles its generic path without AVX2/VNNI. Q4_0 + the W3 kernel
  wins on decode *and* prefill.
- **CONCAT "10% of prefill":** a profiler artifact — it was the previous node's barrier wait. Parallelising the
  copy changed nothing (pp512 29.97 → 29.92).
- **Prefill ceiling:** pp512 is ~2.5 TMAC at 29.9 t/s ≈ 145 GMAC/s against the measured 170 GMAC/s int8 floor,
  i.e. **85-90%**, not the 52% an earlier active-parameter estimate assumed.

### Where the multiple actually is: the GPU lane

The CPU lane is at its ISA ceiling. The 16 GB card is not. Best long-context split on the 35B weights, measured at
a real 32k context with a 140 KB prompt (not `llama-bench`'s short context):

| config | prompt | generation | peak VRAM |
|---|---|---|---|
| **`-ngl 99 -ncmoe 10 -t 24 -c 32768`, f16 KV** | **832.9 t/s** | **52.5 t/s** | 15339 MiB |
| same + KV q8_0 | 839.2 | 48.3 (−8%) | 15339 MiB |
| `-ncmoe 12 -t 24` | 731.7 | 51.6 | 14465 MiB (1.4 GB spare) |
| CPU only (fallback) | 29.9 | 7.86 | — |

**6.7× the CPU lane's decode and 29× its prefill.** Only 508 MiB spare at `-ncmoe 10`, so 32k is the ceiling with
this split; `-ncmoe 12` costs 2% and holds 1.4 GB of headroom. KV quantisation is a net loss here — the hybrid
DeltaNet KV is ~12 KB/token, so q8 mostly adds dequant work.

## What OldIron-AVX is

One binary, one weight file, three lanes:

| lane | how it runs | target
|---|---|---|
| A — CPU | `-ngl 0` | the AVX1 Xeon pair, no GPU at all |
| B — GPU | `-ngl 99` | RTX 5060 Ti 16 GB (sm_120) |
| C — 8 GB budget | `-ngl 99 -ncmoe N` | RTX 3060 8 GB (sm_86): attention on the card, routed experts on the CPU |

### Engine provenance

- source: llama.cpp `2744e48d1` (build 816, 0.7.0-dev), `qwen35moe` architecture support present
- CPU backend: **one variant**, `-march=sandybridge -mno-avx2 -mno-fma -mno-f16c -mno-bmi2` + `-msse4.2 -mavx`
  (ggml prints that as `-msse4.2;-mavx` / `GGML_SSE42;GGML_AVX` — on Sandy Bridge that *is* AVX1, not a lower floor)
- `-march=sandybridge` also sets `-mtune=sandybridge`, so the code is scheduled for the Z820 core even though it is
  built on a Zen4 host
- CUDA fatbin: **`sm_86` + `sm_120`**, 145 kernels each (RTX 3060 is the supported floor, 5060 Ti the test card)
- `GGML_LLAMAFILE=ON`, `GGML_CPU_REPACK=ON`, `GGML_OPENMP=ON`
- `RUNPATH=$ORIGIN/../lib` on binaries, `$ORIGIN` on libs — the tree is relocatable

### ISA gate (measured, not asserted)

`scripts/isa-audit.sh` on the shipped `libggml-cpu.so`:

```
AVX2-only mnemonics : 0        FMA      : 0
int-vector on %ymm  : 0        F16C     : 0
AVX-512 (%zmm)      : 0        AVX1 %ymm ops : 5335
VERDICT: AVX1-clean and using 256-bit width
```

**The audit pattern matters.** A naive grep for "AVX2 opcodes" reports false positives: `vpmulld`, `vpblendvb`,
`vpminsd`/`vpmaxsd` are SSE4.1 instructions whose **VEX.128 form is legal AVX1**, and unanchored patterns like
`v(p[a-z]*|ps[a-z]*)` match the `vps` inside `vdivps`/`vpermilps`. The sound tests are (a) genuinely AVX2-only
mnemonics (`vpermd`, `vpermq`, `vpbroadcast[dwq]`, `vpsllvd`, `vpgather*`, `vpblendd`) and (b) **integer-vector ops
on `%ymm`** — AVX1 has no 256-bit integer arithmetic at all. `vmovdqa/vmovdqu` on `%ymm` are AVX1-legitimate.

### The weights: third-party, and not shipped

`Qwen3.6-35B-A3B` in GGUF — 35B total / 3B active, arch `qwen35moe` (256 experts, 8 routed + 1 shared, 40 layers,
3:1 Gated DeltaNet : full attention, 262k context, 248320 vocab, MTP-trained). Three files appear in these
measurements: the stock `Qwen3.6-35B-A3B-IQ3_M.gguf` (**16.9 GB, 3.52 bits per weight**), an `IQ3_S` mix
(3.66 bpw, 15.73 GiB) used for the lane matrix, and a `Q4_0` re-quant of the same model (4.50 bpw, ~19 GiB)
written only to A/B the AVX1 kernel.

**These weights are not ours, and this project does not include them.** They are third-party GGUF files of a public
model, downloaded and checksum-verified in transit; the `Q4_0` copy is a local re-quant of the same model and is a
test artefact, not a product. What is published here is engine source, kernel source, scripts and measurements — no
weight file is redistributed, and no weight file is claimed as part of the deliverable.

Why this shape of model: decode speed is set by *active* parameters and RAM is the scarce resource, so a 35B/3B
MoE at 3.5-4.5 bits fits a 32 GB box with ~15 GB to spare for a long-context KV cache (20 KB/token → 32k ctx
≈ 0.67 GB), where a 27B dense at a higher quant would be ~7× slower for less quality headroom.

## Measured: HP Z820 (2× E5-2665, 16 cores, RTX 5060 Ti)

`Qwen3.6-35B-A3B-IQ3_S` mix, 3.66 bpw, 15.73 GiB, 34.66B params — coherent on every lane
("The capital of France is **Paris**.").

| lane | config | decode | prefill | peak VRAM | verdict |
|---|---|---|---|---|---|
| A — CPU only | `-ngl 0 -nopo 1 -t 16` | 7.25 ± 0.10 t/s | 28.8 t/s (pp128) | — | works, kernel-bound |
| A — CPU only | `-ngl 0 -nopo 1 -t 24` | **7.56 ± 0.03 t/s** | 28.5 t/s | — | best CPU arm on stock kernels |
| A — CPU, shipped kernel | Q4_0 + W3, `-t 24` | **7.86 ± 0.08 t/s** | 29.70 t/s (pp128) | — | what actually ships |
| B — 5060 Ti maximised | `-ngl 99 -ncmoe 4` | **73.3 ± 1.0 t/s** | **763.7 t/s** (pp512) | 14797 MiB | best short-context GPU arm |
| B — | `-ngl 99 -ncmoe 8` | 50.8 t/s | — | 13345 MiB | |
| C — 3060 8 GB, tight | `-ncmoe 24` | 29.7 t/s | — | 7739 MiB | fits, ~450 MiB margin |
| **C — 3060 8 GB, safe** | **`-ncmoe 32`** | **23.8 t/s** | 63.9 t/s | **4905 MiB** | **recommended: 3 GB headroom for context** |
| C — | `-ncmoe 40` | 19.1 t/s | 52.7 t/s | 1869 MiB | |
| — | `-ngl 99` alone | **OOM** | — | — | 15898 MiB allocation fails |

**"GPU full" does not exist for this weight set on a 16 GB card.** 15.73 GiB of weights + KV + context will not fit
15.8 GiB of VRAM, so the GPU lane is *always* hybrid here: `-ncmoe 4` on a 16 GB card, `-ncmoe 32` on an 8 GB card.
For long context the split that matters is in *Results, plainly* above (`-ncmoe 10`, 52.5 t/s at 32k).

CPU lane effective throughput: ~1.54 GB/token (3B active + the 248k-vocab lm_head) ÷ 132 ms ≈ **11.7 GB/s.** That
is against the **44 GB/s** the pure kernel reaches on this access pattern, not against the box's 74.9 GB/s
sequential streaming figure — see the retractions above for why the difference matters.

### Earlier proxy run (kept for comparison)

`gpt-oss-20b Q4_K_M`, 20.9B total / 3.6B active, 10.81 GiB, measured before the 35B weights finished copying:

| lane | decode | prefill | note |
|---|---|---|---|
| A CPU only, `-t 16` | **8.03 ± 0.02 t/s** | **23.9 t/s** (pp512, `--no-op-offload`) | coherent ("…is Paris") |
| B GPU full (`-ngl 99`) | **142.7 ± 2.4 t/s** | 297 t/s | coherent |
| C `-ncmoe 16` | 19.7 t/s | — | peak VRAM 4605 MiB |
| C `-ncmoe 24/32` | 13.4–13.9 t/s | — | peak VRAM 1373 MiB |

### Two measurement traps, both found here

**1. `-ngl 0` is not CPU-only when a CUDA backend is present.** Op-offload defaults to ON and moves the batched
matmuls to the GPU. The contaminated arms reported **242–341 t/s of "CPU prefill"** — physically impossible on this
box (it implies ~3 TMAC/s against a ~0.3 TMAC/s ceiling). With `--no-op-offload` the same arm reads **23.9 ± 0.9
t/s**. Decode was never affected, only prefill. Any CPU-lane prefill number without `--no-op-offload` is a GPU
number wearing a CPU label.

**2. `-fitt` is not a VRAM cap** (see below).

### AVX1 floor vs `-march=native`: the floor costs nothing

Same source (2744e48d1), same model, same flags, built both ways on the Z820 itself:

| build | decode (tg48/128) | prefill (pp512) |
|---|---|---|
| OldIron-AVX floor (`-march=sandybridge`, cross-built on Zen4) | 8.55 / 9.03 t/s | 23.9 ± 0.9 t/s |
| native (`-march=native`, built on the E5-2665) | 8.74 / 8.83 t/s | ~25 t/s |

Parity within run-to-run spread. Repeated on the 35B weights: floor **7.56 ± 0.03** (t=24) / 7.25 ± 0.10 (t=16)
vs native **7.63 ± 0.05** t/s — ~1% apart. One detached harness pass measured the two back to back:
floor **7.38 ± 0.02** vs native **7.43 ± 0.05**: **0.7%**, which is the run-to-run spread itself. Three independent passes of the same arms now agree within ~3-6%, so the floor is confirmed free. The two targets differ only in `-maes` (AES-NI), which no inference kernel
uses, so the cross-built floor is not leaving AVX1 performance on the table and OldIron-AVX can be built anywhere.

(An A/B against the fleet's older `ik_llama-sb-avx1` floor build could not be run: its `llama-cli` exits
`Permission denied` even after `chmod +x`, so that binary is unusable as an arm — unresolved, and not needed while
the same-commit native comparison exists.)

CPU lane effective weight throughput on that run: 3.6B active × 4.85 bits ≈ 2.18 GB/token ÷ 125 ms ≈
**17.5 GB/s.** Measured at 8 / 16 threads the engine moves 24 / **28 GB/s**, against 44 GB/s for the pure kernel on
a sequential 64 MB matrix — within 1.6×, which is the honest size of the kernel headroom.

### Open item: `-fitt` is not a safe VRAM cap

`-fitt 7400` produced a **peak of 8515 MiB** and `-fitt 6400` a peak of **9447 MiB** — both above an 8 GB card's
budget. The fit heuristic and `-ncmoe` fight each other, and the peak includes the CUDA context. Lane C must be
pinned with an explicit `-ncmoe` and verified by sampling `nvidia-smi` *during* the run (sampling before/after
reads 7 MiB and proves nothing).


## One binary, both Z820s (the compatibility claim, measured)

Same OldIron-AVX binary, CPU lane, `-ngl 0 -nopo 1`:

| host | CPU | quant | decode | prefill | effective weight throughput |
|---|---|---|---|---|---|
| Z820 #1 | 2× E5-2665 @ 2.4 GHz | IQ3_M, 3.66 bpw | 7.45 ± 0.04 t/s (t=24) | 28.5 t/s | **11.96 GB/s** |
| Z820 #2 | 2× E5-2687W @ 3.1 GHz | Q4_K_M, 4.85 bpw | 5.66 ± 0.02 t/s (t=16) | 40.7 t/s | **12.04 GB/s** |

Normalised for quant, the two boxes move **within 0.7% of the same bytes per second** — decode is DRAM-bound, so the
29% clock advantage buys nothing there, exactly as the memory model predicts. Prefill is compute-bound and *does*
scale with clock (1.43×). Coherent on both ("The capital of France is **Paris**.").

That is the compatibility measurement, not an argument: one AVX1 binary covers both boxes with no per-host build, and
the only thing that changed the token rate was the weight file.

## Operational notes (learned the hard way on this build)

- **Run the harness detached, in tmux** — the wifi link to the Z820s drops ssh sessions mid-run. tmux is installed
  without root on both Z820s (`scripts/install-tmux-noroot.sh`: `apt-get download` + `dpkg -x` into `~/tmux-local`,
  wrapper at `~/bin/tmux`); the build host has it from apt.
- **Under tmux the CLI gets a pty and enters its interactive REPL** after the first turn, hanging until the timeout
  instead of exiting. Every non-interactive CLI arm needs `< /dev/null`. `llama-cli` also writes a **0-byte log**
  under a tmux pty; capture CLI output over a plain ssh pipe (`llama-bench` is fine under tmux).
- **Never point `cuobjdump` at a symlink and its target together**: given `libggml-cuda.so libggml-cuda.so.0
  libggml-cuda.so.0.27.0` it prints a bogus mixed summary (`sm_50…sm_101`, 1–2 each) that looks like arch
  contamination. On one `readlink -f` path it reports the truth: 145 `sm_120` + 145 `sm_86`.
- **Coherence capture belongs on non-interactive ssh**, not the tmux arm: under the pty the answer text is not
  captured even with stdin closed, so the gate is run over plain ssh where it has produced the correct answer
  repeatedly.
- **Confirm the box is idle between arms** — a stale `llama-bench` survived an ssh timeout and competed with three
  subsequent measurements for 23 minutes (one produced a nonsense 1.31 t/s).
- **CMake does not rebuild on a compile-definition change** — flipping `GGML_CPU_REPACK` rewrote `flags.make` and
  left every object in place, producing a fake "repack off" result. Delete the target's `*.o` and touch the sources.
- **Never quote a rate from a profiled run** — `OLDIRON_OPPROF=1` costs ~1.6x (19.0 s of op time for 96 tokens vs
  12.3 s), so a profiler run and a clean run are not comparable.

## Acceptance gates (a fast wrong answer is a fail)

1. **ISA** — `scripts/isa-audit.sh` on the shipped libs: 0 above the floor, `%ymm` present.
2. **Fatbin coverage** — `cuobjdump --list-elf` shows `sm_86` and `sm_120`.
3. **Coherence** — fixed prompts through each lane, text compared to a known-good engine. This fleet has shipped a
   floored build that answered `The capital of France is!!!!!!!!` at 65 t/s; speed is measured last, on purpose.
4. **Speed** — `llama-bench`, thread sweep on the CPU lane (expect an optimum, not the largest count).
5. **8 GB class** — lane C must peak under ~7.8 GB sampled during the run.

## What is in this repository

Source, scripts and measurements. Not included, deliberately:

- `src/` - the engine is an unmodified llama.cpp worktree at commit `2744e48d1`; clone it from upstream rather than
  vendoring it here.
- weights - `Qwen3.6-35B-A3B` and `gpt-oss-20b` GGUFs are third-party downloads (HF), not redistributed.
- build output, staged binaries, kernel test binaries.

```
kernels/   the AVX1 quantised GEMV/GEMM work (source only) and RESULTS.md, the waypoint log
scripts/   build (AVX1 floor + sm_86;120), ISA audit, staging, weight transfer, acceptance matrix
results/   the measured runs
```

## License

GPL-3.0-or-later. See `LICENSE`; the copyright notice and the third-party terms (llama.cpp is MIT and stays MIT)
are in `NOTICE`.

## Layout

```
/home/bob/OldIron/
  src/            llama.cpp worktree @ 2744e48d1
  build/          the built engine (bin/ + lib/)
  stage/          patched-RUNPATH copy for shipping
  scripts/
    build-oldiron-avx.sh    build (AVX1 floor + sm_86;120)
    isa-audit.sh            the ISA gate (see note above about the pattern)
    stage-to-target.sh      relocatable copy + load check
    fetch-weights.sh        parallel-stream weight transfer (link is wifi/WAN here)
    acceptance-hp-z820.sh    the full matrix on the Z820
```
On a target machine the engine sits at `<root>/engine/{bin,lib}` with the weight file wherever you keep it — nothing
under this repository's tree.
