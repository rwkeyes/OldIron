# OldIron

A frozen combination of **inference engine + curated weights**, built for hardware that current engines have
written off. OldIron is a product line: each variant names the *lowest* machine it must serve, and nothing in it
is allowed to be slower than that floor needs.

| variant | lowest supported CPU | lowest supported GPU | status |
|---|---|---|---|
| **OldIron-AVX** | Xeon E5-2665 (Sandy Bridge-EP, **AVX1**: no AVX2/FMA/F16C/BMI2) | RTX 3060 8 GB (**sm_86**) | built, staged, **measured on z820a + z820b**; kernel work in progress |
| OldIron-Core2 (planned) | SSE4.1/SSSE3 (2007) | none | not started |

## Results, plainly

**The gains are modest, and worth having.** Nothing here is a 2x, and anyone claiming one on AVX1 hardware is
measuring something else.

Measured on z820a (2x Xeon E5-2665, 32 GB DDR3, RTX 5060 Ti), 35B-A3B weights, coherent output throughout:

| what | measured | note |
|---|---|---|
| CPU decode, stock engine | **7.45 +/- 0.04 t/s** | IQ3_S 3.66 bpw, `-ngl 0 --no-op-offload -t 24` |
| CPU decode, **patched Q4_0 kernel** | **7.93 +/- 0.02 t/s** | same arm, Q4_0 4.50 bpw; **+13.5 %** over stock Q4_0 (6.99 +/- 0.47) |
| W1 GEMV kernel alone | **4.4 GMAC/s/core** from DRAM; 8.9 / 18.1 L1-resident (1 / 2 rows) | exact-integer vs the scalar reference, AVX1-clean |
| NUMA interleave, one flag | **+59 %**: 44.4 -> 70.8 GMAC/s, 27.8 -> 44.2 GB/s | measured, not yet integrated |
| projection when W3 lands | **~20-24 t/s**, about 3x today | at the measured kernel rate |

Why it is still worth doing: the CPU lane moves **11.96 GB/s, just 16 % of this box's 74.9 GB/s**, so the limiter
is the quantised matvec, not memory and not the AVX1 build itself - a `-march=native` build measures **0.7 %**
apart, which is the run-to-run spread. The headroom is measurable, and the kernel that captures it already exists.

What it does **not** buy: the patched Q4_0 arm still does not overtake the narrower IQ3_S (7.93 vs 7.45 t/s) - it
wins back most of the cost of a wider format, nothing more. The GPU and 8 GB lanes are untouched by this kernel
work; they are VRAM-bound, not AVX1-bound.

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

### Curated weights

`Qwen3.6-35B-A3B-IQ3_M.gguf` — 16.9 GB, 3.52 bits/weight, arch `qwen35moe`
(35B total, 3B active, 256 experts / 8 routed + 1 shared, 40 layers, 3:1 Gated DeltaNet : full attention,
262k context, 248320 vocab, MTP-trained).

Why this one: decode speed is set by *active* parameters and RAM is the scarce resource, so a 35B/3B MoE at 3.5
bits fits a 32 GB box with ~15 GB to spare for a long-context KV cache (20 KB/token → 32k ctx ≈ 0.67 GB), where a
27B dense at a higher quant would be ~7× slower for less quality headroom.

## Measured: curated weights on z820a (2× E5-2665, 16 cores, RTX 5060 Ti)

`Qwen3.6-35B-A3B-IQ3_S mix, 3.66 bpw, 15.73 GiB, 34.66B params` — coherent on every lane
("The capital of France is **Paris**.").

| lane | config | decode | prefill | peak VRAM | verdict |
|---|---|---|---|---|---|
| A — CPU only | `-ngl 0 -nopo 1 -t 16` | 7.25 ± 0.10 t/s | 28.8 t/s (pp128) | — | works, kernel-bound |
| A — CPU only | `-ngl 0 -nopo 1 -t 24` | **7.56 ± 0.03 t/s** | 28.5 t/s | — | best CPU arm |
| B — 5060 Ti maximised | `-ngl 99 -ncmoe 4` | **73.3 ± 1.0 t/s** | **763.7 t/s** (pp512) | 14797 MiB | best GPU arm |
| B — | `-ngl 99 -ncmoe 8` | 50.8 t/s | — | 13345 MiB | |
| C — 3060 8 GB, tight | `-ncmoe 24` | 29.7 t/s | — | 7739 MiB | fits, ~450 MiB margin |
| **C — 3060 8 GB, safe** | **`-ncmoe 32`** | **23.8 t/s** | 63.9 t/s | **4905 MiB** | **recommended: 3 GB headroom for context** |
| C — | `-ncmoe 40` | 19.1 t/s | 52.7 t/s | 1869 MiB | |
| — | `-ngl 99` alone | **OOM** | — | — | 15898 MiB allocation fails |

**"GPU full" does not exist for this weight set on a 16 GB card.** 15.73 GiB of weights + KV + context will not fit
15.8 GiB of VRAM, so the GPU lane is *always* hybrid here: `-ncmoe 4` on a 16 GB card, `-ncmoe 32` on an 8 GB card.

CPU lane effective throughput: ~1.54 GB/token (3B active + the 248k-vocab lm_head) ÷ 132 ms ≈ **11.7 GB/s, i.e. 16%
of the 74.9 GB/s this box can deliver.** That is the gap the custom AVX1 kernels are for — it is not memory.

### Earlier proxy run (kept for comparison)

`gpt-oss-20b Q4_K_M`, 20.9B total / 3.6B active, 10.81 GiB, measured before the curated weights finished copying:

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

Parity within run-to-run spread. Repeated on the curated weights: floor **7.56 ± 0.03** (t=24) / 7.25 ± 0.10 (t=16)
vs native **7.63 ± 0.05** t/s — ~1% apart. The full detached harness run (`results/run.log`, 15:52) measured
floor **7.38 ± 0.02** vs native **7.43 ± 0.05** back to back in one pass: **0.7%**, which is the run-to-run spread
itself. Three independent passes of the same arms now agree within ~3-6%, so the floor is confirmed free. The two targets differ only in `-maes` (AES-NI), which no inference kernel
uses, so the cross-built floor is not leaving AVX1 performance on the table and OldIron-AVX can be built anywhere.

(An A/B against the fleet's older `ik_llama-sb-avx1` floor build could not be run: its `llama-cli` exits
`Permission denied` even after `chmod +x`, so that binary is unusable as an arm — unresolved, and not needed while
the same-commit native comparison exists.)

CPU lane effective weight throughput on that run: 3.6B active × 4.85 bits ≈ 2.18 GB/token ÷ 125 ms ≈
**17.5 GB/s, i.e. 23% of the measured 74.9 GB/s DRAM ceiling.** That gap is the kernel headroom this project exists
to capture; it is not a memory limitation.

### Open item: `-fitt` is not a safe VRAM cap

`-fitt 7400` produced a **peak of 8515 MiB** and `-fitt 6400` a peak of **9447 MiB** — both above an 8 GB card's
budget. The fit heuristic and `-ncmoe` fight each other, and the peak includes the CUDA context. Lane C must be
pinned with an explicit `-ncmoe` and verified by sampling `nvidia-smi` *during* the run (sampling before/after
reads 7 MiB and proves nothing).


## One binary, both Z820s (the compatibility claim, measured)

Same OldIron-AVX binary, CPU lane, `-ngl 0 -nopo 1`:

| host | CPU | quant | decode | prefill | effective weight throughput |
|---|---|---|---|---|---|
| z820a | 2× E5-2665 @ 2.4 GHz | IQ3_M, 3.66 bpw | 7.45 ± 0.04 t/s (t=24) | 28.5 t/s | **11.96 GB/s** |
| z820b | 2× E5-2687W @ 3.1 GHz | Q4_K_M, 4.85 bpw | 5.66 ± 0.02 t/s (t=16) | 40.7 t/s | **12.04 GB/s** |

Normalised for quant, the two boxes move **within 0.7% of the same bytes per second** — decode is DRAM-bound, so the
29% clock advantage buys nothing there, exactly as the memory model predicts. Prefill is compute-bound and *does*
scale with clock (1.43×). Coherent on both ("The capital of France is **Paris**.").

That is the compatibility measurement, not an argument: one AVX1 binary covers both boxes with no per-host build, and
the only thing that changed the token rate was the weight file.

## Operational notes (learned the hard way on this build)

- **Run the harness detached, in tmux** — the wifi link to the Z820s drops ssh sessions mid-run. tmux is installed
  without root on both Z820s (`scripts/install-tmux-noroot.sh`: `apt-get download` + `dpkg -x` into `~/tmux-local`,
  wrapper at `~/bin/tmux`); vega has it from apt.
- **Under tmux the CLI gets a pty and enters its interactive REPL** after the first turn, hanging until the timeout
  instead of exiting. Every non-interactive CLI arm needs `< /dev/null`.
- **Never point `cuobjdump` at a symlink and its target together**: given `libggml-cuda.so libggml-cuda.so.0
  libggml-cuda.so.0.27.0` it prints a bogus mixed summary (`sm_50…sm_101`, 1–2 each) that looks like arch
  contamination. On one `readlink -f` path it reports the truth: 145 `sm_120` + 145 `sm_86`.
- **Coherence capture belongs on non-interactive ssh**, not the tmux arm: under the pty the answer text is not
  captured even with stdin closed, so the gate is run over plain ssh where it has produced the correct answer
  repeatedly.

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
    acceptance-z820a.sh     the full matrix on z820a
```
On the target: `/home/bob/OldIron/engine/{bin,lib}`, weights under `/mnt/flare/OldIron/weights`.
