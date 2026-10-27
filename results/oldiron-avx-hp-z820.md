# OldIron-AVX — results on the HP Z820

Provenance: `llama.cpp 2744e48d1` (0.7.0-dev, build 816), cross-built on a Zen4 host with CUDA 13.2, staged to the Z820.
CPU backend **one AVX1 variant** (`-march=sandybridge -mno-avx2 -mno-fma -mno-f16c -mno-bmi2`);
CUDA fatbin `sm_86` + `sm_120`. Measurements 2026-10-25, `nice -n 19`.

Host: 2× Xeon **E5-2665** (8C/socket, 2.4 GHz base / 3.1 turbo, **AVX1 only**), 31 GB DDR3, 2 NUMA nodes,
RTX 5060 Ti 16 GB (15847 MiB) on PCIe 3.0 ×8.

Weights: **Qwen3.6-35B-A3B**, IQ3_S mix, **3.66 bpw, 15.73 GiB, 34.66B params** (`qwen35moe`; 3B active,
256 experts / 8 routed + 1 shared, 40 layers, 262k context, 248k vocab, MTP-trained). sha256 verified in transit.
Third-party GGUF weights of a public model — not ours, not modified, and not redistributed here.

## Gates (all passed)

| gate | result |
|---|---|
| ISA audit of shipped `libggml-cpu.so` | AVX2-only 0, int-vector-on-%ymm 0, FMA 0, F16C 0, AVX-512 0, **%ymm ops 5335** |
| CUDA fatbin | **145 × sm_86, 145 × sm_120** |
| Coherence | `The capital of France is **Paris**.` · `9.9 > 9.11` |

## Decode / prefill, per lane

| lane | config | decode (tg) | prefill (pp) | peak VRAM |
|---|---|---|---|---|
| CPU only | `-ngl 0 --no-op-offload -t 16` | 7.31 ± 0.02 t/s | 28.1–28.8 t/s (pp128) | — |
| **CPU only** | **`-ngl 0 --no-op-offload -t 24`** | **7.45 ± 0.04 t/s** (range 7.45–7.61) | 28.5 t/s | — |
| 5060 Ti maximised | `-ngl 99 -ncmoe 4` | **69–73 t/s** (69.21 ± 0.80 / 71.87 ± 0.99 / 73.26 ± 0.96) | **763.7 ± 32.3 t/s** (pp512) | 14797 MiB |
| | `-ngl 99 -ncmoe 8` | 50.76 ± 2.67 | — | 13345 MiB |
| | `-ngl 99 -ncmoe 16` | 34.66 ± 0.16 | — | 10575 MiB |
| 3060 8 GB, tight | `-ngl 99 -ncmoe 24` | 26.59–29.69 | — | **7739 MiB** (fits, ~450 MiB margin) |
| **3060 8 GB, safe** | **`-ngl 99 -ncmoe 32`** | **20.85–23.84** | 52.7–63.9 t/s | **4905 MiB** |
| | `-ngl 99 -ncmoe 40` | 19.11 ± 2.91 | 52.7 ± 5.1 | 1869 MiB |
| GPU full | `-ngl 99` | **OOM** | — | 15898 MiB allocation fails |

## AVX1 floor vs `-march=native` (same commit, both on the Z820)

| build | decode t=16 | decode t=24 |
|---|---|---|
| floor (`-march=sandybridge`, cross-built on Zen4) | 7.38 ± 0.02 (one pass), 7.25 ± 0.10 (another) | 7.45 / 7.56 / 7.61 |
| native (`-march=native`, built on the E5-2665) | 7.43 ± 0.05, 7.63 ± 0.05 | — |

**0.7% apart** — inside run-to-run spread. Three independent passes agree within ~3–6%. The AVX1 floor costs
nothing, so OldIron-AVX can be built on any host.

## Where the time actually goes

Per decode token the model streams ~**1.61 GB** (3B active + the 248k-vocab lm_head at 3.66 bpw).
At 7.45 t/s that is **11.96 GB/s — 16% of the 74.9 GB/s this box measured as its DRAM ceiling.**

| bound | computed ceiling | measured | utilisation |
|---|---|---|---|
| memory (74.9 GB/s) | 46.7 t/s | 7.45 | 16% |
| compute (170 GMAC/s int8 floor) | 56.6 t/s | 7.45 | 13% |

Neither wall is being hit: DRAM idles ~84% of the time and the cores are 87% idle. This is stock ggml's AVX1
quantised kernel, not the platform. It is the headroom the custom AVX1 kernels are meant to take, and the reason
the CPU lane is not the interesting number yet.

## Reference point on a smaller model

`gpt-oss-20b Q4_K_M` (20.91B total / 3.6B active, 10.81 GiB): CPU lane **8.03 ± 0.02 t/s**, GPU full
**142.72 ± 2.43 t/s**. (Its CPU prefill was measured before the op-offload fix and is discarded — the same
mistake that produced an impossible 242–341 t/s "CPU prefill".)

## Known spread and caveats

- Lane C drifts ~10% between passes (29.69 → 26.59 at `-ncmoe 24`) when run back-to-back with other work; read
  those as mid-20s. The CPU lane is tighter (±0.02–0.14).
- Coherence text is captured over plain ssh, not through the tmux pty (see README operational notes).
- `-fitt` is unusable as an 8 GB cap (7400 target → 8515 MiB peak; 6400 → 9447).
- Open slot: where 24-thread and 16-thread decode are within noise, the box is memory-bound and adding HT threads
  buys nothing — consistent with the earlier proxy sweep.
