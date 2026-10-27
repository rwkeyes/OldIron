#!/bin/bash
# OldIron-AVX acceptance harness -- runs ON the Z820 (2x E5-2665 + RTX 5060 Ti).
#   lane A  CPU only            (-ngl 0)             : AVX1 Xeon pair = lowest supported CPU
#   lane B  GPU full            (-ngl 99)            : sm_120 test card
#   lane C  RTX-3060 8 GB class (-ngl 99 -ncmoe N)   : lowest supported GPU
#   A/B     AVX1-floor build vs -march=native build of the same commit (cost of the floor)
# Gates run in order: ISA -> device -> coherence -> speed. A fast wrong answer is a failure.
#
# LANE A WARNING: -ngl 0 is NOT CPU-only when a CUDA backend is present. Op-offload defaults to ON and
# moves the batched matmuls to the GPU, which reported an impossible 242-341 t/s "CPU prefill" here. Lane A
# must pass --no-op-offload. Measured cost of forgetting: 242 t/s (GPU doing it) vs 23.9 t/s (real CPU).
#
# LANE C WARNING: do NOT use -fitt as an 8 GB cap. Measured on this box, -fitt 7400 peaked at 8515 MiB and
# -fitt 6400 at 9447 MiB - it fights -ncmoe and ignores the CUDA context. Pin -ncmoe and sample nvidia-smi
# DURING the run; sampling before/after reports ~7 MiB and proves nothing.
set -uo pipefail
ENG=${ENG:-/home/bob/OldIron/engine}
NATIVE=${NATIVE:-/home/bob/OldIron/native-src/build-native/bin}
MODEL=${MODEL:-/mnt/flare/OldIron/weights/Qwen3.6-35B-A3B-IQ3_M.gguf}
OUT=${OUT:-/home/bob/OldIron/results}
GPU_BUDGET_MB=${GPU_BUDGET_MB:-7800}
# Host policy (this box is shared): never use more than 24 of the 32 logical CPUs, so the OS and other work keep
# 8; every measurement runs at nice 19. Thread sweeps above MAXT are skipped rather than clamped silently.
MAXT=${MAXT:-24}
mkdir -p "$OUT"
CLI="$ENG/bin/llama-cli"; BENCH="$ENG/bin/llama-bench"
log(){ echo; echo "### $*"; }
tg(){ grep -E "tg[0-9]+" | grep -oE '[0-9]+\.[0-9]+ ± [0-9.]+' | head -1; }

log "engine identity"; "$CLI" --version 2>&1 | head -2; "$CLI" --list-devices 2>&1 | tail -2

log "ISA gate"; /home/bob/OldIron/scripts/isa-audit.sh "$ENG"/lib/libggml-cpu.so* 2>&1 | tail -9
log "CUDA fatbin (must show sm_86 = RTX 3060 floor AND sm_120 = test card)"
# cuobjdump is NOT on PATH on this box; a bare `cuobjdump` printed nothing and the gate looked like a pass.
CUOBJ=$(ls /usr/local/cuda*/bin/cuobjdump 2>/dev/null | head -1); [ -n "${CUOBJ:-}" ] || CUOBJ=$(command -v cuobjdump || true)
if [ -z "${CUOBJ:-}" ]; then echo "  FAIL: no cuobjdump found - fatbin UNVERIFIED"; else
  # ONE resolved path: handing cuobjdump a symlink plus its target makes it print a bogus mixed arch summary
  # (observed: sm_50..sm_101, 1-2 each) instead of the real 145/145 per arch.
  archs=$("$CUOBJ" --list-elf "$(readlink -f "$ENG/lib/libggml-cuda.so")" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort | uniq -c)
  echo "$archs" | sed 's/^/  /'
  echo "$archs" | grep -q sm_86  || echo "  FAIL: sm_86 (RTX 3060 floor) missing"
  echo "$archs" | grep -q sm_120 || echo "  FAIL: sm_120 (5060 Ti) missing"
fi

# Pre-warm the weights into page cache: on this host the weights live on a loopback ext4 image over NTFS-FUSE,
# where a cold 16.9 GB load outruns the per-run timeout and would look like a coherence FAILURE.
log "warming weights into page cache"
if [ "${SKIP_WARM:-0}" != "1" ]; then t0=$(date +%s); cat "$MODEL" > /dev/null; echo "  warmed in $(( $(date +%s) - t0 ))s"; fi

log "coherence, CPU lane (garbled text fails the build regardless of speed)"
for p in "The capital of France is" "Is 9.9 or 9.11 larger? Answer with the number only:"; do
  echo "--- $p"
  # 200-token budget: this model reasons first, and a 40-token budget is consumed by the thinking preamble
  # before the answer appears, which reads as a coherence failure.
  # Show the ANSWER, not the timing footer: tail -4 caught only "[ Prompt: ... | Generation: ... ]" + "Exiting...",
  # which made a healthy run look like an empty/garbled reply. Strip the footer instead.
  # < /dev/null is REQUIRED: under tmux the CLI gets a pty, so without it the process enters its interactive
  # REPL after the first turn and hangs until the timeout instead of exiting.
  timeout 400 "$CLI" -m "$MODEL" -ngl 0 --no-op-offload -t 16 -c 512 -n 200 --temp 0 -st -p "$p" < /dev/null 2>/dev/null \
    | grep -vE '^\[ Prompt:|^Exiting|^\s*$' | tail -6
done | tee "$OUT/coherence.txt"

log "lane A: CPU only, thread sweep (expect an optimum, not the largest count)"
for t in ${THREADS:-8 16 24}; do
  [ "$t" -gt "$MAXT" ] && continue
  printf '  t=%-3s ' "$t"
  # llama-bench spells it -nopo <0|1>; the bare --no-op-offload flag is llama-cli only and aborts the bench
  timeout 900 nice -n 19 "$BENCH" -m "$MODEL" -ngl 0 -nopo 1 -t "$t" -p 128 -n 48 -r 2 < /dev/null 2>/dev/null | tg | tee -a "$OUT/laneA.txt"; echo
done

log "lane B: GPU maximised (NOT -ngl 99: 15.7 GiB of weights cannot fit a 15.8 GiB card - it OOMs)"
printf '  ncmoe=%s ' "${GPU_FULL_NCMOE:-4}"; timeout 900 "$BENCH" -m "$MODEL" -ngl 99 -ncmoe "${GPU_FULL_NCMOE:-4}" -p 128 -n 48 -r 2 2>/dev/null | tg | tee "$OUT/laneB.txt"

log "lane C: 8 GB class, explicit -ncmoe, VRAM sampled during the run"
for n in ${NCMOE_LIST:-24 32 40}; do
  : > /tmp/vram.$n
  ( for i in $(seq 1 120); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /tmp/vram.$n; sleep 0.5; done ) &
  SA=$!
  r=$(timeout 900 "$BENCH" -m "$MODEL" -ngl 99 -ncmoe "$n" -p 128 -n 48 -r 2 2>/dev/null | tg)
  kill $SA 2>/dev/null; wait $SA 2>/dev/null
  peak=$(sort -n /tmp/vram.$n | tail -1)
  if [ "${peak:-0}" -le "$GPU_BUDGET_MB" ]; then v=FITS; else v="OVER BUDGET"; fi
  printf '  ncmoe=%-3s tg=%-14s peakVRAM=%s MiB  %s\n' "$n" "$r" "$peak" "$v" | tee -a "$OUT/laneC.txt"
done

log "A/B: AVX1 floor (cross-built) vs -march=native, same commit, CPU lane t=16"
if [ -x "$NATIVE/llama-bench" ]; then
  printf '  floor  (OldIron-AVX) : '; timeout 900 "$BENCH" -m "$MODEL" -ngl 0 -nopo 1 -t 16 -p 128 -n 48 -r 2 2>/dev/null | tg
  printf '  native (-march=native): '; timeout 900 "$NATIVE/llama-bench" -m "$MODEL" -ngl 0 -t 16 -p 128 -n 48 -r 2 2>/dev/null | tg
else echo "  (native build not present: $NATIVE/llama-bench)"; fi

log "done $(date -Is) -- results in $OUT"
