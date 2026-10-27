#!/bin/bash
# OldIron: pick the GPU/CPU split that maximises throughput at a long context.
# -ncmoe N = keep N layers' MoE weights on the CPU, rest on the GPU. Smaller N = faster but less VRAM
# left for the KV cache, so the choice is a budget question, not just a speed question.
# VRAM is sampled DURING each run: the number llama.cpp prints at startup does not include the CUDA
# context or the KV allocation, and -fitt has already been shown to lie by up to 2 GB.
cd /home/bob/OldIron/engine || exit 1
M=/mnt/flare/OldIron/weights/Qwen3.6-35B-A3B-Q4_0-otq4.gguf
CTX=${CTX:-32768}
THREADS=${THREADS:-24}
R=/home/bob/OldIron/results

echo "=== long-context GPU split sweep: ctx=$CTX threads=$THREADS ==="
free_vram=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}')
echo "free VRAM before: ${free_vram} MiB"

for n in "$@"; do
    ( for i in $(seq 1 240); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done > /tmp/v.$n.samples ) & SA=$!
    # NOTE: llama-bench has no -c flag (it sizes the context from -p/-n), so KV pressure is checked
    # separately below with llama-cli at a real $CTX.
    out=$(nice -n 19 ./bin/llama-bench -m "$M" -ngl 99 -ncmoe "$n" -nopo 1 \
            -t "$THREADS" -p 4096 -n 48 -r 1 2>&1)
    kill $SA 2>/dev/null; wait $SA 2>/dev/null
    pp=$(echo "$out" | grep "pp4096" | grep -oE "[0-9]+\.[0-9]+ ± [0-9.]+" | head -1)
    tg=$(echo "$out" | grep "tg48"   | grep -oE "[0-9]+\.[0-9]+ ± [0-9.]+" | head -1)
    peak=$(sort -n /tmp/v.$n.samples 2>/dev/null | tail -1)
    kvleft=$(( 15847 - ${peak:-0} ))
    printf "ncmoe=%-3s pp4096=%-16s tg48=%-16s peakVRAM=%s MiB  leftForKV=%s MiB\n" \
        "$n" "${pp:-FAIL}" "${tg:-FAIL}" "${peak:-?}" "$kvleft"
done
echo "=== KV headroom at ctx=$CTX for the fastest arm that loaded ==="
best=$(grep -oE "ncmoe=[0-9]+" <<< "$(sort -t= -k4 /dev/null)" 2>/dev/null | head -1)
for n in "$@"; do
    ( for i in $(seq 1 90); do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done > /tmp/k.$n.samples ) & SA=$!
    # load-only at the real context: llama-cli with -n 1 allocates the whole KV cache
    nice -n 19 ./bin/llama-cli -m "$M" -ngl 99 -ncmoe "$n" -c "$CTX" -n 1 -p "hi" -t "$THREADS" \
        --no-warmup < /dev/null > /tmp/k.$n.log 2>&1
    rc=$?
    kill $SA 2>/dev/null; wait $SA 2>/dev/null
    peak=$(sort -n /tmp/k.$n.samples 2>/dev/null | tail -1)
    printf "ctx=%-6s ncmoe=%-3s rc=%s peakVRAM=%s MiB\n" "$CTX" "$n" "$rc" "${peak:-?}"
done
echo "=== done ==="
