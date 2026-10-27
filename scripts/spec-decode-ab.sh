#!/bin/bash
# Speculative-decoding A/B on the 35B weights (single stream).
# For a 3B-active MoE the verify pass reads the weights ONCE for several tokens, so this is the one remaining
# lever that cuts bytes-per-token without needing concurrent requests. Payoff depends entirely on acceptance
# rate, so this measures generated tokens/s with and without the draft, same prompt, temp 0.
cd /home/bob/OldIron/engine || exit 1
T=/mnt/flare/OldIron/weights/Qwen3.6-35B-A3B-Q4_0-otq4.gguf
D=/home/bob/OldIron/drafts/Qwen3.5-0.8B-Q4_K_M.gguf
P="Write a Python function that reverses a singly linked list, then explain the time complexity."
N=96
R=/home/bob/OldIron/results

echo "=== baseline: no draft, t=24 ==="
nice -n 19 ./bin/llama-cli -m "$T" -ngl 0 --no-op-offload -t 24 -c 1024 -n $N --temp 0 -st -p "$P" </dev/null 2>&1 \
  | grep -E "^\[ Prompt:|Generation:" | tail -1

echo "=== speculative: draft Qwen3.5-0.8B Q4, t=24 ==="
nice -n 19 ./bin/llama-cli -m "$T" -md "$D" -ngl 0 --no-op-offload -t 24 -c 1024 -n $N --temp 0 -st -p "$P" </dev/null 2>&1 \
  | grep -E "^\[ Prompt:|Generation:|draft|accept" | tail -4

echo "=== done ==="
