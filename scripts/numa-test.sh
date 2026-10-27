#!/bin/bash
# Does NUMA interleave help the real engine? Only if the weights are in ANONYMOUS memory:
# when the file is served from page cache the pages keep the placement they were faulted with,
# so numactl's policy has no effect. -mm 0 forces a fresh anonymous read under the policy.
cd /home/bob/OldIron/engine || exit 1
M=/mnt/flare/OldIron/weights/Qwen3.6-35B-A3B-Q4_0-otq4.gguf
R=/home/bob/OldIron/results
echo "=== interleave, fresh anon pages (-mm 0) ==="
numactl --interleave=all ./bin/llama-bench -m "$M" -ngl 0 -nopo 1 --load-mode none -t 24 -p 128 -n 48 -r 2 > "$R/numa-interleave-nommap.log" 2>&1
echo "=== default placement, fresh anon pages (-mm 0) ==="
./bin/llama-bench -m "$M" -ngl 0 -nopo 1 --load-mode none -t 24 -p 128 -n 48 -r 2 > "$R/numa-default-nommap.log" 2>&1
echo "=== done ==="
grep -hE "tg48|pp128" "$R/numa-interleave-nommap.log" "$R/numa-default-nommap.log" 2>/dev/null | sed 's/^ *|//'
