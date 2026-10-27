#!/bin/bash
# Does interleaved PLACEMENT of the weights help the real engine?
#
# The earlier attempt was invalid: when the GGUF is already in page cache, `numactl --interleave=all` cannot
# re-place pages that were faulted under a different policy, and `--load-mode none` (anonymous copy) costs ~13%
# which cancels the gain. The only honest way is to materialise a NEW file while the interleave policy is in
# force - the page-cache pages of the copy are allocated by the copying process, so they inherit the policy.
#
# Usage: ./interleave-test.sh <model.gguf>
# Leaves <model>-il.gguf behind (same size again on disk).
set -uo pipefail
TARGET=${TARGET:-z820a-bob}
SRC=${1:?usage: interleave-test.sh <model.gguf>}
IL="${SRC%.gguf}-il.gguf"
RDIR=/home/bob/OldIron/results
THREADS=${THREADS:-24}
NICE="nice -n 19"

echo "=== 1. materialise an interleaved copy (one-off, this is the expensive step) ==="
if ssh -o BatchMode=yes "$TARGET" "[ -f '$IL' ]"; then
  echo "  already exists: $IL"
else
  ssh -o BatchMode=yes "$TARGET" "numactl --interleave=all cp '$SRC' '$IL' && ls -l '$IL'" \
    || { echo "FAILED: copy"; exit 1; }
fi

echo "=== 2. A/B on the engine: default placement vs interleaved copy (same file contents) ==="
for round in 1 2; do
  for tag in default interleaved; do
    [ "$tag" = default ] && M="$SRC" || M="$IL"
    printf '  round %s %-12s ' "$round" "$tag"
    ssh -o BatchMode=yes "$TARGET" "cd /home/bob/OldIron/engine && $NICE ./bin/llama-bench -m '$M' -ngl 0 -nopo 1 -t $THREADS -p 0 -n 48 -r 1" 2>/dev/null \
      | grep tg48 | grep -oE '[0-9]+\.[0-9]+ ± [0-9.]+'
  done
done

echo "=== 3. what the pages actually look like (numastat for the bench's mmap) ==="
ssh -o BatchMode=yes "$TARGET" "numastat -m 2>/dev/null | grep -E 'MemFree|FilePages' | head -4"
