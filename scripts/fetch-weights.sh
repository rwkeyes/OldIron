#!/bin/bash
# Pull the weight file to the Z820 over the (high-latency, wifi-routed) link.
# 4 parallel streams, because per-stream throughput is TCP-window limited here (~6.8 MB/s each).
set -uo pipefail
SRC=${SRC:?set SRC to the local copy of the GGUF}
TARGET=${TARGET:?set TARGET to the ssh alias of the machine that will hold the weights}
DSTDIR=${DSTDIR:-/mnt/flare/OldIron/weights}
MB=$(( $(stat -c %s "$SRC") / 1048576 + 1 ))
CHUNK=$(( MB / 4 + 1 ))
echo "size ${MB} MB, chunk ${CHUNK} MB"

ssh -o BatchMode=yes "$TARGET" "mkdir -p $DSTDIR/parts && rm -f $DSTDIR/parts/p*" || exit 1
for i in 0 1 2 3; do
  ( dd if="$SRC" bs=1M skip=$((i*CHUNK)) count=$CHUNK 2>/dev/null \
      | ssh -o BatchMode=yes "$TARGET" "cat > $DSTDIR/parts/p$i" ) &
done
wait
echo "--- chunks landed ---"
ssh -o BatchMode=yes "$TARGET" "ls -l $DSTDIR/parts | tail -5"

echo "--- assembling ---"
ssh -o BatchMode=yes "$TARGET" "cd $DSTDIR && cat parts/p0 parts/p1 parts/p2 parts/p3 > Qwen3.6-35B-A3B-IQ3_M.gguf && rm -rf parts && ls -l Qwen3.6-35B-A3B-IQ3_M.gguf"
echo "--- hashes (must match) ---"
echo -n "source: "; sha256sum "$SRC" | cut -d' ' -f1
echo -n "target: "; ssh -o BatchMode=yes "$TARGET" "sha256sum $DSTDIR/Qwen3.6-35B-A3B-IQ3_M.gguf" | cut -d' ' -f1
