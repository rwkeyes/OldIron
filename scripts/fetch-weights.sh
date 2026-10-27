#!/bin/bash
# Pull the curated OldIron weight file to z820a over the (high-latency, wifi-routed) link.
# 4 parallel streams, because per-stream throughput is TCP-window limited here (~6.8 MB/s each).
set -uo pipefail
SRC=/media/bob/3d651e2c-e9a4-4758-ba77-863725fe3731/models/Qwen3.6-35B-A3B-IQ3_M.gguf
DSTDIR=/mnt/flare/OldIron/weights
MB=$(( $(stat -c %s "$SRC") / 1048576 + 1 ))
CHUNK=$(( MB / 4 + 1 ))
echo "size ${MB} MB, chunk ${CHUNK} MB"

ssh -o BatchMode=yes z820a-bob "mkdir -p $DSTDIR/parts && rm -f $DSTDIR/parts/p*" || exit 1
for i in 0 1 2 3; do
  ( dd if="$SRC" bs=1M skip=$((i*CHUNK)) count=$CHUNK 2>/dev/null \
      | ssh -o BatchMode=yes z820a-bob "cat > $DSTDIR/parts/p$i" ) &
done
wait
echo "--- chunks landed ---"
ssh -o BatchMode=yes z820a-bob "ls -l $DSTDIR/parts | tail -5"

echo "--- assembling ---"
ssh -o BatchMode=yes z820a-bob "cd $DSTDIR && cat parts/p0 parts/p1 parts/p2 parts/p3 > Qwen3.6-35B-A3B-IQ3_M.gguf && rm -rf parts && ls -l Qwen3.6-35B-A3B-IQ3_M.gguf"
echo "--- hashes (must match) ---"
echo -n "vega : "; sha256sum "$SRC" | cut -d' ' -f1
echo -n "z820a: "; ssh -o BatchMode=yes z820a-bob "sha256sum $DSTDIR/Qwen3.6-35B-A3B-IQ3_M.gguf" | cut -d' ' -f1
