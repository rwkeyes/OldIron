#!/bin/bash
# Push the weight file to another host over the slow wifi link, 4 parallel streams.
# Per-stream throughput is TCP-window limited here (~6 MB/s), so parallelism is what helps.
#   HOST=<ssh-alias> DSTDIR=/home/bob/OldIron/weights ./push-weights.sh
set -uo pipefail
HOST=${HOST:?set HOST}
DSTDIR=${DSTDIR:?set DSTDIR}
SRC=${SRC:-/media/bob/3d651e2c-e9a4-4758-ba77-863725fe3731/models/Qwen3.6-35B-A3B-IQ3_M.gguf}
MB=$(( $(stat -c %s "$SRC") / 1048576 + 1 )); CHUNK=$(( MB / 4 + 1 ))
echo "=== push $(basename "$SRC") -> $HOST:$DSTDIR ($MB MB) $(date -Is)"
ssh -o BatchMode=yes "$HOST" "mkdir -p $DSTDIR/parts && rm -f $DSTDIR/parts/p*" || exit 1
for i in 0 1 2 3; do
  ( dd if="$SRC" bs=1M skip=$((i*CHUNK)) count=$CHUNK 2>/dev/null | ssh -o BatchMode=yes "$HOST" "cat > $DSTDIR/parts/p$i" ) &
done
wait
ssh -o BatchMode=yes "$HOST" "cd $DSTDIR && cat parts/p0 parts/p1 parts/p2 parts/p3 > $(basename "$SRC") && rm -rf parts && ls -l $(basename "$SRC")"
echo -n "src sha256 : "; sha256sum "$SRC" | cut -d' ' -f1
echo -n "dst sha256 : "; ssh -o BatchMode=yes "$HOST" "sha256sum $DSTDIR/$(basename "$SRC")" | cut -d' ' -f1
echo "=== done $(date -Is)"
