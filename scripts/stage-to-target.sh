#!/bin/bash
# Stage the built engine to a target host with a relocatable RUNPATH, then prove it loads.
# A build dir's RUNPATH is absolute to the build host, so a copied bin/ dies with
# "libllama-*-impl.so: cannot open shared object file" until the rpath is rewritten.
set -uo pipefail
BUILD=${BUILD:-/home/bob/OldIron/build}
DST=${DST:-z820a-bob:/home/bob/OldIron/engine}
TMP=/home/bob/OldIron/stage

rm -rf "$TMP"; mkdir -p "$TMP"
cp -a "$BUILD"/bin "$BUILD"/lib "$TMP"/ 2>/dev/null || { echo "no bin/lib in $BUILD"; exit 1; }

# rewrite RUNPATH to $ORIGIN on every ELF in bin/ and lib/
n=0
for f in "$TMP"/bin/* "$TMP"/lib/*.so*; do
  [ -f "$f" ] || continue
  file -b "$f" | grep -q ELF || continue
  patchelf --set-rpath '$ORIGIN:$ORIGIN/../lib' "$f" 2>/dev/null && n=$((n+1))
done
echo "rpath rewritten on $n ELF files"
rsync -a --delete "$TMP"/ "$DST"/
echo "--- verify on target ---"
ssh -o BatchMode=yes "${DST%%:*}" "export LD_LIBRARY_PATH=/home/bob/OldIron/engine/lib; /home/bob/OldIron/engine/bin/llama-cli --version 2>&1 | head -3; /home/bob/OldIron/engine/bin/llama-cli --list-devices 2>&1 | tail -4"
