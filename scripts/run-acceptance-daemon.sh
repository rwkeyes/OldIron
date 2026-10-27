#!/bin/bash
# Entry point for the detached acceptance run (invoked by tmux on the target host).
cd /home/bob/OldIron || exit 1
mkdir -p results
exec env SKIP_WARM=1 THREADS="${THREADS:-16 24}" NCMOE_LIST="${NCMOE_LIST:-24 32}" \
     bash scripts/acceptance-hp-z820.sh > results/run.log 2>&1
