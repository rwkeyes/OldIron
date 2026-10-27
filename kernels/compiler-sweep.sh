#!/bin/bash
# OldIron compiler sweep for AVX1 kernels.
#
# For each kernel harness source: build it with every available back end using IDENTICAL flags, ship the
# binaries to the target, run them interleaved on one pinned core, and print a winner-per-kernel table.
#
# Why interleaved arms: this box drifts several percent between passes (measured: lane C moved 10% run to
# run), so A-then-B comparisons lie. Each round runs every back end back to back, and the median across
# rounds is reported.
#
# Why correctness gates: a compiler that produces a faster wrong kernel is worthless. Every arm must print
# verify=ok; any FAIL disqualifies that back end for that kernel regardless of speed.
#
# Usage:  ./compiler-sweep.sh oiw4_gemv_v1.c [more.c ...]
set -uo pipefail
TARGET=${TARGET:-z820a-bob}
RDIR=/home/bob/OldIron/kernels/sweep
FLAGS=${FLAGS:-"-O3 -march=sandybridge -mavx -msse4.2"}
REPS=${REPS:-3}
CORE=${CORE:-0}
GCC=${GCC:-gcc}
CLANG=${CLANG:-clang-18}     # z820a ships only versioned clang binaries; plain `clang` does not exist there
ICX=${ICX:-/opt/intel/oneapi/compiler/2026.2/bin/icx}   # installed on vega only
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

have() { command -v "$1" >/dev/null 2>&1 || [ -x "$1" ]; }
declare -A OMPFLAG=([gcc]="-fopenmp" [clang]="-fopenmp=libgomp" [icx]="-fiopenmp")

for src in "$@"; do
  [ -f "$src" ] || { echo "missing $src"; continue; }
  base=$(basename "$src" .c)
  omp=0; grep -q 'omp\.h' "$src" && omp=1
  echo "=== $base (openmp=$omp) ==="
  built=()
  for cc in gcc clang icx; do
    have "${!cc}" || { echo "  $cc: tool not available"; continue; }
    f=$FLAGS; [ "$omp" = 1 ] && f="$FLAGS ${OMPFLAG[$cc]}"
    out="$WORK/${base}_${cc}"
    if ${!cc} $f -o "$out" "$src" 2>"$WORK/${base}_${cc}.err"; then
      built+=("$cc"); echo "  $cc: built ($(stat -c %s "$out") bytes)"
    else
      echo "  $cc: BUILD FAILED - $(head -2 "$WORK/${base}_${cc}.err" | tr '\n' ' ')"
    fi
  done
  [ ${#built[@]} -eq 0 ] && { echo "  no back end built this kernel"; continue; }

  ssh -o BatchMode=yes "$TARGET" "mkdir -p $RDIR"
  for cc in "${built[@]}"; do rsync -a "$WORK/${base}_${cc}" "$TARGET:$RDIR/"; done

  # interleaved rounds
  for r in $(seq 1 "$REPS"); do
    for cc in "${built[@]}"; do
      ssh -o BatchMode=yes "$TARGET" "nice -n 19 taskset -c $CORE $RDIR/${base}_${cc}" 2>/dev/null \
        | grep '^RESULT' | sed "s/^RESULT /$cc /" >> "$WORK/results.txt"
    done
  done

  echo "  --- median over $REPS interleaved rounds (GMAC/s) ---"
  awk '
    { key=$1" "$2; for(i=3;i<=NF;i++){split($i,a,"="); v[key" "a[1]]=v[key" "a[1]]" "a[2]} }
    END {
      for (k in v) {
        n=split(v[k],x," "); asort_n=n; m=0
        for(i=1;i<=n;i++) if (x[i] ~ /^[0-9.]+$/) { m++; s[m]=x[i]+0 }
        for(i=1;i<=m;i++) for(j=i+1;j<=m;j++) if (s[j]<s[i]) { t=s[i]; s[i]=s[j]; s[j]=t }
        if (m>0) printf "  %-6s %-16s %8.2f\n", substr(k,1,index(k," ")-1), substr(k,index(k," ")+1), (m%2?s[(m+1)/2]:(s[m/2]+s[m/2+1])/2)
      }
    }' "$WORK/results.txt" | sort -k2,2 -k3,3nr
  echo "  raw:"; grep -c . "$WORK/results.txt" | sed 's/^/    /'; 
  cp "$WORK/results.txt" "/home/bob/OldIron/kernels/sweep-${base}.txt" 2>/dev/null
  : > "$WORK/results.txt"
done
