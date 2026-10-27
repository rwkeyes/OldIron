#!/bin/bash
# OldIron-AVX ISA gate: prove a CPU backend is AVX1 and nothing above it.
# Usage: isa-audit.sh <file-or-dir>...
#
# Why this file exists: a naive grep for "AVX2 opcodes" gives false positives both ways.
#   - vpmulld / vpblendvb / vpminsd / vpmaxsd / vpslld-q  are SSE4.1 instructions whose VEX.128
#     form is AVX1-legal, so their presence does NOT mean AVX2.
#   - unanchored patterns like  v(p[a-z]*|ps[a-z]*)  match the "vps" inside "vdivps"/"vpermilps".
# The only sound tests are (a) genuinely AVX2-only mnemonics and (b) integer-vector ops on %ymm
# registers - AVX1 has no 256-bit integer arithmetic at all.
set -uo pipefail
fails=0
audit_one() {
  local f="$1" asm
  asm=$(mktemp)
  objdump -d "$f" > "$asm" 2>/dev/null || { echo "  ! cannot disassemble $f"; return; }
  local avx2only ymminteger fma f16c zmm ymm
  avx2only=$(grep -cE '\bvpermd\b|\bpermq\b|\bvpermps\b|\bvpbroadcast[bwdq]\b|\bvpsllvd\b|\bvpsrlvd\b|\bvpsravd\b|\bvpgather[dq][dq]\b|\bvpmaskmov[dq]\b|\bvpblendd\b' "$asm")
  # integer-domain mnemonic with a ymm operand = AVX2, no exceptions on Sandy Bridge
  ymminteger=$(grep -E '\bv(padd|psub|pmul|pmin|pmax|pand|pandn|por|pxor|pcmp|punpck|pack|pshuf|palign|pblend|psll|psrl|psra|pabs|pavg|pmadd|psign|pmovs|pmovz)[a-z0-9]*\b' "$asm" | grep -c '%ymm')
  fma=$(grep -cE '\bvfmadd|\bvfnmadd|\bvfmsub|\bvfnmadd' "$asm")
  f16c=$(grep -c '\bvcvtph2ps\b\|\bvcvtps2ph\b' "$asm")
  zmm=$(grep -c '%zmm' "$asm")
  ymm=$(grep -c '%ymm' "$asm")
  # vm movdq* on ymm IS AVX1 (256-bit moves exist in AVX1), so it is deliberately not counted above.
  echo "  $f"
  printf '    AVX2-only mnemonics : %s\n    int-vector on %%ymm  : %s\n    FMA                 : %s\n    F16C                : %s\n    AVX-512 (%%zmm)      : %s\n    AVX1 %%ymm ops       : %s\n' \
    "$avx2only" "$ymminteger" "$fma" "$f16c" "$zmm" "$ymm"
  if [ "$avx2only" != "0" ] || [ "$ymminteger" != "0" ] || [ "$fma" != "0" ] || [ "$f16c" != "0" ] || [ "$zmm" != "0" ]; then
    echo "    VERDICT: FAIL - above the AVX1 floor"; fails=$((fails+1))
  elif [ "$ymm" = "0" ]; then
    echo "    VERDICT: AVX1-clean but NOT using 256-bit width (performance loss, not a crash)"
  else
    echo "    VERDICT: AVX1-clean and using 256-bit width"
  fi
  rm -f "$asm"
}
for arg in "$@"; do
  if [ -d "$arg" ]; then find "$arg" -name '*.o' -o -name '*.so*' | while read -r f; do audit_one "$f"; done
  else audit_one "$arg"; fi
done
echo "=== $fails file(s) above the AVX1 floor ==="
exit $((fails>0))
