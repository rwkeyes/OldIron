#!/bin/bash
# OldIron-AVX engine build.
#   CPU floor : SSE4.2 + AVX only (Xeon E5-2665 / Z820 gen-1 = lowest supported CPU)
#   CUDA      : sm_86 (RTX 3060 8GB = lowest supported GPU) + sm_120 (RTX 5060 Ti, test card)
#   No AVX2 / FMA / F16C / BMI2 anywhere in the CPU backend.
# Idempotent: re-runnable; a second run just rebuilds changed objects.
set -uo pipefail
SRC=/home/bob/llama.cpp
TREE=/home/bob/OldIron/src
BUILD=/home/bob/OldIron/build
CUDA=/usr/local/cuda-13.2
COMMIT=2744e48d1

echo "=== OldIron-AVX build $(date -Is) ==="
[ -d "$TREE/.git" ] || [ -f "$TREE/.git" ] || git -C "$SRC" worktree add "$TREE" "$COMMIT" || { echo "FATAL: worktree"; exit 1; }
echo "source: $(git -C "$TREE" log -1 --oneline)"

cmake -B "$BUILD" -S "$TREE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF \
  -DGGML_BMI2=OFF -DGGML_AVX512=OFF \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120" \
  -DCUDAToolkit_ROOT="$CUDA" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
  -DCMAKE_C_FLAGS="-march=sandybridge -mno-avx2 -mno-fma -mno-f16c -mno-bmi2" \
  -DCMAKE_CXX_FLAGS="-march=sandybridge -mno-avx2 -mno-fma -mno-f16c -mno-bmi2" \
  -DCMAKE_BUILD_RPATH='$ORIGIN' \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON \
  > "$BUILD.configure.log" 2>&1
if [ ! -f "$BUILD/Makefile" ] && [ ! -f "$BUILD/build.ninja" ]; then
  echo "FATAL: configure failed"; tail -25 "$BUILD.configure.log"; exit 1
fi
echo "configure OK; CPU backend flags reported by cmake:"
grep -E "Adding CPU backend variant|ggml-cpu.*(AVX|FMA|F16C)" "$BUILD.configure.log" | head -4
grep -E "^-- (CUDA|Found CUDAToolkit)" "$BUILD.configure.log" | head -3

cmake --build "$BUILD" -j 14 > "$BUILD.build.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "FATAL: build rc=$rc"; tail -30 "$BUILD.build.log"; exit $rc; fi
echo "=== build OK $(date -Is) ==="
ls -l "$BUILD/bin/llama-cli" "$BUILD/bin/llama-server" 2>/dev/null | awk '{print $5, $9}'
