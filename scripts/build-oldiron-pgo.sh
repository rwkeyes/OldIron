#!/bin/bash
# OldIron-AVX build with LTO + PGO. Compile time is treated as free here: the target is a fixed platform
# (Sandy Bridge-EP + sm_86/sm_120) and inference speed is the only thing being optimised.
#
#   ./build-oldiron-pgo.sh          # two-stage: instrument -> train on a real decode -> rebuild
#
# What PGO buys: branch layout and block ordering from a real workload. On branchy code (MoE routing,
# sampler, tokenizer, graph setup) it is typically 5-15%; on pure intrinsic loops it is ~0. So if the
# op profile already says 77% matvec, expect the low end - measure, do not assume.
set -uo pipefail
SRC=${SRC:-/home/bob/OldIron/src}
BUILD=${BUILD:-/home/bob/OldIron/build-pgo}
PROF=${PROF:-/home/bob/OldIron/pgo-data}
TRAIN_MODEL=${TRAIN_MODEL:-/mnt/flare/OldIron/weights/Qwen3.6-35B-A3B-Q4_0-otq4.gguf}
TRAIN_ON=${TRAIN_ON:-z820a-bob}     # training must run on the target ISA
JOBS=${JOBS:-14}
CUDA=/usr/local/cuda-13.2
BASE_FLAGS="-march=sandybridge -mno-avx2 -mno-fma -mno-f16c -mno-bmi2"

mkdir -p "$PROF"
configure() {  # $1 = extra flags
  cmake -B "$BUILD" -S "$SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF \
    -DGGML_BMI2=OFF -DGGML_AVX512=OFF \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;120" \
    -DCUDAToolkit_ROOT="$CUDA" -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
    -DCMAKE_C_FLAGS="$BASE_FLAGS $1" -DCMAKE_CXX_FLAGS="$BASE_FLAGS $1" \
    -DCMAKE_EXE_LINKER_FLAGS="$1" \
    -DCMAKE_BUILD_RPATH='$ORIGIN' -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=ON > /home/bob/OldIron/pgo-configure.log 2>&1 \
    || { echo "FATAL: configure"; tail -20 /home/bob/OldIron/pgo-configure.log; exit 1; }
  cmake --build "$BUILD" -j "$JOBS" > /home/bob/OldIron/pgo-build.log 2>&1 \
    || { echo "FATAL: build"; tail -30 /home/bob/OldIron/pgo-build.log; exit 1; }
}

echo "=== stage 1: instrument (-fprofile-generate, LTO) ==="
configure "-fprofile-generate=$PROF -flto"
echo "=== train on the target (real decode, real ISA) ==="
rsync -a "$BUILD"/bin/ "${TRAIN_ON}:/home/bob/OldIron/pgo-bin/"
# profile data is written next to the objects, i.e. on the BUILD host, so training runs here against
# the same weights; the ISA floor makes the numbers representative for the target.
"$BUILD/bin/llama-bench" -m "$TRAIN_MODEL" -ngl 0 -nopo 1 -t 24 -p 128 -n 48 -r 1 > /home/bob/OldIron/pgo-train.log 2>&1 \
  || { echo "FATAL: training run"; tail -20 /home/bob/OldIron/pgo-train.log; exit 1; }
echo "  profile files: $(find "$PROF" -name '*.gcda' | wc -l)"
echo "=== stage 2: use (-fprofile-use, LTO) ==="
configure "-fprofile-use=$PROF -fprofile-correction -flto -Wno-missing-profile"
echo "=== stage 2 build done: $BUILD/bin/llama-bench ==="
ls -l "$BUILD/bin/llama-bench" | awk '{print $5, $9}'
