#!/usr/bin/env bash
# build llama.cpp with CUDA for the Orin (SM 8.7) at a pinned
# commit, into build/llama.cpp (gitignored). Uses the host's CUDA
# (/usr/local/cuda, 13.2 on JetPack 7.2.1). Re-running rebuilds
# incrementally; a different pin re-checks-out.
#
#   runner/scripts/build_llama_cpp.sh [jobs]
#
# GGML_CUDA_NO_VMM: llama.cpp reserves a fixed 32 GB virtual-memory pool,
# which crashes image requests on Orin (llama.cpp issue #29142).
# Don't set CUDA_SCALE_LAUNCH_QUEUES (jetson-containers sets 4x): it hung
# llama-server on Orin with JetPack 6 (issue #19219).
set -euo pipefail
cd "$(dirname "$0")/../.."

COMMIT=05af0d2b1398394cfa67e1918fee7feabccaa9bc   # master, 2026-09-30 (b11284-18)
ARCH=${CUDA_ARCH:-87}   # 87 = Orin; 80 = A100; 120 = Blackwell workstation GPUs

main() {
    local jobs=${1:-6} src=build/llama.cpp
    [[ -d $src/.git ]] || git clone https://github.com/ggml-org/llama.cpp "$src"
    git -C "$src" fetch --quiet origin "$COMMIT" 2>/dev/null || git -C "$src" fetch --quiet origin
    git -C "$src" checkout --quiet "$COMMIT"
    cmake -S "$src" -B "$src/build-sm$ARCH" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
        -DGGML_CUDA_NO_VMM=ON \
        -DCMAKE_CUDA_COMPILER="${CUDACXX:-/usr/local/cuda/bin/nvcc}" \
        -DLLAMA_CURL=OFF
    cmake --build "$src/build-sm$ARCH" -j "$jobs" --target llama-server llama-mtmd-cli llama-bench
    "$src/build-sm$ARCH/bin/llama-server" --version
    echo "$COMMIT sm$ARCH" > "$src/build-sm$ARCH/.built"
}
main "$@"; exit
