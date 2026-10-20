#!/usr/bin/env bash
# build TensorRT Edge-LLM's C++ tools (llm_build, visual_build,
# llm_inference, llm_bench) for the Orin at a pinned tag, into
# build/edge-llm (gitignored). The Python side is the published wheel in
# runner/edgellm/ (uv sync --group build also brings the kernel generator).
#
#   runner/scripts/build_edgellm.sh [jobs]
#
# Uses JetPack 7.2.1's CUDA 13.2 and TensorRT 10.16.2 (/usr). The CuTe DSL
# kernels are generated on the board: the v0.11.0 release ships no prebuilt
# artifact.
set -euo pipefail
cd "$(dirname "$0")/../.."

TAG=v0.11.0   # 95515c2f87fba8982db5a519f9022277667b3cc9
SM=${CUTE_SM:-sm_87}
TARGET=${EMBEDDED_TARGET:-jetson-orin}

main() {
    local jobs=${1:-8} src=build/edge-llm py=$PWD/runner/edgellm/.venv/bin/python
    [[ -d $src/.git ]] || git clone https://github.com/NVIDIA/TensorRT-Edge-LLM "$src"
    git -C "$src" fetch --quiet --tags origin
    git -C "$src" checkout --quiet "$TAG"
    git -C "$src" submodule update --init --recursive
    (cd runner/edgellm && uv sync --locked --group build)
    if [[ ! -f $src/cpp/kernels/cuteDSLArtifact/aarch64/$SM/metadata.json ]]; then
        (cd "$src" && "$py" kernelSrcs/build_cutedsl.py --kernels ALL --gpu_arch "$SM" \
            --arch aarch64 --cuda-version 13 --clean)
    fi
    cmake -S "$src" -B "$src/build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DTRT_PACKAGE_DIR=/usr \
        -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
        -DEMBEDDED_TARGET="$TARGET" \
        -DCUDA_CTK_VERSION=13.2 \
        -DENABLE_CUTE_DSL=ALL
    cmake --build "$src/build" -j "$jobs"
    ls "$src/build/examples/llm/" "$src/build/examples/multimodal/" | grep -E '^(llm|visual)_' || true
    echo "$TAG $SM $TARGET" > "$src/build/.built"
}
main "$@"; exit
