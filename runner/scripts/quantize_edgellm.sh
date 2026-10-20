#!/usr/bin/env bash
# an Edge-LLM (ModelOpt) INT4 AWQ checkpoint of a model, made on
# the board with NVIDIA's defaults (128 calibration samples, FP16, vision
# tower unquantized), into models/edgellm/<name>/. Writes <name>/.done.
#
#   runner/scripts/quantize_edgellm.sh <hf model dir> <name>
#   systemd-run --user --unit=tar-quant -- runner/scripts/quantize_edgellm.sh \
#       models/Qwen3-VL-8B-Instruct qwen3-vl-8b-int4awq          # detached
#
# The venv's bin/ must be on PATH: ModelOpt builds its CUDA quantization
# kernels with ninja from PATH, and falls back to CPU ones without it.
# Takes ~1.5 h for Qwen3-VL-8B on a 32 GB Orin in MAXN (25 GB resident).
set -euo pipefail
cd "$(dirname "$0")/../.."

main() {
    local src=$1 name=$2 out=models/edgellm/$2 log=runs/quantize_$2.log
    mkdir -p models/edgellm runs
    [[ -f $out/.done ]] && { echo "$out already made"; return 0; }
    rm -rf "$out"
    # QUANT_ENV=edgellm-quant: Edge-LLM's tools on transformers 5.17, for
    # models its pinned 5.14.1 cannot load (Cosmos3-Edge).
    export PATH=$PWD/runner/${QUANT_ENV:-edgellm}/.venv/bin:$PATH
    { echo "start $(date -Is) used=$(free -m | awk '/Mem:/{print $3}')MB"
      tensorrt-edgellm-quantize llm --model_dir "$src" --output_dir "$out" \
          --quantization int4_awq --dtype fp16
      echo "end $(date -Is) used=$(free -m | awk '/Mem:/{print $3}')MB"; } > "$log" 2>&1
    touch "$out/.done"
}
main "$@"; exit
