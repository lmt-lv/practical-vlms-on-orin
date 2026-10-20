#!/usr/bin/env bash
# TensorRT Edge-LLM engines for an edge-llm checkpoint tag
# (runner/checkpoints.json), sized for the 16-frame contract, into
# models/edgellm/<tag>.engines/{onnx,llm,visual}, with edgellm.json (versions,
# arguments, the checkpoint's sha256) that runs record. The checkpoint itself
# is models/edgellm/<tag>/ (runner/scripts/quantize_edgellm.sh, or a
# downloaded ModelOpt checkpoint).
#
#   runner/scripts/edgellm_engines.sh <tag>
#   EXPORT_ARGS="--task reasoning" runner/scripts/edgellm_engines.sh cosmos3-edge...   # per-model export flags
#
# Limits: one request at a time (--maxBatchSize 1; MAX_BATCH=12 for the
# batched cell, a video's 12 questions in one request); 4,096 input tokens (a
# 16-frame prompt is ~2.6K; NVIDIA's 2,048 is too short) and a 5,120-token
# KV cache (input plus the contract's 1,024-token cap); NVIDIA's visual
# limits (16,384 image tokens, 2,048 per image).
set -euo pipefail
cd "$(dirname "$0")/../.."

MAX_INPUT=${MAX_INPUT:-4096}
MAX_KV=${MAX_KV:-5120}
MAX_BATCH=${MAX_BATCH:-1}

main() {
    # ENGINES_DIR: another power mode's engines (TensorRT plans are per
    # device, and MODE_30W's gated GPU counts as another one), e.g.
    # models/edgellm/<tag>.engines-30w; the ONNX export is shared.
    local tag=$1 ck=models/edgellm/$1 out=${ENGINES_DIR:-models/edgellm/$1.engines} ex=build/edge-llm/build/examples
    [[ -f $ck/config.json ]] || { echo "!! $ck has no checkpoint"; return 1; }
    export PATH=$PWD/runner/edgellm/.venv/bin:$PATH
    export EDGELLM_PLUGIN_PATH=$PWD/build/edge-llm/build/libNvInfer_edgellm_plugin.so
    mkdir -p "$out"
    local export_args=(--externalize-weights int4_ffn ${EXPORT_ARGS:-})
    [[ -d $out/onnx/llm ]] || tensorrt-edgellm-export "$ck" "$out/onnx" "${export_args[@]}" > "$out/export.log" 2>&1
    local llm_args=(--maxBatchSize "$MAX_BATCH" --maxInputLen "$MAX_INPUT" --maxKVCacheCapacity "$MAX_KV")
    local vis_args=(--minImageTokens 8 --maxImageTokens 16384 --maxImageTokensPerImage 2048)
    [[ -f $out/llm/config.json ]] || "$ex/llm/llm_build" --onnxDir "$out/onnx/llm" --engineDir "$out/llm" "${llm_args[@]}" > "$out/llm_build.log" 2>&1
    [[ -d $out/visual ]] || "$ex/multimodal/visual_build" --onnxDir "$out/onnx/visual" --engineDir "$out" "${vis_args[@]}" > "$out/visual_build.log" 2>&1
    local sha
    sha=$(sha256sum "$ck"/model*.safetensors | awk '{print $1}' | sha256sum | awk '{print $1}')
    python3 - "$tag" "$sha" "${export_args[*]}" "${llm_args[*]}" "${vis_args[*]}" > "$out/edgellm.json" <<'PY'
import json, sys, importlib.metadata as md
tag, sha, exp, llm, vis = sys.argv[1:]
print(json.dumps({"tag": tag, "edgellm": "0.11.0 (95515c2f)", "checkpoint_sha256": sha,
                  "export_args": exp.split(), "llm_build_args": llm.split(), "visual_build_args": vis.split()}, indent=2))
PY
    echo "engines for $tag in $out"
}
main "$@"; exit
