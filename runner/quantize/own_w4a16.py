"""Our own W4A16 checkpoints, for models with no community build, and W8A16
ones for the weight-precision setting of the deployment sweep (tags ending in
-w8a16: the same method at 8 bits).

    cd runner/quantize && uv run python own_w4a16.py <tag> [<tag> ...]

Data-free round-to-nearest (RTN) W4A16 with llm-compressor's model_free_ptq:
int4 weights, group 32, symmetric, 16-bit activations, which vLLM runs with
Marlin on SM 8.0 and 8.7. Group 32, not the W4A16 preset's 128: at 128,
North Micro kept only 6 of 15 bcq/mcq answers on five dev videos, and a
BF16 "fake quant" of the same weights showed the loss is the rounding's, not
vLLM's. model_free_ptq works
on the safetensors files directly, without the model's code: no transformers
version conflict (Molmo2's code needs 4.57), no tracing, and the checkpoint
keeps its own tensor names and folder layout, which is what vLLM already
loads for the BF16 model. Only the language model's decoder linear layers are
quantized; the vision tower, projector, embeddings, lm_head and generation-
only weights stay 16-bit, and the quantization config lists them as ignored,
by patterns that match both the checkpoint's names and vLLM's.

RTN rather than GPTQ/AWQ: data-free and deterministic, and on these models
within the noise of the community builds made with calibration data.

Writes models/own/<tag>/ (the quantized weights plus the BF16 repo's other
files) and models/own/<tag>/quantize.json (method, tool versions, patterns,
the git commit, each weight file's sha256). `python -m tar_runner.weights
fetch <tag>` then builds the run directory from it.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
MANIFEST = REPO / "runner" / "checkpoints.json"

#: Per checkpoint: the BF16 source, the folder holding the language model's
#: weights (quantized), folders copied as they are, and what stays 16-bit.
SPECS = {
    "north-micro.own-w4a16": {
        "source": "North-Micro-Vision-Instruct", "quantize_dir": ".", "keep_dirs": [],
        "ignore": ["re:.*visual.*", "re:.*embed_tokens.*", "re:.*lm_head.*"],
    },
    "molmo2-8b.own-w4a16": {
        "source": "Molmo2-8B", "quantize_dir": ".", "keep_dirs": [],
        "ignore": ["re:.*vision_backbone.*", "re:.*wte.*", "re:.*lm_head.*", "re:.*ln_f.*"],
    },
    # The reasoner tower's weights are in transformer/, the vision encoder in
    # vision_encoder/ (kept 16-bit); VAE, scheduler and assets are not needed.
    # Generation-only tensors in transformer/ (vLLM drops them) stay as they are.
    "cosmos3-edge.own-w4a16": {
        "source": "Cosmos3-Edge", "quantize_dir": "transformer", "keep_dirs": ["vision_encoder"],
        "ignore": ["re:.*visual.*", "re:.*projector.*", "re:.*embed_tokens.*", "re:.*lm_head.*",
                   "re:.*_moe_gen.*", "re:.*add_[qkv]_proj.*", "re:.*to_add_out.*", "re:.*proj_in.*",
                   "re:.*proj_out.*", "re:.*time_embedder.*", "re:.*action_.*", "re:.*norm.*"],
    },
}
# The deployment sweep's W8A16 setting, for the models with no community
# 8-bit build that can be in the sweep (the Cosmos3 models can't). The same tensors as each model's INT4 build stay 16-bit; for
# LFM2.5, the community INT4 build's own list (sahilchachra-awq).
SPECS["north-micro.own-w8a16"] = {**SPECS["north-micro.own-w4a16"], "bits": 8}
SPECS["molmo2-8b.own-w8a16"] = {**SPECS["molmo2-8b.own-w4a16"], "bits": 8}
SPECS["lfm2.5-vl-3b.own-w8a16"] = {
    "source": "LFM2.5-VL-3B", "quantize_dir": ".", "keep_dirs": [], "bits": 8,
    "ignore": ["re:.*vision_tower.*", "re:.*multi_modal_projector.*", "re:.*embed_tokens.*",
               "re:.*lm_head.*", r"re:.*layers\.\d+\.(short_)?conv\.(in|out)_proj",
               # The short convolutions' kernels are 3-D, not linear layers.
               r"re:.*\.conv\.conv.*"],
}
GROUP_SIZE = 32


def scheme(bits: int = 4):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    return QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
        num_bits=bits, type="int", symmetric=True, strategy="group", group_size=GROUP_SIZE))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def make(tag: str) -> Path:
    from llmcompressor import model_free_ptq
    spec = SPECS[tag]
    src, out = MODELS / spec["source"], MODELS / "own" / tag
    if (out / "quantize.json").exists():
        print(f"{tag}: already made ({out})")
        return out
    if out.exists():
        shutil.rmtree(out)
    qdir = src / spec["quantize_dir"]
    qout = out / spec["quantize_dir"]
    bits = spec.get("bits", 4)
    model_free_ptq(str(qdir), str(qout), scheme=scheme(bits), ignore=spec["ignore"],
                   device=os.environ.get("QUANT_DEVICE"))
    for d in spec["keep_dirs"]:   # relative, so the folder can be copied to another host
        (out / d).symlink_to(os.path.relpath(src / d, out), target_is_directory=True)
    if spec["quantize_dir"] != ".":
        # vLLM reads the quantization config from the top-level config.json.
        top = json.loads((src / "config.json").read_text())
        top["quantization_config"] = json.loads((qout / "config.json").read_text())["quantization_config"]
        (out / "config.json").write_text(json.dumps(top, indent=2) + "\n")

    weights = sorted(p for p in out.rglob("*.safetensors") if not p.is_symlink()
                     and not any(p.is_relative_to(out / d) for d in spec["keep_dirs"]))
    record = {
        "tag": tag, "made": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": "RTN (llm-compressor model_free_ptq, data-free)",
        "scheme": {"weights": f"int{bits}", "symmetric": True, "group_size": GROUP_SIZE, "activations": "16-bit"},
        "source": {"dir": f"models/{spec['source']}", **source_revision(src)},
        "quantized_dir": spec["quantize_dir"], "kept_16bit_dirs": spec["keep_dirs"],
        "ignore": spec["ignore"],
        "tools": {p: version(p) for p in ("llmcompressor", "compressed-tensors", "transformers", "torch")},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                     text=True).stdout.strip(),
        "weights_sha256": {str(p.relative_to(out)): sha256(p) for p in weights},
    }
    (out / "quantize.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{tag}: {out} ({sum(p.stat().st_size for p in weights) / 1e9:.2f} GB of quantized weights)")
    return out


def source_revision(src: Path) -> dict:
    """The BF16 source's pinned revision, from the checkpoint manifest's model."""
    marker = next(src.glob(".fetched-*"), None)
    return {"revision": marker.name.removeprefix(".fetched-") if marker else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", choices=sorted(SPECS))
    args = ap.parse_args()
    for tag in args.tags:
        make(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
