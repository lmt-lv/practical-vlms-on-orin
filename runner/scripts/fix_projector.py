"""undo ModelOpt's INT4 quantization of Cosmos3-Edge's vision
projector in an Edge-LLM checkpoint.

tensorrt-edgellm-quantize 0.11 excludes vision modules by name prefix
(visual, vision_tower, multi_modal_projector, ...); Cosmos3-Edge's projector
is `model.projector`, so it was INT4-quantized with AWQ scales, and the
visual ONNX export then fails. This restores the projector's weights from the
BF16 source (cast to FP16, the checkpoint's 16-bit dtype), drops its
quantization tensors, and adds it to hf_quant_config's exclude_modules: the
checkpoint as if the projector had been excluded. The LLM's quantization is
untouched (AWQ smoothing scales live on the quantized layers themselves).

    runner/edgellm/.venv/bin/python runner/scripts/fix_projector.py \\
        models/edgellm/cosmos3-edge.own-edgellm-awq4 models/Cosmos3-Edge/vision_encoder/model.safetensors
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ck, src = Path(sys.argv[1]), sys.argv[2]
PREFIX = "model.projector."
f = ck / "model.safetensors"
with safe_open(str(f), "pt") as a:
    meta = a.metadata()
    tensors = {k: a.get_tensor(k) for k in a.keys() if not k.startswith(PREFIX)}
with safe_open(src, "pt") as s:
    restored = {k: s.get_tensor(k).to(torch.float16) for k in s.keys() if k.startswith(PREFIX)}
tensors.update(restored)
tmp = f.with_suffix(".tmp")
save_file(tensors, str(tmp), metadata=meta)
tmp.replace(f)
q = json.loads((ck / "hf_quant_config.json").read_text())
ex = q["quantization"].setdefault("exclude_modules", [])
if "model.projector*" not in ex:
    ex.append("model.projector*")
(ck / "hf_quant_config.json").write_text(json.dumps(q, indent=4) + "\n")
cfg = json.loads((ck / "config.json").read_text())
qc = cfg.get("quantization_config")
if qc is not None:
    ig = qc.setdefault("ignore", qc.get("exclude_modules", []))
    for key in ("ignore", "exclude_modules"):
        if key in qc and "model.projector*" not in qc[key]:
            qc[key].append("model.projector*")
    (ck / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print(f"restored {sorted(restored)}; exclude_modules {ex}")
