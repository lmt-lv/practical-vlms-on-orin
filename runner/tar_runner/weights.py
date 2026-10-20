"""Model weights: the BF16 references and the other checkpoints (INT4, INT8, GGUF).

    uv run python -m tar_runner.weights base <model> [<model> ...]   # BF16, at the pinned revision
    uv run python -m tar_runner.weights list
    uv run python -m tar_runner.weights fetch <tag> [<tag> ...]
    uv run tar-run --model qwen3-vl-8b --weights <tag> --dtype float16 ...

Checkpoints are listed in runner/checkpoints.json by tag, each with its
model (an adapter key), runtime, repo and pinned revision. `fetch`
downloads one into models/quant-src/<author>--<name>/ and builds
models/quant/<tag>/, the directory vLLM loads (llama.cpp checkpoints: their
listed GGUF files, into models/gguf/<tag>/):
  - from the quantized repo: its weights, config.json (with its
    quantization_config) and anything else that is not on the input side;
  - from the adapter's own BF16 repo, linked in place of the quantized
    repo's copies: the input side (tokenizer, processor and its code, chat
    template, generation config). An INT4 run's inputs are then
    byte-identical to the BF16 run's, and only the weights differ.
The input-side files that differed between the two repos are recorded in
models/quant/<tag>/overlay.json, next to the quantization summary that runs
record in their configuration.

A checkpoint saved under names vLLM's model class does not map (the Cosmos3
classes load only NVIDIA's own layout: `layers.N...`, `blocks...`, from a
`transformer/` folder) can carry a `layout` in checkpoints.json: prefix
renames, applied to a copy of its safetensors and to its quantization
config's ignore list, and the subfolder its weights go in. That is a
conversion of names only; the tensors are the checkpoint's own. Runs record
the layout with the rest of the weights configuration.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
from functools import cache
from pathlib import Path

from .adapters import MODELS
from .data import MODELS_ROOT, REPO_ROOT

MANIFEST = REPO_ROOT / "runner" / "checkpoints.json"
SRC_ROOT = MODELS_ROOT / "quant-src"
#: Our own quants (runner/quantize/own_w4a16.py), made on one host and
#: copied to the others.
OWN_ROOT = MODELS_ROOT / "own"
OVERLAY_ROOT = MODELS_ROOT / "quant"
#: llama.cpp checkpoints: the listed GGUF files (weights and mmproj), as shipped.
GGUF_ROOT = MODELS_ROOT / "gguf"
#: TensorRT Edge-LLM checkpoints (ModelOpt) in <tag>/, their engines in
#: <tag>.engines/ (runner/scripts/edgellm_engines.sh).
EDGELLM_ROOT = MODELS_ROOT / "edgellm"
#: Always taken from the BF16 repo: everything that turns a request into
#: model inputs, and the generation defaults.
INPUT_SIDE = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
              "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt", "*.tiktoken",
              "chat_template.jinja", "chat_template.json", "preprocessor_config.json",
              "processor_config.json", "video_preprocessor_config.json", "generation_config.json",
              "*.py")
WEIGHTS = ("*.safetensors", "*.safetensors.index.json", "*.bin", "*.pt", "*.pth", "*.gguf")


@cache
def manifest() -> dict:
    return json.loads(MANIFEST.read_text())["checkpoints"]


def entry(tag: str) -> dict:
    if tag not in manifest():
        raise SystemExit(f"unknown checkpoint {tag!r}; see {MANIFEST}")
    return manifest()[tag]


def overlay_dir(tag: str) -> Path:
    return OVERLAY_ROOT / tag


def gguf_dir(tag: str) -> Path:
    return GGUF_ROOT / tag


def local_path(tag: str) -> Path:
    """Where a checkpoint's runtime loads it from."""
    rt = entry(tag)["runtime"]
    if rt == "llama.cpp":
        return gguf_dir(tag)
    if rt == "edge-llm":
        return EDGELLM_ROOT / tag
    return overlay_dir(tag)


def build_edgellm(tag: str) -> Path:
    """An Edge-LLM checkpoint: our own is made on the board
    (runner/scripts/quantize_edgellm.sh); a Hub one is downloaded without
    its ONNX exports, which must come from the Edge-LLM version that builds
    the engines. Engines: runner/scripts/edgellm_engines.sh <tag>."""
    from huggingface_hub import snapshot_download
    e = entry(tag)
    out = EDGELLM_ROOT / tag
    if e["role"] != "own":
        snapshot_download(e["repo"], revision=e["revision"], local_dir=out, ignore_patterns=["onnx/*"])
    if not (out / "config.json").exists():
        raise SystemExit(f"{tag}: no checkpoint in {out}")
    print(f"{tag}: {out}; engines: runner/scripts/edgellm_engines.sh {tag}")
    return out


def build_gguf(tag: str) -> Path:
    """A llama.cpp checkpoint: its listed files, downloaded at the pinned
    revision, with their sha256 recorded in gguf.json. No input side is
    overlaid: a GGUF carries its own tokenizer, and the backend renders the
    prompt with the adapter's own chat template."""
    from huggingface_hub import hf_hub_download
    e = entry(tag)
    out = gguf_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in e["files"]:
        path = out / name
        if not path.exists() and (twin := _gguf_twin(e, name)):
            # Another entry has the same file of the same repo and revision
            # (the Q8_0-mmproj setting shares its weights file): link, don't
            # download it twice.
            path.hardlink_to(twin)
        else:
            path = Path(hf_hub_download(e["repo"], name, revision=e["revision"], local_dir=out))
        files[name] = _sha(path)
    rec = {"tag": tag, "repo": e["repo"], "revision": e["revision"], "files": files}
    (out / "gguf.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"{tag}: {out} ({', '.join(files)})")
    return out


def _gguf_twin(e: dict, name: str) -> Path | None:
    """The same file, already fetched for another llama.cpp entry."""
    for other, o in manifest().items():
        if o["runtime"] == "llama.cpp" and o.get("repo") == e["repo"] and o.get("revision") == e["revision"] \
                and name in o.get("files", []) and (gguf_dir(other) / "gguf.json").exists():
            return gguf_dir(other) / name
    return None


def _matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _sha(path: Path) -> str:
    """Streamed: a GGUF can be most of the board's memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def quantization_summary(config: dict) -> dict:
    """What a run records about its weights' quantization."""
    qc = config.get("quantization_config") or (config.get("text_config") or {}).get("quantization_config") or {}
    out = {"method": qc.get("quant_method")}
    groups = qc.get("config_groups") or {}
    if groups:
        w = next(iter(groups.values())).get("weights") or {}
        out |= {"format": qc.get("format"), "bits": w.get("num_bits"), "group_size": w.get("group_size"),
                "symmetric": w.get("symmetric")}
    else:
        out |= {"bits": qc.get("bits"), "group_size": qc.get("group_size")}
    return out


def _renamed(src: Path, rename: list[list[str]]) -> Path:
    """A copy of src's weights under renamed prefixes, made once."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    key = hashlib.sha256(json.dumps(rename).encode()).hexdigest()[:8]
    dst = src.with_name(f"{src.name}--renamed-{key}")
    if (dst / ".done").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            meta = fh.metadata()
            tensors = {_rename(k, rename): fh.get_tensor(k) for k in fh.keys()}
        save_file(tensors, str(dst / f.name), metadata=meta)
        del tensors
    index = src / "model.safetensors.index.json"
    if index.exists():
        j = json.loads(index.read_text())
        j["weight_map"] = {_rename(k, rename): v for k, v in j["weight_map"].items()}
        (dst / index.name).write_text(json.dumps(j, indent=2) + "\n")
    (dst / ".done").touch()
    return dst


def _rename(name: str, rename: list[list[str]]) -> str:
    for old, new in rename:
        if name.startswith(old):
            return new + name[len(old):]
    return name


def build(tag: str) -> Path:
    from huggingface_hub import snapshot_download
    e = entry(tag)
    if e["runtime"] == "llama.cpp":
        return build_gguf(tag)
    if e["runtime"] == "edge-llm":
        return build_edgellm(tag)
    if e["runtime"] != "vllm":
        raise SystemExit(f"{tag}: runtime {e['runtime']} is not built here (vllm, llama.cpp, edge-llm)")
    adapter = MODELS[e["model"]]
    bf16 = Path(adapter.local_dir)
    if not (bf16 / "config.json").exists():
        raise SystemExit(f"{tag}: the BF16 model {bf16} is not fetched")
    if e["role"] == "own":
        src = OWN_ROOT / tag
        if not (src / "quantize.json").exists():
            raise SystemExit(f"{tag}: not made; run runner/quantize/own_w4a16.py {tag}")
    else:
        src = SRC_ROOT / e["repo"].replace("/", "--")
        snapshot_download(e["repo"], revision=e["revision"], local_dir=src)

    layout = e.get("layout") or {}
    rename = layout.get("rename") or []
    weights_src = _renamed(src, rename) if rename else src
    out = overlay_dir(tag)
    if out.exists():   # links and small files only: rebuilt from scratch
        shutil.rmtree(out)
    out.mkdir(parents=True)
    wdir = out / layout.get("weights_subdir", "")
    wdir.mkdir(exist_ok=True)
    differed, from_bf16, dropped = [], [], []
    for f in sorted(src.iterdir()):
        if f.name.startswith("."):
            continue
        if f.is_dir():   # weight folders (a transformer/ layout): linked whole
            (out / f.name).symlink_to(f.resolve(), target_is_directory=True)
            continue
        if _matches(f.name, INPUT_SIDE):   # the input side is the BF16 repo's alone
            if not (bf16 / f.name).exists():
                dropped.append(f.name)
            continue
        if _matches(f.name, WEIGHTS):
            (wdir / f.name).symlink_to((weights_src / f.name).resolve())
        elif f.name == "config.json" and rename:
            cfg = json.loads(f.read_text())
            qc = cfg.get("quantization_config") or {}
            if qc.get("ignore"):
                qc["ignore"] = [i if i.startswith("re:") else _rename(i, rename) for i in qc["ignore"]]
            (out / f.name).write_text(json.dumps(cfg, indent=2) + "\n")
        else:
            (out / f.name).symlink_to(f.resolve())
    for f in sorted(bf16.iterdir()):
        if not f.is_file() or f.name.startswith(".") or _matches(f.name, WEIGHTS) or f.name == "config.json":
            continue
        if _matches(f.name, INPUT_SIDE) or not (src / f.name).exists():
            if (src / f.name).exists() and _sha(src / f.name) != _sha(f):
                differed.append(f.name)
            (out / f.name).symlink_to(f.resolve())
            from_bf16.append(f.name)
    config = json.loads((src / "config.json").read_text())
    bf16_config = json.loads((bf16 / "config.json").read_text())
    config_diff = sorted(k for k in set(config) | set(bf16_config)
                         if k not in ("quantization_config", "transformers_version", "torch_dtype", "dtype")
                         and config.get(k) != bf16_config.get(k))
    record = {"tag": tag, **e, "bf16": {"hf_id": adapter.hf_id, "revision": adapter.revision},
              "quantization": quantization_summary(config),
              "input_side_from_bf16": from_bf16, "input_side_differed": differed,
              "input_side_dropped": dropped, "layout": layout or None,
              "config_keys_differing_from_bf16": config_diff}
    (out / "overlay.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{tag}: {out} ({len(from_bf16)} files from BF16; differed: {differed or 'none'}; "
          f"dropped: {dropped or 'none'}; config keys differing: {config_diff or 'none'})")
    return out


def run_config(tag: str) -> dict:
    """The weights part of a run's configuration (and so of its hash)."""
    e = entry(tag)
    if e["runtime"] == "edge-llm":
        path = EDGELLM_ROOT / f"{tag}.engines" / "edgellm.json"
        if not path.exists():
            raise SystemExit(f"{tag}: no engines; run runner/scripts/edgellm_engines.sh {tag}")
        return json.loads(path.read_text())
    if e["runtime"] == "llama.cpp":
        path = gguf_dir(tag) / "gguf.json"
        if not path.exists():
            raise SystemExit(f"{tag}: not fetched; run `python -m tar_runner.weights fetch {tag}`")
        return json.loads(path.read_text())
    path = overlay_dir(tag) / "overlay.json"
    if not path.exists():
        raise SystemExit(f"{tag}: not fetched; run `python -m tar_runner.weights fetch {tag}`")
    rec = json.loads(path.read_text())
    cfg = {"tag": tag, "repo": e["repo"], "revision": e["revision"], "quantization": rec["quantization"]}
    if e["role"] == "own":
        made = json.loads((OWN_ROOT / tag / "quantize.json").read_text())
        cfg["made"] = {k: made[k] for k in ("method", "scheme", "source", "ignore", "tools", "weights_sha256")}
    if e.get("layout"):
        cfg["layout"] = e["layout"]
    return cfg


#: Files a base model needs no copy of: Cosmos3-Nano runs as its reasoner
#: tower only, without the generator's VAE, sound tokenizer or scheduler.
BASE_EXCLUDE = {"cosmos3-nano": ["vae/*", "sound_tokenizer/*", "scheduler/*", "assets/*", "images/*"]}


def fetch_base(model: str) -> Path:
    """A model's BF16 weights at the adapter's pinned revision, into models/<repo name>/.
    A complete download is marked with .fetched-<revision>; a gated repo needs
    `uvx --from huggingface_hub hf auth login` first."""
    from huggingface_hub import snapshot_download

    a = MODELS[model]
    d = Path(a.local_dir)
    mark = d / f".fetched-{a.revision}"
    if mark.exists():
        print(f"== {model}: {a.hf_id} @ {a.revision[:10]} already fetched")
        return d
    print(f"== {model}: {a.hf_id} @ {a.revision[:10]} -> {d}")
    snapshot_download(a.hf_id, revision=a.revision, local_dir=d,
                      ignore_patterns=BASE_EXCLUDE.get(model))
    mark.touch()
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("tags", nargs="+")
    b = sub.add_parser("base", help="a model's BF16 weights at its pinned revision")
    b.add_argument("models", nargs="+", choices=sorted(MODELS))
    args = ap.parse_args(argv)
    if args.cmd == "base":
        for m in args.models:
            fetch_base(m)
        return 0
    if args.cmd == "list":
        for tag, e in manifest().items():
            src = f"{e['repo']}@{e['revision'][:10]}" if e.get("repo") else "(our own, not yet made)"
            print(f"{tag:42} {e['role']:9} {e['runtime']:10} {src}")
        return 0
    for tag in args.tags:
        build(tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
