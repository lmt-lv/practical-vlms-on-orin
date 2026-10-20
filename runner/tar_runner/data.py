"""Inputs a run needs: slice rows, frames with timestamps, source video facts.

Everything is read from files produced elsewhere — the frozen slices
(`subsets/`), the frame cache and its manifest (`data/frames/`), the
frame sets (`outputs/eval/frame_cache.json`) and the media probe
(`data/tar/derived/media_probe.jsonl`). Nothing is imported from `eval/`,
which pins a different transformers.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
import os
from functools import cache
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
#: TAR's data (data/tar/), or $TAR_ROOT; the frame cache sits beside it (data/frames/).
TAR_ROOT = Path(os.environ["TAR_ROOT"]).expanduser().resolve() if os.environ.get("TAR_ROOT") \
    else REPO_ROOT / "data" / "tar"
SLICES_DIR = REPO_ROOT / "subsets"
FRAMES_ROOT = TAR_ROOT.parent / "frames"
MODELS_ROOT = REPO_ROOT / "models"
RUNS_ROOT = REPO_ROOT / "runs"
PROBE_FILE = TAR_ROOT / "derived" / "media_probe.jsonl"
FRAME_CACHE_REPORT = REPO_ROOT / "outputs" / "eval" / "frame_cache.json"

SHORTLIST = frozenset({"bcq", "mcq", "mcq_openended", "causal_linkage",
                       "video_summarization", "temporal_localization"})
ALL_TASKS = ("bcq", "bcq_openended", "mcq", "mcq_openended", "open_qa",
             "scene_description", "video_summarization", "temporal_localization",
             "temporal_description", "causal_linkage")


@dataclass(frozen=True)
class Frame:
    pos: int
    frame_index: int
    pts_s: float
    path: Path

    def image(self) -> Image.Image:
        return Image.open(self.path).convert("RGB")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def slice_file(name: str) -> Path:
    return Path(name) if name.endswith(".jsonl") else SLICES_DIR / f"{name}.jsonl"


def load_rows(slice_name: str, tasks: str = "all", prefix: int | None = None) -> list[dict]:
    if tasks == "shortlist":
        wanted = set(SHORTLIST)
    elif tasks == "all":
        wanted = set(ALL_TASKS)
    else:
        wanted = set(tasks.split(","))
        unknown = wanted - set(ALL_TASKS)
        if unknown:
            raise SystemExit(f"unknown task(s): {sorted(unknown)}")
    rows = read_jsonl(slice_file(slice_name))
    return [r for r in rows if r["task"] in wanted and (prefix is None or r["priority"] <= prefix)]


#: The frame counts' positions among the cache's 32 (as eval's tar_eval.common.FRAME_SETS).
FRAME_SETS = {32: tuple(range(32)), 16: tuple(range(1, 32, 2)), 8: tuple(range(1, 32, 4)),
              4: tuple(range(5, 32, 8))}


@cache
def frame_sets() -> dict[int, tuple[int, ...]]:
    """The cache's report where it was built here (tar-eval-frames), else the fixed sets."""
    if not FRAME_CACHE_REPORT.exists():
        return FRAME_SETS
    sets = json.loads(FRAME_CACHE_REPORT.read_text())["frame_sets"]
    return {int(k): tuple(v) for k, v in sets.items()}


#: TAR-Bench's clips have their own manifest, so the dev/eval one —
#: and the hash every dev/eval run records — is unchanged by adding them.
MANIFESTS = ("manifest.jsonl", "manifest_tarbench.jsonl", "manifest_calib.jsonl",
             "manifest_tarbench_extra.jsonl")


@cache
def _manifest() -> dict[tuple[str, int], list[dict]]:
    index: dict[tuple[str, int], list[dict]] = {}
    for name in MANIFESTS:
        if (FRAMES_ROOT / name).exists():
            for r in read_jsonl(FRAMES_ROOT / name):
                index.setdefault((r["video_id"], r["cap"]), []).append(r)
    return index


def frames_for(video_id: str, n_frames: int, cap: int) -> list[Frame]:
    """The run's frames for one video, in time order."""
    positions = set(frame_sets()[n_frames])
    stem = video_id.rsplit(".", 1)[0]
    rows = sorted((r for r in _manifest()[(video_id, cap)] if r["pos"] in positions),
                  key=lambda r: r["pos"])
    return [Frame(r["pos"], r["frame_index"], r["pts_s"],
                  FRAMES_ROOT / f"c{cap}" / stem / f"{r['pos']:02d}.png") for r in rows]


@cache
def _probe() -> dict[str, dict]:
    return {r["path"]: r for r in read_jsonl(PROBE_FILE) if "path" in r}


def source_video(video_id: str) -> dict:
    """Source fps and frame count, which Qwen-style native video input needs."""
    p = _probe()[video_id]
    fps = p["fps"]
    total = p.get("nb_frames") or int(round(p["duration_s"] * fps))
    return {"fps": fps, "total_num_frames": total, "duration_s": p["duration_s"]}


def frame_manifest_sha256(slice_name: str = "") -> str:
    """The manifest a run's frames come from: TAR-Bench's for the tarbench
    slice, the dev/eval one otherwise."""
    name = {"tarbench": "manifest_tarbench.jsonl", "calib": "manifest_calib.jsonl",
            "tarbench_extra": "manifest_tarbench_extra.jsonl"}.get(
        Path(slice_name).stem, "manifest.jsonl")
    return sha256_file(FRAMES_ROOT / name)
