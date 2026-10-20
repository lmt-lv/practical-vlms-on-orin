"""Shared definitions: paths, item ids, task sets, frame sets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tar_tools.annotations import TASKS  # noqa: F401  (re-exported)
from tar_tools.paths import REPO_ROOT, data_root

EVAL_ROOT = REPO_ROOT / "eval"
SLICES_DIR = REPO_ROOT / "subsets"            # the frozen subsets, committed
REPORTS_DIR = REPO_ROOT / "outputs" / "eval"   # the tools' reports, not committed
FRAMES_ROOT = data_root().parent / "frames"      # data/frames
RUNS_ROOT = REPO_ROOT / "runs"

SEED = 20260925

#: The study's headline set: 7 questions per video (bcq contributes two).
SHORTLIST = frozenset({
    "bcq", "mcq", "mcq_openended", "causal_linkage",
    "video_summarization", "temporal_localization",
})

#: Tasks scored by exact match on an extracted answer.
BINARY_TASKS = frozenset({"bcq", "mcq"})

#: Frame counts and the positions (out of 32) each uses. Nested, so fewer
#: frames is a strict subset of the same information. Positions sit on the
#: grid t = (i + 0.5) * d / 32; the subsets are slightly off-centre:
#: 16 frames +0.25 of a step late, 8 frames -0.125, 4 frames +0.19.
FRAME_SETS: dict[int, tuple[int, ...]] = {
    32: tuple(range(32)),
    16: tuple(range(1, 32, 2)),
    8: tuple(range(1, 32, 4)),
    4: tuple(range(5, 32, 8)),
}
N_FRAMES_MAX = 32
RESOLUTION_CAPS = (448, 896)


def item_id(task: str, video_id: str, question: str) -> str:
    """Stable 16-hex id, the same shape as the test split's item_index."""
    return hashlib.sha256(f"{task}\x1f{video_id}\x1f{question}".encode()).hexdigest()[:16]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def slice_path(name: str) -> Path:
    return SLICES_DIR / f"{name}.jsonl"


def frame_dir(cap: int, video_id: str) -> Path:
    """data/frames/c<cap>/<video_id without extension>/"""
    stem = video_id.rsplit(".", 1)[0]
    return FRAMES_ROOT / f"c{cap}" / stem
