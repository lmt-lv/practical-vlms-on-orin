"""Loading TAR annotations, and the official answer extractors.

The extractors below are transcribed from `test/evaluate.py` as published in
the dataset repo (AI City Challenge 2026 Track 3 evaluator). They are copied
rather than imported because the upstream file needs pandas/numpy and is not
importable as a module. Any divergence between these and upstream is a bug:
we score against the same code paths the organizers do.

Upstream scoring, for reference:
  bcq / bcq_openended   -> accuracy on extracted Yes/No
  mcq / mcq_openended   -> accuracy on extracted option letter
  temporal_localization -> mean IoU over {"start": "MM:SS", "end": "MM:SS"}
  open-ended tasks      -> BERTScore F1 (roberta-large, rescale_with_baseline)
  headline              -> unweighted mean of the per-task metrics

Two upstream behaviours worth carrying forward into our own scoring:

1. `temporal_localization` **skips** unparseable predictions instead of
   scoring them zero, so a model that emits garbage can post a higher mIoU
   than one that emits parseable-but-wrong intervals. mIoU is therefore
   meaningless without the parse rate reported beside it.
2. `bcq`/`mcq` extraction falls back to a bare `\\b(yes|no)\\b` / `\\b([A-D])\\b`
   search anywhere in the output, so a chatty refusal that happens to
   contain "no" scores as an answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .paths import annotations_dir

TASKS = (
    "bcq",
    "bcq_openended",
    "mcq",
    "mcq_openended",
    "open_qa",
    "scene_description",
    "video_summarization",
    "temporal_localization",
    "temporal_description",
    "causal_linkage",
)

#: Tasks scored with BERTScore F1 upstream.
OPEN_ENDED_TASKS = frozenset({
    "bcq_openended", "mcq_openended", "open_qa",
    "causal_linkage", "scene_description", "temporal_description",
    "video_summarization",
})

#: Tasks with an exactly-scorable answer; the cheap axes for wide sweeps.
CLOSED_FORM_TASKS = frozenset({"bcq", "mcq", "temporal_localization"})


@dataclass(frozen=True)
class Item:
    task: str
    video_id: str
    question: str
    answer: str
    reasoning: str
    item_index: str | None = None

    @property
    def source(self) -> str:
        """Upstream dataset the video came from, e.g. 'so-tad'."""
        return self.video_id.split("/", 1)[0]


def load_task(task: str, root: Path | None = None) -> tuple[dict, list[Item]]:
    """Load one train task file. Returns (envelope metadata, items)."""
    root = root or annotations_dir()
    path = root / "train" / f"{task}.json"
    with path.open() as fh:
        envelope = json.load(fh)
    items = [
        Item(
            task=task,
            video_id=raw["video_id"],
            question=raw.get("question", ""),
            answer=raw.get("answer", ""),
            reasoning=raw.get("reasoning", ""),
            item_index=raw.get("item_index"),
        )
        for raw in envelope["items"]
    ]
    return {k: v for k, v in envelope.items() if k != "items"}, items


def load_all_tasks(root: Path | None = None) -> dict[str, list[Item]]:
    return {task: load_task(task, root)[1] for task in TASKS}


def load_test(root: Path | None = None) -> tuple[dict, list[Item]]:
    """Load the redacted test split. Items carry `item_index` and a task field."""
    root = root or annotations_dir()
    with (root / "test" / "test.json").open() as fh:
        envelope = json.load(fh)
    items = [
        Item(
            task=raw.get("task", "unknown"),
            video_id=raw["video_id"],
            question=raw.get("question", ""),
            answer=raw.get("answer", "") or "",
            reasoning=raw.get("reasoning", "") or "",
            item_index=raw.get("item_index"),
        )
        for raw in envelope["items"]
    ]
    return {k: v for k, v in envelope.items() if k != "items"}, items


# --------------------------------------------------------------------------
# Official extractors, transcribed from test/evaluate.py
# --------------------------------------------------------------------------


def extract_yesno(text: str | None) -> str | None:
    if text is None or not str(text).strip():
        return None
    s = str(text).strip().lower()
    m = re.match(r"^(yes|no)\b", s)
    if m:
        return m.group(1)
    m = re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def extract_letter(text: str | None) -> str | None:
    if text is None or not str(text).strip():
        return None
    s = str(text).strip()
    m = re.match(r"^\(?([A-Za-z])\)?[).\s,:]", s)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"[A-Da-d]", s):
        return s.upper()
    m = re.search(r"\b([A-D])\b", s)
    return m.group(1) if m else None


def gt_yesno(answer: str) -> str | None:
    """Upstream asserts here; we return None so a malformed GT is counted, not fatal."""
    if not answer or not str(answer).strip():
        return None
    first = str(answer).strip().lower().split(".")[0].split()[0]
    return first if first in ("yes", "no") else None


def gt_letter(answer: str) -> str | None:
    if not answer or not str(answer).strip():
        return None
    m = re.match(r"^([A-Za-z])\)", str(answer).strip())
    return m.group(1).upper() if m else None


def parse_timestamp(ts) -> float:
    parts = str(ts).strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(ts)


def extract_json(text: str | None):
    if text is None or not str(text).strip():
        return None
    s = str(text).strip()
    m = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) \
                    and "start" in obj[0] and "end" in obj[0]:
                return obj[0]
            return obj
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def parse_interval(text: str | None) -> tuple[float, float] | None:
    """Extract a (start, end) pair in seconds, or None if unparseable."""
    obj = extract_json(text)
    if not isinstance(obj, dict) or "start" not in obj or "end" not in obj:
        return None
    try:
        return parse_timestamp(obj["start"]), parse_timestamp(obj["end"])
    except (KeyError, ValueError, TypeError):
        return None


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Temporal IoU, transcribed from `_score_temporal_localization`.

    Union is computed as (len_a + len_b - inter) rather than
    (max_end - min_start), so disjoint intervals score 0 without the gap
    inflating the denominator.
    """
    a_start, a_end = a
    b_start, b_end = b
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(0.0, (a_end - a_start) + (b_end - b_start) - inter)
    return inter / union if union > 0 else 0.0
