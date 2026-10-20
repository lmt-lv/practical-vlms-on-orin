"""Seeded, stratified samples for the 001c audit.

Both samples are written before anything is scored, and committed, so the
audit cannot be quietly re-rolled after seeing the results.

Two samples:

  audit   50 annotation items, 10 each across five task types spanning the
          difficulty ladder, judged individually for answerability and
          correctness.
  camera  200 videos, one frame each, labelled for camera type and traffic
          relevance — "is this actually a traffic dataset", answered with
          numbers instead of source reputation.

so-tad is 60% of the corpus and is strikingly uniform (20 s, 640x480, 25 fps),
so proportional sampling would make both samples mostly a description of
so-tad. Allocation is therefore square-root-weighted across sources: still
dominated by the big sources, but with every small source represented.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict

from .annotations import load_task, parse_interval
from .frames import duration_of
from .paths import reports_dir
from .probe_media import PROBE_FILE, load_existing
from .paths import derived_dir

AUDIT_TASKS = ("bcq", "mcq", "temporal_localization", "causal_linkage", "video_summarization")
AUDIT_PER_TASK = 10
CAMERA_N = 200
SEED = 20260921


def resolved_videos(include_test: bool = False) -> set[str]:
    """Readable files that the annotations actually reference.

    The probe sweep covers everything on disk, which is more than the dataset:
    TAD's stitcher emits 510 mp4s where only 191 are annotated. Sampling from
    the filesystem would describe files we will never evaluate on.
    """
    readable = {
        k for k, v in load_existing(derived_dir() / PROBE_FILE).items() if v.get("ok")
    }
    _, items = load_task("mcq")  # one item per video: the complete train list
    annotated = {i.video_id for i in items}
    if include_test:
        annotated |= {k for k in readable if k.startswith("tar_test/")}
    return readable & annotated


def sqrt_allocation(counts: dict[str, int], total: int) -> dict[str, int]:
    """Allocate `total` slots across sources, square-root weighted."""
    weights = {s: math.sqrt(n) for s, n in counts.items() if n}
    scale = total / sum(weights.values())
    alloc = {s: max(1, round(w * scale)) for s, w in weights.items()}
    # Trim or pad the largest source to hit the target exactly.
    biggest = max(alloc, key=lambda s: alloc[s])
    alloc[biggest] += total - sum(alloc.values())
    return alloc


def buggy_temporal(resolved: set[str]) -> set[str]:
    """Videos whose temporal GT interval runs past the end of the video."""
    _, items = load_task("temporal_localization")
    bad = set()
    for item in items:
        dur = duration_of(item.video_id)
        interval = parse_interval(item.answer)
        if dur and interval and interval[1] > dur + 0.5:
            bad.add(item.video_id)
    return bad & resolved


def build_audit(rng: random.Random, resolved: set[str]) -> list[dict]:
    rows = []
    bugged = buggy_temporal(resolved)
    for task in AUDIT_TASKS:
        _, items = load_task(task)
        pool = [i for i in items if i.video_id in resolved]
        by_source: dict[str, list] = defaultdict(list)
        for item in pool:
            by_source[item.source].append(item)
        counts = {s: len(v) for s, v in by_source.items()}

        picked = []
        if task == "temporal_localization":
            # Deliberately include two known-bad items to confirm the unit-bug
            # diagnosis visually rather than only arithmetically.
            bad_items = sorted((i for i in pool if i.video_id in bugged),
                               key=lambda i: i.video_id)
            picked += rng.sample(bad_items, min(2, len(bad_items)))

        alloc = sqrt_allocation(counts, AUDIT_PER_TASK - len(picked))
        for source, n in sorted(alloc.items()):
            candidates = [i for i in by_source[source] if i not in picked]
            picked += rng.sample(candidates, min(n, len(candidates)))
        picked = picked[:AUDIT_PER_TASK]

        for item in picked:
            rows.append({
                "task": task,
                "video_id": item.video_id,
                "source": item.source,
                "duration_s": duration_of(item.video_id),
                "stratum": "unit_bug" if (task == "temporal_localization"
                                          and item.video_id in bugged) else "normal",
                "question": item.question.replace("\n", " | "),
                "answer": item.answer.replace("\n", " "),
            })
    return rows


def build_camera(rng: random.Random, resolved: set[str]) -> list[dict]:
    by_source: dict[str, list[str]] = defaultdict(list)
    for vid in sorted(resolved):
        by_source[vid.split("/", 1)[0]].append(vid)
    counts = {s: len(v) for s, v in by_source.items()}
    alloc = sqrt_allocation(counts, CAMERA_N)
    rows = []
    for source, n in sorted(alloc.items()):
        for vid in rng.sample(by_source[source], min(n, len(by_source[source]))):
            rows.append({
                "video_id": vid, "source": source, "duration_s": duration_of(vid),
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    resolved = resolved_videos()
    with_test = resolved_videos(include_test=True)
    print(f"{len(resolved)} annotated train videos resolved "
          f"({len(with_test) - len(resolved)} test clips also available)")

    audit = build_audit(random.Random(args.seed), resolved)
    camera = build_camera(random.Random(args.seed + 1), with_test)

    for name, rows in (("audit_sample", audit), ("camera_sample", camera)):
        path = reports_dir() / f"{name}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {len(rows)} rows -> {path}")

    from collections import Counter
    print("audit  by task  :", dict(Counter(r["task"] for r in audit)))
    print("audit  by source:", dict(Counter(r["source"] for r in audit)))
    print("camera by source:", dict(Counter(r["source"] for r in camera)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
