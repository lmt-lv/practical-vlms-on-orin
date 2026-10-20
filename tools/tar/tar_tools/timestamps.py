"""Timestamps in annotation text, and the ones that point past the video's end.

The label pipeline sometimes writes seconds.centiseconds in HH:MM:SS form —
20.04 s as `00:20:04`, 10.60 s as `00:10:60` — which the official scorer's
parser reads as 1,204 s and 660 s. 001b found this in 50 temporal answers;
the viewer then showed it is not confined to them. This check measures it
across every task and field.

It only catches the bug where it pushes a timestamp past the end of the
clip, so the counts are a lower bound: a bugged `00:00:42` meant as 0.42 s
still lands inside a 60 s clip and goes unflagged.

For eval slices the fields that matter are `question` (model input) and
`answer` (ground truth); `reasoning` is a training-time trace and is never
scored.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime

from .annotations import TASKS, load_all_tasks, parse_timestamp
from .frames import duration_of
from .paths import reports_dir

#: MM:SS, HH:MM:SS, optionally with fractional seconds — the forms the
#: annotations actually use. Anchored so it cannot start or end mid-number.
TIMESTAMP = re.compile(r"(?<![\d:.])\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?![\d:])")

#: Slack for rounding at the end of a clip.
TOLERANCE_S = 0.5

SCORED_FIELDS = ("question", "answer")
ALL_FIELDS = ("question", "answer", "reasoning")


def overruns(text: str, duration: float) -> list[tuple[str, float]]:
    """Timestamps in `text` that parse to a time past the end of the clip."""
    out = []
    for m in TIMESTAMP.finditer(text):
        try:
            secs = parse_timestamp(m.group(0))
        except ValueError:
            continue
        if secs > duration + TOLERANCE_S:
            out.append((m.group(0), secs))
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    by_task = load_all_tasks()
    per_task = {}
    field_hits: Counter = Counter()
    flagged: list[dict] = []

    for task in TASKS:
        resolved = scored_bad = any_bad = 0
        for item in by_task[task]:
            duration = duration_of(item.video_id)
            if not duration:
                continue
            resolved += 1
            hits = {f: overruns(getattr(item, f), duration) for f in ALL_FIELDS}
            for field, found in hits.items():
                field_hits[(task, field)] += len(found)
            if any(hits[f] for f in SCORED_FIELDS):
                scored_bad += 1
                flagged.append({
                    "task": task, "video_id": item.video_id, "duration_s": duration,
                    "question": [s for s, _ in hits["question"]],
                    "answer": [s for s, _ in hits["answer"]],
                })
            any_bad += any(hits.values())
        per_task[task] = {
            "resolved_items": resolved,
            "overrun_in_question_or_answer": scored_bad,
            "pct_question_or_answer": round(100 * scored_bad / resolved, 1),
            "overrun_anywhere_incl_reasoning": any_bad,
        }

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tolerance_s": TOLERANCE_S,
        "note": "Lower bound: only bugged timestamps that land past the clip end are caught.",
        "per_task": per_task,
        "timestamps_by_field": {f"{t}.{f}": n for (t, f), n in sorted(field_hits.items()) if n},
        "total_items_to_exclude": len(flagged),
        "items": flagged,
    }
    out = reports_dir() / "timestamp_overruns.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{'task':22s} {'q/a overrun':>12s} {'of':>6s} {'pct':>6s}   incl. reasoning")
    for task, s in per_task.items():
        print(f"{task:22s} {s['overrun_in_question_or_answer']:12d} {s['resolved_items']:6d} "
              f"{s['pct_question_or_answer']:5.1f}%   {s['overrun_anywhere_incl_reasoning']}")
    print(f"\n{len(flagged)} items to exclude from eval slices -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
