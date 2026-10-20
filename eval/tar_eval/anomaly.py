"""Anomalous/normal labels per video.

TAR does not ship a per-video anomaly label, and for so-tad the path does
not carry one either. But the `video_summarization` question was generated
with the label in hand: it asks about "anomalies" when the video is
anomalous and about "traffic conditions" or similar when it is not. On the
sources where the path does give the truth, that rule has ~98% precision
and recall, and it flags 966 videos in total against the track's stated
965 — it reconstructs TAR's own labelling. It is used as the label for
every source, so the stratification is consistent; the path-derived truth
is kept alongside for the agreement report.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict

from .common import REPORTS_DIR


def path_truth(video_id: str) -> bool | None:
    """The source dataset's own label, where the path encodes one."""
    source = video_id.split("/", 1)[0]
    if source == "TAD-benchmark":
        return "/normal/" not in video_id
    if source == "TAD":
        return "/Normal_" not in video_id
    if source == "Vad-R1" and ("Abnormal" in video_id or "Normal" in video_id):
        return "Abnormal" in video_id
    if source == "UCF_Crimes":
        return "Normal" not in video_id
    if source in ("HTV", "barbados_traffic_challenge"):
        return False      # plain traffic footage by construction
    if source == "Accident-Bench":
        return True       # an accident benchmark by construction
    return None


def question_rule(summarization_question: str) -> bool:
    return bool(re.search(r"anomal", summarization_question, re.I))


def label_videos(by_task: dict[str, list]) -> dict[str, dict]:
    """video_id -> {anomalous, path_truth}. `anomalous` is the question rule."""
    questions = {i.video_id: i.question for i in by_task["video_summarization"]}
    return {
        vid: {"anomalous": question_rule(q), "path_truth": path_truth(vid)}
        for vid, q in questions.items()
    }


def agreement_report(labels: dict[str, dict], pool: set[str]) -> dict:
    by_source: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                                      "no_truth": 0, "flagged": 0, "videos": 0})
    for vid, lab in labels.items():
        if vid not in pool:
            continue
        s = by_source[vid.split("/", 1)[0]]
        s["videos"] += 1
        s["flagged"] += lab["anomalous"]
        truth = lab["path_truth"]
        if truth is None:
            s["no_truth"] += 1
        elif truth and lab["anomalous"]:
            s["tp"] += 1
        elif truth:
            s["fn"] += 1
        elif lab["anomalous"]:
            s["fp"] += 1
        else:
            s["tn"] += 1
    total = {k: sum(s[k] for s in by_source.values()) for k in ("tp", "fp", "fn", "tn")}
    return {
        "by_source": dict(sorted(by_source.items())),
        "overall": total | {
            "precision": round(total["tp"] / max(1, total["tp"] + total["fp"]), 3),
            "recall": round(total["tp"] / max(1, total["tp"] + total["fn"]), 3),
        },
    }


def write_report(report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "anomaly_labels.json").write_text(json.dumps(report, indent=2) + "\n")
    o = report["overall"]
    lines = [
        "# Anomaly labels",
        "",
        "Label = the `video_summarization` question mentions anomalies "
        "(see `eval/tar_eval/anomaly.py`). Agreement with the path-derived "
        "label, where the source has one, over the slice pool:",
        "",
        f"**Precision {o['precision']}, recall {o['recall']}** "
        f"(tp {o['tp']}, fp {o['fp']}, fn {o['fn']}, tn {o['tn']}).",
        "",
        "| Source | Videos | Flagged anomalous | tp | fp | fn | tn | No path label |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for src, s in report["by_source"].items():
        lines.append(f"| {src} | {s['videos']} | {s['flagged']} | {s['tp']} | {s['fp']} | "
                     f"{s['fn']} | {s['tn']} | {s['no_truth']} |")
    (REPORTS_DIR / "anomaly_labels.md").write_text("\n".join(lines) + "\n")
