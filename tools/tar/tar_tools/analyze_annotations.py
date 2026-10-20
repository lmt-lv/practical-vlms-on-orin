"""Static analysis of the TAR annotations — no videos required.

Answers the questions 001a owes: do the counts match the dataset card, what
is the per-source distribution, is every video annotated in every task, does
the ground truth survive the official extractors, and what do the answer and
reasoning lengths imply for token budgets.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime

from .annotations import (
    TASKS,
    Item,
    extract_letter,
    extract_yesno,
    gt_letter,
    gt_yesno,
    iou,
    load_all_tasks,
    load_task,
    load_test,
    parse_interval,
)
from .paths import annotations_dir, ensure_dirs, reports_dir

#: From the dataset card, for reconciliation.
CARD_COUNTS = {
    "bcq": 7340, "bcq_openended": 7340, "mcq": 3670, "mcq_openended": 3670,
    "open_qa": 3670, "scene_description": 3670, "video_summarization": 3670,
    "temporal_localization": 3670, "temporal_description": 3670,
    "causal_linkage": 3670,
}
CARD_VIDEOS = 3670
CARD_ANNOTATIONS = 44040


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
        return round(ordered[idx], 2)

    return {
        "n": len(ordered),
        "min": round(ordered[0], 2),
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p95": pct(95),
        "max": round(ordered[-1], 2),
        "mean": round(statistics.fmean(ordered), 2),
    }


def analyse_formats(task: str, items: list[Item]) -> dict:
    """Run the official GT extractors over the ground truth itself."""
    out: dict = {}
    if task == "bcq":
        parsed = [gt_yesno(i.answer) for i in items]
        out["gt_parseable"] = sum(p is not None for p in parsed)
        out["gt_unparseable"] = sum(p is None for p in parsed)
        out["yes_no_balance"] = dict(Counter(p for p in parsed if p))
        # The lenient prediction-side extractor should agree with the strict GT one.
        lenient = [extract_yesno(i.answer) for i in items]
        out["lenient_disagrees_with_strict"] = sum(
            a != b for a, b in zip(parsed, lenient, strict=True)
        )
    elif task == "mcq":
        parsed = [gt_letter(i.answer) for i in items]
        out["gt_parseable"] = sum(p is not None for p in parsed)
        out["gt_unparseable"] = sum(p is None for p in parsed)
        out["letter_distribution"] = dict(sorted(Counter(p for p in parsed if p).items()))
        lenient = [extract_letter(i.answer) for i in items]
        out["lenient_disagrees_with_strict"] = sum(
            a != b for a, b in zip(parsed, lenient, strict=True)
        )
    elif task == "temporal_localization":
        intervals = [parse_interval(i.answer) for i in items]
        ok = [iv for iv in intervals if iv is not None]
        out["gt_parseable"] = len(ok)
        out["gt_unparseable"] = len(intervals) - len(ok)
        out["interval_seconds"] = describe([e - s for s, e in ok])
        out["start_seconds"] = describe([s for s, _ in ok])
        out["zero_length_intervals"] = sum(1 for s, e in ok if e <= s)
        out["subsecond_precision_answers"] = sum(
            1 for i in items if isinstance(i.answer, str) and "." in i.answer
        )
    else:
        out["empty_answers"] = sum(1 for i in items if not i.answer.strip())
    return out


def temporal_quantisation_ceiling(items: list[Item]) -> dict:
    """Best mIoU a *perfect* model can score while obeying the MM:SS format.

    The question text instructs `{"start": "MM:SS", "end": "MM:SS"}` — whole
    seconds — but a large share of the ground truth carries centisecond
    precision. A model that knows the event exactly and rounds to the
    instructed format still loses IoU. This is the ceiling that number sits
    under, and it needs no video to compute.
    """
    exact: list[tuple[float, float]] = []
    for i in items:
        iv = parse_interval(i.answer)
        if iv is not None:
            exact.append(iv)

    def ceiling(intervals: list[tuple[float, float]], fn) -> float | None:
        scores = [iou(iv, (fn(iv[0]), fn(iv[1]))) for iv in intervals if iv[1] > iv[0]]
        return round(statistics.fmean(scores), 4) if scores else None

    subsecond = [iv for iv in exact if iv[0] % 1 or iv[1] % 1]
    return {
        "items_with_subsecond_gt": len(subsecond),
        "share_subsecond_pct": round(100 * len(subsecond) / len(exact), 1) if exact else None,
        "ceiling_miou_round_all": ceiling(exact, lambda x: round(x)),
        "ceiling_miou_round_subsecond_only": ceiling(subsecond, lambda x: round(x)),
        "ceiling_miou_floor_all": ceiling(exact, int),
        "worst_case_miou_round": round(
            min((iou(iv, (round(iv[0]), round(iv[1]))) for iv in exact if iv[1] > iv[0]),
                default=0.0), 4),
        "note": (
            "Upstream skips unparseable predictions rather than scoring them 0, "
            "so mIoU must always be reported next to the parse rate."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", default="annotation_stats")
    args = ap.parse_args()

    ensure_dirs()
    if not (annotations_dir() / "train").exists():
        raise SystemExit(
            f"No annotations at {annotations_dir()}. Run tar-fetch-annotations first."
        )

    by_task = load_all_tasks()
    report: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "annotations_dir": str(annotations_dir()),
        "tasks": {},
    }

    all_videos: set[str] = set()
    videos_by_task: dict[str, set[str]] = {}

    for task in TASKS:
        items = by_task[task]
        meta, _ = load_task(task)
        videos = {i.video_id for i in items}
        videos_by_task[task] = videos
        all_videos |= videos
        per_video = Counter(i.video_id for i in items)

        report["tasks"][task] = {
            "items": len(items),
            "items_per_card": CARD_COUNTS[task],
            "matches_card": len(items) == CARD_COUNTS[task],
            "unique_videos": len(videos),
            "items_per_video": dict(sorted(Counter(per_video.values()).items())),
            "metadata_date": meta.get("metadata", {}).get("date"),
            "lengths_chars": {
                "question": describe([len(i.question) for i in items]),
                "answer": describe([len(i.answer) for i in items]),
                "reasoning": describe([len(i.reasoning) for i in items]),
            },
            "duplicate_question_per_video": sum(
                c - 1 for c in Counter((i.video_id, i.question) for i in items).values() if c > 1
            ),
            "formats": analyse_formats(task, items),
        }

    # ---- cross-task consistency -------------------------------------------
    counts = Counter()
    for videos in videos_by_task.values():
        counts.update(videos)
    in_all = {v for v, c in counts.items() if c == len(TASKS)}
    partial = {v: c for v, c in counts.items() if c < len(TASKS)}

    report["videos"] = {
        "unique_total": len(all_videos),
        "card_claim": CARD_VIDEOS,
        "annotated_in_all_10_tasks": len(in_all),
        "partially_annotated": {v: counts[v] for v in sorted(partial)},
        "per_source": dict(
            Counter(v.split("/", 1)[0] for v in all_videos).most_common()
        ),
        "per_source_share_pct": {
            src: round(100 * n / len(all_videos), 1)
            for src, n in Counter(v.split("/", 1)[0] for v in all_videos).most_common()
        },
    }

    total_items = sum(len(v) for v in by_task.values())
    report["totals"] = {
        "annotations": total_items,
        "annotations_per_card": CARD_ANNOTATIONS,
        "matches_card": total_items == CARD_ANNOTATIONS,
    }

    report["temporal_quantisation"] = temporal_quantisation_ceiling(
        by_task["temporal_localization"]
    )

    # ---- scoring hazards, found by running the official extractors on the GT ----
    mcq_strict = sum(gt_letter(i.answer) is not None for i in by_task["mcq"])
    # Tasks are one-item-per-video except bcq*, which are two by design.
    multi_item_videos = {
        video
        for task in TASKS if not task.startswith("bcq")
        for video, count in Counter(i.video_id for i in by_task[task]).items()
        if count > 1
    }
    report["scoring_hazards"] = {
        "mcq_gt_unparseable_by_official_extractor": {
            "strict_parses": mcq_strict,
            "of": len(by_task["mcq"]),
            "detail": (
                "Train-split mcq answers are bare letters ('B'); the official "
                "_gt_letter requires '^([A-Za-z])\\\\)' and asserts on them. The "
                "prediction-side _extract_letter handles them via its "
                "fullmatch [A-Da-d] branch. Local scoring of the train split "
                "must use the lenient extractor for GT, and say so."
            ),
        },
        "duplicate_video_id": {
            "video_id": sorted(multi_item_videos),
            "detail": (
                "so-tad/test/47.mp4 carries double annotations in every task: "
                "the documented dedup re-anchored 12 items from so-tad/train/"
                "106.mp4 onto it. 3,669 unique videos, not the card's 3,670."
            ),
        },
    }

    # ---- test split --------------------------------------------------------
    try:
        test_meta, test_items = load_test()
        answered = sum(1 for i in test_items if i.answer.strip())
        report["test_split"] = {
            "items": len(test_items),
            "unique_videos": len({i.video_id for i in test_items}),
            "per_task": dict(sorted(Counter(i.task for i in test_items).items())),
            "answers_present": answered,
            "answers_redacted_flag": test_meta.get("metadata", {}).get("answers_redacted"),
            "has_item_index": sum(1 for i in test_items if i.item_index),
        }
    except FileNotFoundError:
        report["test_split"] = {"error": "test/test.json not found"}

    # ---- emit --------------------------------------------------------------
    json_path = reports_dir() / f"{args.out_prefix}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")

    md_path = reports_dir() / f"{args.out_prefix}.md"
    md_path.write_text(render_markdown(report))

    print(f"videos: {report['videos']['unique_total']} unique "
          f"(card says {CARD_VIDEOS}), "
          f"{report['videos']['annotated_in_all_10_tasks']} annotated in all 10 tasks")
    print(f"items : {total_items} (card says {CARD_ANNOTATIONS})")
    print(f"wrote  {json_path}\n       {md_path}")
    return 0


def render_markdown(report: dict) -> str:
    v = report["videos"]
    lines = [
        "# TAR annotations — static analysis",
        "",
        f"Generated {report['generated_at']} from `{report['annotations_dir']}`.",
        "",
        "## Totals",
        "",
        f"- Annotations: **{report['totals']['annotations']}** "
        f"(card: {report['totals']['annotations_per_card']}, "
        f"{'match' if report['totals']['matches_card'] else 'MISMATCH'})",
        f"- Unique videos: **{v['unique_total']}** (card: {v['card_claim']})",
        f"- Annotated in all 10 task types: **{v['annotated_in_all_10_tasks']}**",
        "",
        "## Videos per source",
        "",
        "| Source | Videos | Share |",
        "|---|---:|---:|",
    ]
    for src, n in v["per_source"].items():
        lines.append(f"| {src} | {n} | {v['per_source_share_pct'][src]}% |")

    lines += [
        "",
        "## Per task",
        "",
        "| Task | Items | Card | Videos | Items/video | Answer chars (median) | Reasoning chars (median) |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for task, t in report["tasks"].items():
        ipv = ", ".join(f"{k}x{val}" for k, val in t["items_per_video"].items())
        lines.append(
            f"| {task} | {t['items']} | {t['items_per_card']} | {t['unique_videos']} | "
            f"{ipv} | {t['lengths_chars']['answer'].get('median')} | "
            f"{t['lengths_chars']['reasoning'].get('median')} |"
        )

    lines += [
        "",
        "## Temporal localization: format-imposed IoU ceiling",
        "",
        "```json",
        json.dumps(report["temporal_quantisation"], indent=2),
        "```",
        "",
        "## Scoring hazards",
        "",
        "```json",
        json.dumps(report["scoring_hazards"], indent=2),
        "```",
        "",
    ]
    lines += ["", "## Ground-truth format conformance", ""]
    for task in ("bcq", "mcq", "temporal_localization"):
        lines += [f"### {task}", "", "```json",
                  json.dumps(report["tasks"][task]["formats"], indent=2), "```", ""]

    if "error" not in report["test_split"]:
        lines += ["## Test split", "", "```json",
                  json.dumps(report["test_split"], indent=2), "```", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
