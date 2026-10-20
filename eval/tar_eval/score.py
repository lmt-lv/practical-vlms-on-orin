"""Score predictions against a slice, mirroring the official TAR evaluator.

Official metrics, computed exactly as `test/evaluate.py` does:
  bcq                    accuracy on the extracted Yes/No
  mcq                    accuracy on the extracted option letter
  temporal_localization  mean IoU over *parseable* predictions (unparseable
                         ones are skipped, not scored zero)
  open-ended tasks       BERTScore F1, roberta-large, rescale_with_baseline

Additions, never mixed into the official numbers:
  - `mcq_openended` and `bcq_openended` also get the answer letter / Yes-No
    scored exactly, so reasoning tasks have an exactly-scored anchor;
  - parse outcome per item: `strict` (the answer opens the reply), `lenient`
    (found only by the fallback search somewhere in a longer reply) or
    `fail` — a model that grows chatty shows up as lenient before it shows
    up as failures;
  - temporal mIoU with unparseable predictions scored zero, beside the
    official figure;
  - a "final answer" reading (`*_final`): the answer where it opens the
    reply, as the official extractors take it, else the *last* Yes/No,
    option letter or {start, end} object in the reply rather than the
    first. The official extractors read an explain-first reply's first
    stray letter, and accept the interval JSON only alone or fenced;
    models that reason aloud before answering (Qwen3.5-9B) are scored on
    their answer here. Secondary, never a replacement for the official
    figure.

Train-split mcq ground truth is a bare letter, which the official GT parser
rejects; it goes through the lenient prediction-side extractor instead.

Note an upstream quirk this faithfully reproduces: the letter extractor's
first pattern accepts any leading capital followed by a space, so a reply
opening with the article "A ..." parses as option A.

Predictions are JSONL with at least `item_id` and `prediction`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from tar_tools.annotations import (
    OPEN_ENDED_TASKS, TASKS, extract_letter, extract_yesno, gt_yesno, iou, parse_interval,
)

from .common import SHORTLIST, read_jsonl, slice_path, write_jsonl

STRICT_YESNO = re.compile(r"^(yes|no)\b", re.I)
STRICT_LETTER = re.compile(r"^\(?([A-Za-z])\)?[).\s,:]")
LETTER_ONLY = re.compile(r"[A-Da-d]")

_SCORER = None

LAST_YESNO = re.compile(r"\b(yes|no)\b", re.I)
LAST_LETTER = re.compile(r"\b([A-D])\b")
INTERVAL_OBJ = re.compile(r"\{[^{}]*\"start\"[^{}]*\"end\"[^{}]*\}", re.S)


def final_yesno(text: str) -> str | None:
    s = (text or "").strip()
    if (m := STRICT_YESNO.match(s)):
        return m.group(1).lower()
    found = LAST_YESNO.findall(s)
    return found[-1].lower() if found else None


def final_letter(text: str) -> str | None:
    s = (text or "").strip()
    if (m := STRICT_LETTER.match(s)) or (m := re.fullmatch(r"\(?([A-Da-d])\)?\.?", s)):
        return m.group(1).upper()
    found = LAST_LETTER.findall(s)
    return found[-1] if found else None


def final_interval(text: str):
    """The official parse if it succeeds, else the last {start, end} object."""
    pr = parse_interval(text or "")
    if pr is not None:
        return pr
    for obj in reversed(INTERVAL_OBJ.findall(text or "")):
        if (pr := parse_interval(obj)) is not None:
            return pr
    return None


def bertscore_f1(candidates: list[str], references: list[str]) -> list[float]:
    """Per-pair BERTScore F1 with the official settings."""
    global _SCORER
    if not candidates:
        return []
    if _SCORER is None:
        import bert_score
        _SCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    _, _, f1 = _SCORER.score(candidates, references)
    return [float(x) for x in f1]


def yesno_parse(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "fail"
    if STRICT_YESNO.match(s):
        return "strict"
    return "lenient" if extract_yesno(s) else "fail"


def letter_parse(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return "fail"
    if STRICT_LETTER.match(s) or LETTER_ONLY.fullmatch(s):
        return "strict"
    return "lenient" if extract_letter(s) else "fail"


def select_rows(rows: list[dict], tasks: str | None, prefix: int | None) -> list[dict]:
    if tasks == "shortlist":
        wanted = SHORTLIST
    elif tasks in (None, "all"):
        wanted = set(TASKS)
    else:
        wanted = set(tasks.split(","))
        unknown = wanted - set(TASKS)
        if unknown:
            raise SystemExit(f"unknown task(s): {sorted(unknown)}")
    return [r for r in rows if r["task"] in wanted and (prefix is None or r["priority"] <= prefix)]


def score_items(rows: list[dict], predictions: dict[str, str],
                allow_missing: bool = False) -> list[dict]:
    """One record per item; open-ended items get BERTScore in one batch."""
    missing = [r["item_id"] for r in rows if r["item_id"] not in predictions]
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} items have no prediction (first: {missing[0]}); "
                         "pass --allow-missing to score the rest")
    out, bert_idx, cands, refs = [], [], [], []
    for r in rows:
        if r["item_id"] not in predictions:
            continue
        pred = predictions[r["item_id"]] or ""
        task = r["task"]
        rec = {k: r[k] for k in ("item_id", "video_id", "task", "priority", "anomalous", "cites_time")}
        rec |= {"parse": "n/a", "correct": None, "iou": None, "bertscore_f1": None,
                "correct_final": None, "iou_final": None}
        if task in ("bcq", "bcq_openended"):
            rec["parse"] = yesno_parse(pred)
            rec["correct"] = int(extract_yesno(pred) == gt_yesno(r["answer"]))
            rec["correct_final"] = int(final_yesno(pred) == gt_yesno(r["answer"]))
        elif task in ("mcq", "mcq_openended"):
            rec["parse"] = letter_parse(pred)
            rec["correct"] = int(extract_letter(pred) == extract_letter(r["answer"]))
            rec["correct_final"] = int(final_letter(pred) == extract_letter(r["answer"]))
        elif task == "temporal_localization":
            gt, pr = parse_interval(r["answer"]), parse_interval(pred)
            rec["parse"] = "strict" if pr is not None else "fail"
            rec["iou"] = iou(gt, pr) if (gt is not None and pr is not None) else None
            prf = final_interval(pred)
            rec["iou_final"] = iou(gt, prf) if (gt is not None and prf is not None) else None
        if task in OPEN_ENDED_TASKS:
            bert_idx.append(len(out))
            cands.append(pred)
            refs.append(r["answer"])
        out.append(rec)
    for i, f1 in zip(bert_idx, bertscore_f1(cands, refs), strict=True):
        out[i]["bertscore_f1"] = f1
    return out


def aggregate(items: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for rec in items:
        by_task.setdefault(rec["task"], []).append(rec)

    official, extra, parse = {}, {}, {}
    for task in TASKS:
        recs = by_task.get(task)
        if not recs:
            continue
        n = len(recs)
        if task == "bcq":
            official["bcq_accuracy"] = sum(r["correct"] for r in recs) / n
        elif task == "mcq":
            official["mcq_accuracy"] = sum(r["correct"] for r in recs) / n
        elif task == "temporal_localization":
            ious = [r["iou"] for r in recs if r["iou"] is not None]
            official["temporal_localization_miou"] = statistics.fmean(ious) if ious else 0.0
            extra["temporal_localization_miou_unparseable_as_zero"] = sum(ious) / n
        if task in OPEN_ENDED_TASKS:
            official[f"{task}_bertscore_f1"] = statistics.fmean(r["bertscore_f1"] for r in recs)
        if task == "bcq_openended":
            extra["bcq_openended_exact_accuracy"] = sum(r["correct"] for r in recs) / n
        if task == "mcq_openended":
            extra["mcq_openended_letter_accuracy"] = sum(r["correct"] for r in recs) / n
        if recs[0]["correct_final"] is not None:
            extra[f"{task}_accuracy_final"] = sum(r["correct_final"] for r in recs) / n
        if task == "temporal_localization":
            ious_f = [r["iou_final"] for r in recs if r["iou_final"] is not None]
            extra["temporal_localization_miou_final"] = sum(ious_f) / n
            extra["temporal_localization_parse_rate_final"] = len(ious_f) / n
        if recs[0]["parse"] != "n/a":
            parse[task] = {k: round(sum(r["parse"] == k for r in recs) / n, 4)
                           for k in ("strict", "lenient", "fail")}

    summary = {
        "items": len(items),
        "videos": len({r["video_id"] for r in items}),
        "items_per_task": {t: len(v) for t, v in by_task.items()},
        "official": {k: round(v, 6) for k, v in official.items()},
        "extra": {k: round(v, 6) for k, v in extra.items()},
        "parse_rates": parse,
    }
    if official:
        # What the published evaluate.py reports as "mean": all metrics present.
        summary["official_mean"] = round(statistics.fmean(official.values()), 6)
    nine = {k: v for k, v in official.items() if not k.startswith("temporal_localization")}
    if len(nine) == 9:
        # The leaderboard's headline since 2026-07-01. Not comparable to the
        # leaderboard on our custom mixture; never a headline number here.
        summary["mean_9_tasks"] = round(statistics.fmean(nine.values()), 6)
    return summary


def load_predictions(path: Path) -> dict[str, str]:
    preds = {}
    for row in read_jsonl(path):
        preds[row["item_id"]] = row.get("prediction") or ""
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", default="eval", help="dev | eval | path to a slice JSONL")
    ap.add_argument("--pred", required=True, type=Path, help="predictions JSONL")
    ap.add_argument("--tasks", default="all", help="all | shortlist | comma-separated task list")
    ap.add_argument("--prefix", type=int, help="score only the first N videos by priority")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    args = ap.parse_args()

    slice_file = Path(args.slice) if args.slice.endswith(".jsonl") else slice_path(args.slice)
    rows = select_rows(read_jsonl(slice_file), args.tasks, args.prefix)
    items = score_items(rows, load_predictions(args.pred), args.allow_missing)
    summary = aggregate(items) | {"slice": str(slice_file), "predictions": str(args.pred),
                                  "tasks": args.tasks, "prefix": args.prefix}
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "items.jsonl", items)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{summary['items']} items over {summary['videos']} videos")
    for k, v in {**summary["official"], **summary["extra"]}.items():
        print(f"  {k:48s} {v:.4f}")
    for task, p in summary["parse_rates"].items():
        print(f"  parse {task:24s} strict {p['strict']:.3f}  lenient {p['lenient']:.3f}  fail {p['fail']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
