"""TAR-Bench, the AI City Challenge 2026 Track 3 test set: its slice and the
submission file.

    uv run tar-eval-tarbench slice                      # subsets/tarbench.jsonl
    uv run tar-eval-tarbench slice-extra                # subsets/tarbench_extra.jsonl
    uv run tar-eval-tarbench export --pred ../runs/.../predictions.jsonl [more ...] --out submission.csv

The test set is 960 items on 80 clips with answers redacted
(data/tar/annotations/test/test.json), so only the challenge's eval server
scores it. The slice holds the items whose clip is on disk, in the runner's
slice format, with an empty answer. In the study, one YouTube source was
age-gated at first: its clip went into a second slice, tarbench_extra. `export` writes the official two-column CSV for every one of
the 960 items; items with no prediction (the missing clip's 12) are written
empty and counted, and the result is checked with the official evaluate.py
(format validation only, as answers are redacted).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from tar_tools.paths import annotations_dir, derived_dir, videos_dir

from .common import REPO_ROOT, read_jsonl, slice_path

TEST_DIR = annotations_dir() / "test"
TEST_JSON = TEST_DIR / "test.json"
PROBE = derived_dir() / "media_probe.jsonl"
SHORTLIST = {"bcq", "mcq", "mcq_openended", "causal_linkage", "video_summarization",
             "temporal_localization"}


def test_items() -> list[dict]:
    return json.loads(TEST_JSON.read_text())["items"]


def build_slice() -> int:
    probe = {r["path"]: r for r in read_jsonl(PROBE) if r.get("ok")}
    items = test_items()
    videos = sorted({i["video_id"] for i in items})
    held = [v for v in videos if (videos_dir() / v).exists() and v in probe]
    priority = {v: n for n, v in enumerate(held, 1)}
    rows = [{"item_id": i["item_index"], "slice": "tarbench", "priority": priority[i["video_id"]],
             "video_id": i["video_id"], "source": "tar_test", "anomalous": None,
             "duration_s": probe[i["video_id"]]["duration_s"], "task": i["task_type"],
             "shortlist": i["task_type"] in SHORTLIST, "cites_time": None,
             "question": i["question"], "answer": ""}
            for i in items if i["video_id"] in priority]
    rows.sort(key=lambda r: (r["priority"], r["item_id"]))
    out = slice_path("tarbench")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    missing = [v for v in videos if v not in priority]
    print(f"{out}: {len(rows)} items on {len(held)} of {len(videos)} clips; missing: {missing}")
    return 0


def build_extra_slice() -> int:
    """subsets/tarbench_extra.jsonl: the clips obtained after the
    tarbench slice was built (the age-gated 80th), so the 79-clip slice,
    and every run named after it, stays unchanged. Its runs' predictions are
    merged with the 79-clip run's at export."""
    probe = {r["path"]: r for r in read_jsonl(PROBE) if r.get("ok")}
    have = {r["video_id"] for r in read_jsonl(slice_path("tarbench"))}
    items = test_items()
    new = sorted({i["video_id"] for i in items} - have)
    held = [v for v in new if (videos_dir() / v).exists() and v in probe]
    priority = {v: n for n, v in enumerate(held, 1)}
    rows = [{"item_id": i["item_index"], "slice": "tarbench", "priority": priority[i["video_id"]],
             "video_id": i["video_id"], "source": "tar_test", "anomalous": None,
             "duration_s": probe[i["video_id"]]["duration_s"], "task": i["task_type"],
             "shortlist": i["task_type"] in SHORTLIST, "cites_time": None,
             "question": i["question"], "answer": ""}
            for i in items if i["video_id"] in priority]
    rows.sort(key=lambda r: (r["priority"], r["item_id"]))
    out = slice_path("tarbench_extra")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{out}: {len(rows)} items on {len(held)} clips: {held}")
    return 0


def export(pred: list[Path], out: Path) -> int:
    """One submission from one or more runs' predictions (the 79-clip run and
    the extra slice's): every test item once."""
    preds = {}
    for p in pred:
        for r in read_jsonl(p):
            if r["item_id"] in preds:
                raise SystemExit(f"{r['item_id']} predicted twice ({p})")
            preds[r["item_id"]] = r.get("prediction") or ""
    items = test_items()
    empty = [i["item_index"] for i in items if i["item_index"] not in preds]
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_index", "prediction"])
        for i in items:
            w.writerow([i["item_index"], preds.get(i["item_index"], "")])
    print(f"{out}: {len(items)} rows, {len(empty)} written empty (no prediction)")
    check = subprocess.run([sys.executable, str(TEST_DIR / "evaluate.py"), "--gt", str(TEST_JSON),
                            "--submission", str(out), "--validate"],
                           capture_output=True, text=True, cwd=TEST_DIR)
    print(check.stdout[-2000:], check.stderr[-2000:], sep="\n")
    return check.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("slice")
    sub.add_parser("slice-extra")
    ex = sub.add_parser("export")
    ex.add_argument("--pred", required=True, type=Path, nargs="+")
    ex.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "slice":
        return build_slice()
    if args.cmd == "slice-extra":
        return build_extra_slice()
    return export(args.pred, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
