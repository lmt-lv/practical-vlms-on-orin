"""Check the paper's quality numbers against the published per-question scores.

    uv run tar-eval-results check            # every configuration with scores
    uv run tar-eval-results check pivot.qwen3-vl-8b f32.qwen3-vl-8b
    uv run tar-eval-results score <items.jsonl> --like pivot.qwen3-vl-8b   # your own run

`check` recomputes, from results/scores/, each configuration's exact-answer
score (the mean accuracy over the yes/no, multiple-choice and
multiple-choice-with-explanation letter questions, with its 95% video
bootstrap CI) and each sweep setting's paired change against its BF16
reference, and compares them with results/results.json. `score` does the same
for a run of your own, scored with `tar-eval-score`, beside the published
configuration it reproduces.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from .common import REPO_ROOT, SEED
from .compare import compare

RESULTS = REPO_ROOT / "results"
CLOSED = ("bcq", "mcq", "mcq_openended")
SLOT = {"anchor": "anchor", "pivot": "pivot", "noframes": "question_only"}
#: Within this many points, a recomputed number matches the store's (rounding).
TOLERANCE = 0.06


def read_scores(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    if path.suffix in (".gz", ".csv"):
        with opener(path, "rt") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in ("correct", "correct_final"):
                r[k] = int(r[k]) if r[k] != "" else None
            for k in ("iou", "iou_final", "bertscore_f1"):
                r[k] = float(r[k]) if r[k] != "" else None
        return rows
    return [json.loads(line) for line in path.open() if line.strip()]


def closed_form(items: list[dict], field: str) -> tuple[float, float, float]:
    """The exact-answer score in points, with its 95% CI (videos resampled)."""
    vids: dict[str, np.ndarray] = {}
    for r in items:
        if r["task"] in CLOSED:
            v = vids.setdefault(r["video_id"], np.zeros((len(CLOSED), 2)))
            v[CLOSED.index(r["task"])] += (r[field] or 0, 1)
    arr = np.stack(list(vids.values()))

    def score(a):
        s = a.sum(axis=0)
        return float(np.mean(s[:, 0] / s[:, 1]))

    rng = np.random.default_rng(SEED)
    boot = [score(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return 100 * score(arr), 100 * lo, 100 * hi


def closed_form_delta(ref: list[dict], cand: list[dict]) -> float:
    """The mean of the three exact-answer tasks' paired changes, in points."""
    res = {r["task"]: r["delta"] for r in compare(ref, cand)["results"] if r["metric"] == "correct"}
    return 100 * float(np.mean([res[t] for t in CLOSED]))


def expected(store: dict, cid: str) -> tuple[str, float] | None:
    kind, model = cid.split(".", 1)
    m = store["models"].get(model, {})
    if kind in SLOT and m.get(SLOT[kind], {}).get("closed_form"):
        return "exact-answer", m[SLOT[kind]]["closed_form"]["value"]
    sw = store.get("sweep", {}).get(model, {}).get(kind)
    if sw and sw.get("closed_form_delta") is not None:
        return "change vs BF16", sw["closed_form_delta"]
    if cid == "edgellm.qwen3-vl-8b":
        q = store["runtimes"]["edgellm_quality"]["vs_anchor"]
        return "change vs BF16", float(np.mean([q[f"{t}/correct"]["delta"] for t in CLOSED]))
    return None


def check(ids: list[str]) -> int:
    store = json.loads((RESULTS / "results.json").read_text())
    index = {e["id"]: e for e in json.loads((RESULTS / "index.json").read_text())}
    bad = done = 0
    for cid, e in index.items():
        if ids and cid not in ids or not e["scores"].endswith(".csv.gz"):
            continue
        exp = expected(store, cid)
        if exp is None:
            continue
        what, want = exp
        kind, model = cid.split(".", 1)
        field = "correct_final" if store["models"][model]["explains_first"] else "correct"
        items = read_scores(RESULTS / e["scores"])
        if what == "exact-answer":
            got = closed_form(items, field)[0]
        else:
            ref = read_scores(RESULTS / index[f"anchor.{model}"]["scores"])
            got = closed_form_delta(ref, items)
        ok = abs(got - want) <= TOLERANCE
        bad += not ok
        done += 1
        print(f"{'ok ' if ok else 'BAD'} {cid:38s} {what:15s} recomputed {got:7.2f}  published {want:7.2f}")
    print(f"{done} checked, {bad} mismatched")
    return 1 if bad else 0


def score_own(path: Path, like: str) -> int:
    store = json.loads((RESULTS / "results.json").read_text())
    index = {e["id"]: e for e in json.loads((RESULTS / "index.json").read_text())}
    model = like.split(".", 1)[1]
    field = "correct_final" if store["models"][model]["explains_first"] else "correct"
    mine = read_scores(path)
    pub = read_scores(RESULTS / index[like]["scores"])
    ids = {r["item_id"] for r in mine}
    pub_same = [r for r in pub if r["item_id"] in ids]
    v, lo, hi = closed_form(mine, field)
    pv, plo, phi = closed_form(pub_same, field)
    same = sum(1 for a, b in zip(sorted(mine, key=lambda r: r["item_id"]),
                                 sorted(pub_same, key=lambda r: r["item_id"]))
               if a["task"] in CLOSED and a[field] == b[field])
    n = sum(1 for r in mine if r["task"] in CLOSED)
    print(f"your run      {v:6.2f} [{lo:.2f}, {hi:.2f}]  on {len({r['video_id'] for r in mine})} videos")
    print(f"published     {pv:6.2f} [{plo:.2f}, {phi:.2f}]  ({like}, the same questions)")
    print(f"exact-answer questions scored the same: {same} of {n} ({100 * same / max(n, 1):.1f}%)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("ids", nargs="*")
    s = sub.add_parser("score")
    s.add_argument("items", type=Path, help="scored/items.jsonl from tar-eval-score")
    s.add_argument("--like", required=True, help="the published configuration it reproduces")
    a = ap.parse_args()
    return check(a.ids) if a.cmd == "check" else score_own(a.items, a.like)


if __name__ == "__main__":
    raise SystemExit(main())
