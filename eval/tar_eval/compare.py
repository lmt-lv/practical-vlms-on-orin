"""Paired comparison of two scored runs on the same items.

Every study claim is a paired delta: a configuration against its anchor, on
identical items. Questions about the same video are correlated, so the
confidence interval comes from a cluster bootstrap that resamples videos,
not items. McNemar's exact test is reported beside it for binary metrics.

Inputs are the `items.jsonl` files written by `tar-eval-score`: A is the
reference (usually the anchor), B the candidate; deltas are B - A.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import SEED, read_jsonl

#: (task, metric field) pairs compared. Temporal IoU is compared with
#: unparseable predictions scored zero, so both runs cover the same items.
METRICS = (
    ("bcq", "correct"), ("mcq", "correct"),
    ("bcq_openended", "correct"), ("mcq_openended", "correct"),
    ("temporal_localization", "iou0"),
    ("bcq_openended", "bertscore_f1"), ("mcq_openended", "bertscore_f1"),
    ("open_qa", "bertscore_f1"), ("scene_description", "bertscore_f1"),
    ("video_summarization", "bertscore_f1"), ("temporal_description", "bertscore_f1"),
    ("causal_linkage", "bertscore_f1"),
)
BINARY = {"correct"}


def value(rec: dict, field: str) -> float | None:
    if field == "iou0":
        return rec["iou"] if rec["iou"] is not None else 0.0
    return rec.get(field)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_bootstrap(per_video: dict[str, tuple[float, int]], n_boot: int,
                      rng: np.random.Generator) -> tuple[float, float, float]:
    """Delta and 95% CI of sum(d)/sum(n), resampling videos with replacement."""
    sums = np.array([s for s, _ in per_video.values()], dtype=float)
    counts = np.array([n for _, n in per_video.values()], dtype=float)
    delta = sums.sum() / counts.sum()
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(delta), float(lo), float(hi)


def compare(a_items: list[dict], b_items: list[dict], n_boot: int = 2000) -> dict:
    a = {r["item_id"]: r for r in a_items}
    b = {r["item_id"]: r for r in b_items}
    shared = sorted(a.keys() & b.keys())
    rng = np.random.default_rng(SEED)
    results = []
    for task, field in METRICS:
        pairs = [(a[i], b[i]) for i in shared if a[i]["task"] == task]
        pairs = [(x, y) for x, y in pairs if value(x, field) is not None and value(y, field) is not None]
        if not pairs:
            continue
        per_video: dict[str, list] = defaultdict(lambda: [0.0, 0])
        for x, y in pairs:
            pv = per_video[x["video_id"]]
            pv[0] += value(y, field) - value(x, field)
            pv[1] += 1
        delta, lo, hi = cluster_bootstrap({k: tuple(v) for k, v in per_video.items()}, n_boot, rng)
        row = {
            "task": task, "metric": "iou (unparseable = 0)" if field == "iou0" else field,
            "items": len(pairs), "videos": len(per_video),
            "a": sum(value(x, field) for x, _ in pairs) / len(pairs),
            "b": sum(value(y, field) for _, y in pairs) / len(pairs),
            "delta": delta, "ci95": [lo, hi],
        }
        if field in BINARY:
            bb = sum(1 for x, y in pairs if x["correct"] and not y["correct"])
            cc = sum(1 for x, y in pairs if not x["correct"] and y["correct"])
            row |= {"discordant_a_only": bb, "discordant_b_only": cc,
                    "mcnemar_p": mcnemar_exact(bb, cc)}
        results.append(row)
    return {
        "shared_items": len(shared),
        "only_in_a": len(a.keys() - b.keys()),
        "only_in_b": len(b.keys() - a.keys()),
        "bootstrap": {"resamples": n_boot, "unit": "video", "seed": SEED},
        "results": results,
    }


#: The exactly scored tasks, alone and pooled ("closed-form"): the four-run
#: comparison below gates on them.
CLOSED_FORM = ("bcq", "bcq_openended", "mcq", "mcq_openended")


def delta_of_deltas(pivot_a: list[dict], knob_a: list[dict], pivot_b: list[dict], knob_b: list[dict],
                    n_boot: int = 2000, field: str = "correct") -> dict:
    """Does a knob's paired delta on host A equal the same delta on host B?

    D = (knob - pivot on A) - (knob - pivot on B), per task and pooled over
    the closed-form tasks, on the items all four runs share. The 95% CI
    comes from a cluster bootstrap that resamples videos jointly for all
    four runs (one video's four answers stay together). Each host's own
    delta is reported with its CI beside it. `max_abs_d` bounds |D| by the
    items on which the hosts disagree, in either configuration: D can reach
    it only if every one of those flips the same way."""
    runs = [{r["item_id"]: r for r in x} for x in (pivot_a, knob_a, pivot_b, knob_b)]
    shared = sorted(set.intersection(*(set(r) for r in runs)))
    rng = np.random.default_rng(SEED)
    rows = []
    for task in CLOSED_FORM + ("closed-form",):
        tasks = CLOSED_FORM if task == "closed-form" else (task,)
        ids = [i for i in shared if runs[0][i]["task"] in tasks
               and all(value(r[i], field) is not None for r in runs)]
        if not ids:
            continue
        per = {k: defaultdict(lambda: [0.0, 0]) for k in ("d", "a", "b")}
        disagree = 0
        for i in ids:
            pa, ka, pb, kb = (value(r[i], field) for r in runs)
            v = runs[0][i]["video_id"]
            for k, x in (("d", (ka - pa) - (kb - pb)), ("a", ka - pa), ("b", kb - pb)):
                per[k][v][0] += x
                per[k][v][1] += 1
            disagree += (pa != pb) + (ka != kb)
        est = {k: cluster_bootstrap({v: tuple(x) for v, x in per[k].items()}, n_boot, rng) for k in per}
        rows.append({
            "task": task, "items": len(ids), "videos": len(per["d"]),
            "delta_a": est["a"][0], "delta_a_ci95": list(est["a"][1:]),
            "delta_b": est["b"][0], "delta_b_ci95": list(est["b"][1:]),
            "d": est["d"][0], "d_ci95": list(est["d"][1:]),
            "d_ci_contains_0": est["d"][1] <= 0 <= est["d"][2],
            "max_abs_d": disagree / len(ids),
        })
    return {"shared_items": len(shared), "metric": field,
            "bootstrap": {"resamples": n_boot, "unit": "video", "seed": SEED}, "results": rows}


def render(report: dict, name_a: str, name_b: str) -> str:
    lines = [f"# {name_b} vs {name_a}", "",
             f"{report['shared_items']} shared items; deltas are B - A with 95% "
             "cluster-bootstrap CIs over videos.", "",
             "| Task | Metric | Items | A | B | Delta | 95% CI | McNemar p |",
             "|---|---|---:|---:|---:|---:|---|---:|"]
    for r in report["results"]:
        p = f"{r['mcnemar_p']:.3g}" if "mcnemar_p" in r else ""
        lines.append(f"| {r['task']} | {r['metric']} | {r['items']} | {r['a']:.4f} | {r['b']:.4f} | "
                     f"{r['delta']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {p} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="reference items.jsonl (e.g. the anchor)")
    ap.add_argument("b", type=Path, help="candidate items.jsonl")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, help="write comparison.json and comparison.md here")
    args = ap.parse_args()

    report = compare(read_jsonl(args.a), read_jsonl(args.b), args.boot)
    md = render(report, args.a.parent.name or str(args.a), args.b.parent.name or str(args.b))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.out / "comparison.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
