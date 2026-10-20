"""Build and freeze the dev and eval slices.

Slices are built around videos: a video is chosen, and every question TAR
asks about it comes along, minus items whose timestamps point past the end
of the clip. A run then picks which tasks to evaluate.

Strata are source × anomalous/normal. Both slices target a 50% anomalous
share (the corpus is ~26%), cap the two single-camera sources, and
otherwise follow availability. The eval slice is frozen in a priority
order such that every prefix is itself a stratified sample, so a sweep cell
can run the first N videos its budget allows without re-sampling.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime

from tar_tools.annotations import TASKS, load_all_tasks, parse_timestamp
from tar_tools.frames import duration_of
from tar_tools.sample import resolved_videos
from tar_tools.timestamps import TIMESTAMP, TOLERANCE_S, overruns

from . import anomaly
from .common import REPORTS_DIR, SEED, SHORTLIST, item_id, sha256_file, slice_path, write_jsonl

DROPPED_SOURCES = {"Vad-R1"}                 # mostly campus walkways and web violence clips
#: so-tad/test/47.mp4 carries two annotation sets (the organizers re-anchored
#: so-tad/train/106.mp4's items onto it), including the same scene
#: question twice with two different answers. One video; dropped.
DROPPED_VIDEOS = {"so-tad/test/47.mp4"}
SINGLE_CAMERA = ("HTV", "barbados_traffic_challenge")
TARGETS = {
    # slice: (videos, anomalous share, cap per single-camera source)
    "dev": (100, 0.5, 4),
    "eval": (1000, 0.5, 40),
}
PREFIXES = (250, 500, 1000)


def allocate(available: dict, total: int, caps: dict) -> dict:
    """Split `total` across strata in proportion to availability, respecting
    caps; strata that would exceed their cap are pinned to it and the rest
    re-shared. Largest-remainder rounding keeps the total exact."""
    limits = {s: min(n, caps.get(s, n)) for s, n in available.items()}
    active = {s for s, n in limits.items() if n > 0}
    pinned: dict = {}
    target = total
    while True:
        pool = sum(available[s] for s in active)
        shares = {s: target * available[s] / pool for s in active}
        over = {s for s in active if shares[s] > limits[s]}
        if not over:
            break
        for s in over:
            pinned[s] = limits[s]
            target -= limits[s]
            active.discard(s)
    floors = {s: int(shares[s]) for s in active}
    remainder = target - sum(floors.values())
    for s in sorted(active, key=lambda s: shares[s] - floors[s], reverse=True):
        if remainder == 0:
            break
        if floors[s] < limits[s]:
            floors[s] += 1
            remainder -= 1
    if remainder:
        raise ValueError(f"cannot place {remainder} videos within availability and caps")
    return pinned | floors


def interleave(strata_videos: dict, rng: random.Random) -> list[str]:
    """Order videos so every prefix carries each stratum near its share.

    Within a stratum of size n, the i-th video gets key (i + 0.5) / n; sorting
    all videos by key interleaves the strata proportionally."""
    keyed = []
    for videos in strata_videos.values():
        n = len(videos)
        keyed += [((i + 0.5) / n, rng.random(), v) for i, v in enumerate(videos)]
    return [v for _, _, v in sorted(keyed)]


def cites_time(question: str, duration: float) -> bool:
    for m in TIMESTAMP.finditer(question):
        try:
            if parse_timestamp(m.group(0)) <= duration + TOLERANCE_S:
                return True
        except ValueError:
            continue
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="rebuild into outputs/eval/subsets/ and compare with the frozen subsets/")
    args = ap.parse_args()
    out_dir = REPORTS_DIR / "subsets" if args.check else None

    by_task = load_all_tasks()
    labels = anomaly.label_videos(by_task)
    pool = {v for v in resolved_videos()
            if v.split("/", 1)[0] not in DROPPED_SOURCES and v not in DROPPED_VIDEOS}

    # ---- items per video, minus timestamp overruns -------------------------
    items_by_video: dict[str, list] = defaultdict(list)
    excluded: Counter = Counter()
    for task in TASKS:
        for it in by_task[task]:
            if it.video_id not in pool:
                continue
            dur = duration_of(it.video_id)
            if overruns(it.question, dur) or overruns(it.answer, dur):
                excluded[task] += 1
                continue
            items_by_video[it.video_id].append(it)
    pool = {v for v in pool if items_by_video[v]}

    # ---- strata ------------------------------------------------------------
    def stratum(v: str) -> tuple[str, str]:
        return (v.split("/", 1)[0], "anomalous" if labels[v]["anomalous"] else "normal")

    strata: dict[tuple, list[str]] = defaultdict(list)
    for v in sorted(pool):
        strata[stratum(v)].append(v)
    for key, videos in strata.items():
        random.Random(f"{SEED}:{key[0]}:{key[1]}").shuffle(videos)

    # ---- allocation --------------------------------------------------------
    allocation: dict[str, dict] = {}
    for name, (n_videos, anom_share, cam_cap) in TARGETS.items():
        n_anom = round(n_videos * anom_share)
        alloc = {}
        for cls, n in (("anomalous", n_anom), ("normal", n_videos - n_anom)):
            avail = {k: len(v) for k, v in strata.items() if k[1] == cls}
            caps = {k: cam_cap for k in avail if k[0] in SINGLE_CAMERA}
            alloc |= allocate(avail, n, caps)
        allocation[name] = alloc

    chosen: dict[str, dict[tuple, list[str]]] = {"dev": {}, "eval": {}}
    for key, videos in strata.items():
        d = allocation["dev"].get(key, 0)
        e = allocation["eval"].get(key, 0)
        if d + e > len(videos):
            raise ValueError(f"stratum {key}: need {d + e}, have {len(videos)}")
        chosen["dev"][key] = videos[:d]
        chosen["eval"][key] = videos[d:d + e]

    # ---- freeze ------------------------------------------------------------
    summary: dict = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": SEED,
        "dropped_sources": sorted(DROPPED_SOURCES),
        "dropped_videos": sorted(DROPPED_VIDEOS),
        "targets": {k: {"videos": v[0], "anomalous_share": v[1], "single_camera_cap": v[2]}
                    for k, v in TARGETS.items()},
        "pool_videos": len(pool),
        "timestamp_overrun_items_excluded_from_pool": dict(excluded),
        "slices": {},
    }
    for name in ("dev", "eval"):
        order = interleave(chosen[name], random.Random(f"{SEED}:order:{name}"))
        rows = []
        for rank, vid in enumerate(order, 1):
            src, cls = stratum(vid)
            dur = duration_of(vid)
            for it in sorted(items_by_video[vid], key=lambda i: (TASKS.index(i.task), i.question)):
                rows.append({
                    "item_id": item_id(it.task, vid, it.question),
                    "slice": name,
                    "priority": rank,
                    "video_id": vid,
                    "source": src,
                    "anomalous": cls == "anomalous",
                    "duration_s": dur,
                    "task": it.task,
                    "shortlist": it.task in SHORTLIST,
                    "cites_time": cites_time(it.question, dur),
                    "question": it.question,
                    "answer": it.answer,
                })
        ids = [r["item_id"] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name}: duplicate item ids")
        path = out_dir / f"{name}.jsonl" if out_dir else slice_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(path, rows)

        strata_counts = Counter(f"{stratum(v)[0]}|{stratum(v)[1]}" for v in order)
        prefix_check = {}
        for n in PREFIXES:
            if n > len(order):
                continue
            head = Counter(f"{stratum(v)[0]}|{stratum(v)[1]}" for v in order[:n])
            prefix_check[n] = {
                "anomalous_share": round(sum(c for k, c in head.items() if k.endswith("anomalous")) / n, 3),
                "max_abs_share_deviation": round(max(
                    abs(head[k] / n - strata_counts[k] / len(order)) for k in strata_counts), 4),
            }
        summary["slices"][name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "videos": len(order),
            "items": len(rows),
            "items_shortlist": sum(r["shortlist"] for r in rows),
            "items_per_task": dict(Counter(r["task"] for r in rows)),
            "items_citing_time_per_task": dict(Counter(r["task"] for r in rows if r["cites_time"])),
            "strata": dict(sorted(strata_counts.items())),
            "anomalous_share": round(sum(labels[v]["anomalous"] for v in order) / len(order), 3),
            "hours_of_video": round(sum(duration_of(v) for v in order) / 3600, 2),
            "prefix_check": prefix_check,
        }

    dev_videos = {r for k in chosen["dev"].values() for r in k}
    eval_videos = {r for k in chosen["eval"].values() for r in k}
    summary["dev_eval_overlap"] = len(dev_videos & eval_videos)

    anomaly.write_report(anomaly.agreement_report(labels, pool))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "slices.json").write_text(json.dumps(summary, indent=2) + "\n")
    (REPORTS_DIR / "slices.md").write_text(render(summary))

    for name, s in summary["slices"].items():
        print(f"{name:5s} {s['videos']:5d} videos {s['items']:6d} items "
              f"({s['items_shortlist']} shortlist)  anomalous {s['anomalous_share']:.3f}  "
              f"{s['hours_of_video']} h  sha256 {s['sha256'][:12]}")
        for n, c in s["prefix_check"].items():
            print(f"      prefix {n:5d}: anomalous {c['anomalous_share']:.3f}, "
                  f"max stratum deviation {c['max_abs_share_deviation']:.4f}")
    print(f"dev/eval video overlap: {summary['dev_eval_overlap']}")
    if args.check:
        bad = 0
        for name, s in summary["slices"].items():
            frozen = sha256_file(slice_path(name))
            same = frozen == s["sha256"]
            bad += not same
            print(f"{name}: {'identical to' if same else 'DIFFERS from'} subsets/{name}.jsonl")
        return 1 if bad else 0
    return 0


def render(s: dict) -> str:
    lines = [
        "# Eval slices",
        "",
        f"Generated {s['generated_at']}, seed {s['seed']}, by `uv run tar-eval-slices` "
        "(see `eval/tar_eval/slices.py`). Vad-R1 dropped; timestamp-overrun items "
        "excluded per item.",
        "",
        "| Slice | Videos | Items | Shortlist items | Anomalous | Hours | sha256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, x in s["slices"].items():
        lines.append(f"| {name} | {x['videos']} | {x['items']} | {x['items_shortlist']} | "
                     f"{x['anomalous_share']:.0%} | {x['hours_of_video']} | `{x['sha256'][:16]}` |")
    lines += ["", f"Dev/eval video overlap: **{s['dev_eval_overlap']}**.", "",
              "## Strata (videos)", "", "| Stratum | dev | eval |", "|---|---:|---:|"]
    keys = sorted(set(s["slices"]["dev"]["strata"]) | set(s["slices"]["eval"]["strata"]))
    for k in keys:
        lines.append(f"| {k.replace('|', ' / ')} | {s['slices']['dev']['strata'].get(k, 0)} | "
                     f"{s['slices']['eval']['strata'].get(k, 0)} |")
    lines += ["", "## Items per task", "", "| Task | Shortlist | dev | eval | eval citing a time |",
              "|---|:-:|---:|---:|---:|"]
    for t in TASKS:
        lines.append(f"| {t} | {'yes' if t in SHORTLIST else ''} | "
                     f"{s['slices']['dev']['items_per_task'].get(t, 0)} | "
                     f"{s['slices']['eval']['items_per_task'].get(t, 0)} | "
                     f"{s['slices']['eval']['items_citing_time_per_task'].get(t, 0)} |")
    lines += ["", "## Eval prefixes", "",
              "Every prefix of the priority order should be a stratified sample.", "",
              "| Prefix | Anomalous share | Max stratum-share deviation |", "|---:|---:|---:|"]
    for n, c in s["slices"]["eval"]["prefix_check"].items():
        lines.append(f"| {n} | {c['anomalous_share']:.1%} | {c['max_abs_share_deviation']:.2%} |")
    lines += ["", "## Excluded from the pool (timestamp past clip end)", "",
              "```json", json.dumps(s["timestamp_overrun_items_excluded_from_pool"], indent=2), "```"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
