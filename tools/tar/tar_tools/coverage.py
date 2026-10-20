"""Resolve every annotation `video_id` against what is actually on disk.

This is the authoritative statement of what we have. A video counts as
resolved only if the file exists at the annotated path *and* ffprobe could
read it; anything else is recorded with a reason, per source.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime

from .annotations import load_task
from .paths import derived_dir, ensure_dirs, reports_dir, videos_dir
from .probe_media import PROBE_FILE, load_existing
from .sources import BY_SUBDIR, SOURCES


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def pct(p: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))], 2)

    return {"n": len(ordered), "min": round(ordered[0], 2), "p25": pct(25),
            "median": pct(50), "p75": pct(75), "p95": pct(95),
            "max": round(ordered[-1], 2), "mean": round(statistics.fmean(ordered), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    _ = args

    ensure_dirs()
    root = videos_dir()
    probes = load_existing(derived_dir() / PROBE_FILE)

    _, items = load_task("mcq")  # one item per video: the complete video list
    video_ids = sorted({i.video_id for i in items})

    rows = []
    for vid in video_ids:
        source = vid.split("/", 1)[0]
        path = root / vid
        probe = probes.get(vid)
        if probe and probe.get("ok"):
            status, reason = "resolved", ""
        elif probe and not probe.get("ok"):
            status, reason = "corrupt", probe.get("error", "")[:120]
        elif path.exists():
            status, reason = "unprobed", "file present, ffprobe sweep not run"
        else:
            status, reason = "missing", "no file at annotated path"
        rows.append({
            "video_id": vid, "source": source, "status": status,
            "bytes": (probe or {}).get("bytes", ""),
            "duration_s": (probe or {}).get("duration_s", ""),
            "width": (probe or {}).get("width", ""),
            "height": (probe or {}).get("height", ""),
            "fps": (probe or {}).get("fps", ""),
            "codec": (probe or {}).get("codec", ""),
            "reason": reason,
        })

    csv_path = reports_dir() / "coverage.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---- per-source summary ------------------------------------------------
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source"]].append(row)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "videos_root": str(root),
        "total_annotated_videos": len(rows),
        "by_status": dict(Counter(r["status"] for r in rows)),
        "sources": {},
    }
    for source in sorted(by_source):
        srows = by_source[source]
        resolved = [r for r in srows if r["status"] == "resolved"]
        durations = [float(r["duration_s"]) for r in resolved if r["duration_s"] != ""]
        expected = getattr(BY_SUBDIR.get(source), "expected_videos", None)
        summary["sources"][source] = {
            "tier": getattr(BY_SUBDIR.get(source), "tier", "?"),
            "annotated": len(srows),
            "expected_from_registry": expected,
            "by_status": dict(Counter(r["status"] for r in srows)),
            "coverage_pct": round(100 * len(resolved) / len(srows), 1),
            "hours": round(sum(durations) / 3600, 2),
            "duration_s": describe(durations),
            "resolutions_top5": dict(
                Counter(f"{r['width']}x{r['height']}" for r in resolved).most_common(5)),
            "fps_top5": dict(Counter(str(r["fps"]) for r in resolved).most_common(5)),
            "codecs": dict(Counter(r["codec"] for r in resolved).most_common()),
        }

    resolved_all = [r for r in rows if r["status"] == "resolved"]
    all_durations = [float(r["duration_s"]) for r in resolved_all if r["duration_s"] != ""]
    summary["overall"] = {
        "resolved": len(resolved_all),
        "coverage_pct": round(100 * len(resolved_all) / len(rows), 1),
        "hours": round(sum(all_durations) / 3600, 2),
        "hours_claimed_by_card": 26.1,
        "duration_s": describe(all_durations),
    }

    json_path = reports_dir() / "coverage_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    (reports_dir() / "coverage_summary.md").write_text(render_markdown(summary))

    print(f"{summary['overall']['resolved']}/{len(rows)} videos resolved "
          f"({summary['overall']['coverage_pct']}%), "
          f"{summary['overall']['hours']} h of {summary['overall']['hours_claimed_by_card']} h claimed")
    for source, s in summary["sources"].items():
        print(f"  {source:28s} tier {s['tier']}  {s['coverage_pct']:5.1f}%  "
              f"{s['hours']:6.2f} h  {s['by_status']}")
    print(f"wrote {csv_path}\n      {json_path}")
    return 0


def render_markdown(summary: dict) -> str:
    o = summary["overall"]
    lines = [
        "# TAR video coverage",
        "",
        f"Generated {summary['generated_at']} from `{summary['videos_root']}`.",
        "",
        f"**{o['resolved']} / {summary['total_annotated_videos']} annotated videos "
        f"resolved ({o['coverage_pct']}%)**, totalling {o['hours']} h against the "
        f"card's claimed {o['hours_claimed_by_card']} h.",
        "",
        f"Status counts: `{summary['by_status']}`",
        "",
        "| Source | Tier | Annotated | Resolved | Coverage | Hours | Median dur (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, s in summary["sources"].items():
        lines.append(
            f"| {source} | {s['tier']} | {s['annotated']} | "
            f"{s['by_status'].get('resolved', 0)} | {s['coverage_pct']}% | "
            f"{s['hours']} | {s['duration_s'].get('median', '-')} |"
        )
    lines += ["", "## Per-source detail", "", "```json",
              json.dumps(summary["sources"], indent=2), "```", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
