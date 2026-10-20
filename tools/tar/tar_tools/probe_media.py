"""ffprobe sweep over everything fetched.

Writes one JSON line per file to `data/tar/derived/media_probe.jsonl`. A file
that exists but does not decode is a miss, not a hit — that distinction is
the whole point of probing rather than trusting the filesystem.

Incremental by default: files already probed with an unchanged size are
skipped, so this can be re-run after each tier without redoing the work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .paths import derived_dir, ensure_dirs, videos_dir

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".webm", ".mov", ".MP4", ".AVI"}
PROBE_FILE = "media_probe.jsonl"


def parse_fps(rate: str | None) -> float | None:
    if not rate or "/" not in rate:
        return None
    num, den = rate.split("/", 1)
    try:
        return round(int(num) / int(den), 3) if int(den) else None
    except ValueError:
        return None


def probe_one(path: Path, root: Path) -> dict:
    rel = str(path.relative_to(root))
    record: dict = {"path": rel, "bytes": path.stat().st_size}
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames",
         "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return record | {"ok": False, "error": proc.stderr.strip()[:200]}
    try:
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        duration = data.get("format", {}).get("duration")
        return record | {
            "ok": True,
            "duration_s": round(float(duration), 3) if duration else None,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "fps": parse_fps(stream.get("avg_frame_rate")),
            "codec": stream.get("codec_name"),
            "nb_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        }
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        return record | {"ok": False, "error": f"unparseable ffprobe output: {exc}"}


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "path" in rec:  # skip the _meta header line
                out[rec["path"]] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-probe everything")
    ap.add_argument("--subdir", help="limit to one source subdirectory")
    args = ap.parse_args()

    ensure_dirs()
    root = videos_dir()
    scan_root = root / args.subdir if args.subdir else root
    if not scan_root.exists():
        raise SystemExit(f"Nothing to probe at {scan_root}")

    out_path = derived_dir() / PROBE_FILE
    existing = {} if args.force else load_existing(out_path)

    files = [p for p in scan_root.rglob("*") if p.is_file() and p.suffix in VIDEO_SUFFIXES]
    todo = [
        p for p in files
        if (rel := str(p.relative_to(root))) not in existing
        or existing[rel].get("bytes") != p.stat().st_size
    ]
    print(f"{len(files)} media files under {scan_root}; {len(todo)} to probe")

    results: list[dict] = []
    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(probe_one, p, root): p for p in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                if n % 250 == 0:
                    print(f"  probed {n}/{len(todo)}")

    merged = existing | {r["path"]: r for r in results}
    with out_path.open("w") as fh:
        fh.write(json.dumps({
            "_meta": {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                      "videos_root": str(root)}
        }) + "\n")
        for rec in sorted(merged.values(), key=lambda r: r["path"]):
            fh.write(json.dumps(rec) + "\n")

    bad = [r for r in merged.values() if not r.get("ok")]
    total_s = sum(r.get("duration_s") or 0 for r in merged.values() if r.get("ok"))
    print(f"probed {len(merged)} files, {len(bad)} unreadable, "
          f"{total_s / 3600:.2f} h of video")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
