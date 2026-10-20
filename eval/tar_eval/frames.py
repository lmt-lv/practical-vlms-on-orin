"""Build the frame cache for the dev and eval slices.

Per video: 32 frames at uniform positions t = (i + 0.5) * d / 32, where d is
the video stream's own duration, each the decoded frame nearest its target
time, at two long-side caps (448, 896)
that never upscale, stored as PNG. The 16-, 8- and 4-frame sets are nested
subsets of the 32 (see common.FRAME_SETS), so no extra frames are stored.

Every frame's actual presentation time is recorded, relative to the first
frame, alongside its target time — models are told these times, because
most causal_linkage questions cite moments in the clip.

The cache is decoded once and copied to every host, so its bytes are
identical everywhere by construction. `--check-determinism N` re-extracts N
videos into a scratch directory and compares hashes, to show the cache can
also be rebuilt from scratch.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import struct
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from tar_tools.frames import duration_of
from tar_tools.paths import videos_dir

from .common import (
    FRAME_SETS, FRAMES_ROOT, N_FRAMES_MAX, REPORTS_DIR, RESOLUTION_CAPS, SEED,
    frame_dir, read_jsonl, sha256_file, slice_path,
)

MANIFEST = FRAMES_ROOT / "manifest.jsonl"
BITEXACT = ["-pix_fmt", "rgb24", "-fflags", "+bitexact", "-flags:v", "+bitexact"]


def decoded_frame_times(path: Path) -> list[float]:
    """Presentation time of every decoded frame, in decoder output order —
    the same order ffmpeg's select filter counts in."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    times = []
    for line in proc.stdout.split():
        line = line.strip().rstrip(",")
        if line and line != "N/A":
            times.append(float(line))
    return times


def cap_filter(cap: int) -> str:
    return (f"scale='if(gte(iw,ih),min({cap},iw),-2)':"
            f"'if(gte(iw,ih),-2,min({cap},ih))':flags=bicubic")


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        head = fh.read(24)
    return struct.unpack(">II", head[16:24])


def extract(video_id: str, out_root: Path | None = None) -> dict:
    """Extract one video's frames at every cap; return its manifest rows."""
    src = videos_dir() / video_id
    times = decoded_frame_times(src)
    if not times:
        raise RuntimeError(f"{video_id}: no decodable frames")
    t0 = times[0]
    rel = [t - t0 for t in times]
    # Space targets over the video stream itself, not the container: an audio
    # track running longer makes the container duration overshoot the last
    # frame (by 0.8 s on one TAD-benchmark clip), pushing late targets past it.
    steps = sorted(b - a for a, b in zip(rel, rel[1:]) if b > a)
    frame_step = steps[len(steps) // 2] if steps else 0.0
    duration = rel[-1] + frame_step

    targets = [(i + 0.5) * duration / N_FRAMES_MAX for i in range(N_FRAMES_MAX)]
    # nearest decoded frame to each target time
    chosen = []
    j = 0
    for t in targets:
        while j + 1 < len(rel) and abs(rel[j + 1] - t) <= abs(rel[j] - t):
            j += 1
        chosen.append(j)
    unique = sorted(set(chosen))

    select = "+".join(f"eq(n\\,{k})" for k in unique)
    chains = [f"[0:v]select='{select}',scale=iw*sar:ih,setsar=1,split={len(RESOLUTION_CAPS)}"
              + "".join(f"[s{c}]" for c in RESOLUTION_CAPS)]
    chains += [f"[s{c}]{cap_filter(c)}[o{c}]" for c in RESOLUTION_CAPS]

    with tempfile.TemporaryDirectory(prefix="tarframes_") as tmp:
        cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-filter_complex", ";".join(chains)]
        for c in RESOLUTION_CAPS:
            cmd += ["-map", f"[o{c}]", "-fps_mode", "passthrough", *BITEXACT,
                    str(Path(tmp) / f"c{c}_%03d.png")]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        rows = []
        for c in RESOLUTION_CAPS:
            produced = sorted(Path(tmp).glob(f"c{c}_*.png"))
            if len(produced) != len(unique):
                raise RuntimeError(f"{video_id}: expected {len(unique)} frames at cap {c}, "
                                   f"got {len(produced)}")
            by_index = dict(zip(unique, produced, strict=True))
            dest_dir = (out_root / f"c{c}" / video_id.rsplit(".", 1)[0]) if out_root \
                else frame_dir(c, video_id)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for pos, k in enumerate(chosen):
                dest = dest_dir / f"{pos:02d}.png"
                shutil.copyfile(by_index[k], dest)
                w, h = png_size(dest)
                rows.append({
                    "video_id": video_id, "cap": c, "pos": pos, "frame_index": k,
                    "pts_s": round(rel[k], 4), "target_s": round(targets[pos], 4),
                    "width": w, "height": h, "sha256": sha256_file(dest),
                })
    return {"video_id": video_id, "duration_s": duration,
            "container_duration_s": duration_of(video_id), "decoded_frames": len(times),
            "repeated_positions": N_FRAMES_MAX - len(unique), "rows": rows}


#: TAR-Bench's clips get their own manifest, so that adding them leaves the
#: dev/eval manifest — and the hash every dev/eval run records — unchanged.
TARBENCH_MANIFEST = FRAMES_ROOT / "manifest_tarbench.jsonl"
#: The calibration set for our own quants, likewise.
CALIB_MANIFEST = FRAMES_ROOT / "manifest_calib.jsonl"
#: TAR-Bench clips obtained later (the 80th): their own manifest too.
TARBENCH_EXTRA_MANIFEST = FRAMES_ROOT / "manifest_tarbench_extra.jsonl"


def slice_videos(names=("dev", "eval")) -> list[str]:
    videos: dict[str, None] = {}
    for name in names:
        for row in read_jsonl(slice_path(name)):
            videos.setdefault(row["video_id"], None)
    return list(videos)


def build(workers: int, slices=("dev", "eval"), manifest: Path = MANIFEST,
          report_path: Path = REPORTS_DIR / "frame_cache.json") -> int:
    videos = slice_videos(slices)
    print(f"{len(videos)} videos, {N_FRAMES_MAX} frames x {len(RESOLUTION_CAPS)} caps each")
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract, v): v for v in videos}
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append({"video_id": futures[fut], "error": str(exc)[:300]})
            if n % 100 == 0:
                print(f"  {n}/{len(videos)}")

    rows = sorted((r for res in results for r in res["rows"]),
                  key=lambda r: (r["video_id"], r["cap"], r["pos"]))
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # This build's own frames only (the tarbench build shares the frame root).
    files = {(r["cap"], frame_dir(r["cap"], r["video_id"]) / f"{r['pos']:02d}.png") for r in rows}
    total_bytes = sum(f.stat().st_size for _, f in files)
    upscaled = [r for r in rows if r["cap"] == max(RESOLUTION_CAPS)
                and max(r["width"], r["height"]) > max(RESOLUTION_CAPS)]
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "videos": len(results),
        "failures": failures,
        "frames": len(rows),
        "caps": list(RESOLUTION_CAPS),
        "frame_sets": {str(k): list(v) for k, v in FRAME_SETS.items()},
        "slices": list(slices),
        "manifest": manifest.name,
        "manifest_sha256": sha256_file(manifest),
        "cache_gb": round(total_bytes / 1e9, 2),
        "gb_per_cap": {
            str(c): round(sum(f.stat().st_size for cap, f in files if cap == c) / 1e9, 2)
            for c in RESOLUTION_CAPS
        },
        "videos_with_repeated_positions": sum(1 for r in results if r["repeated_positions"]),
        "max_abs_target_error_s": round(max(abs(r["pts_s"] - r["target_s"]) for r in rows), 4),
        "frames_exceeding_cap": len(upscaled),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "frame_sets"}, indent=2))
    return 1 if failures else 0


def verify() -> int:
    """Every manifest entry exists and hashes to what the manifest says."""
    rows = read_jsonl(MANIFEST)
    bad = [r for r in rows
           if sha256_file(frame_dir(r["cap"], r["video_id"]) / f"{r['pos']:02d}.png") != r["sha256"]]
    print(f"{len(rows)} frames checked, {len(bad)} mismatched")
    return 1 if bad else 0


def check_determinism(n: int, workers: int = 8) -> int:
    """Re-extract n random videos from scratch and compare against the manifest."""
    manifest = {(r["video_id"], r["cap"], r["pos"]): r["sha256"] for r in read_jsonl(MANIFEST)}
    videos = sorted({k[0] for k in manifest})
    sample = random.Random(SEED).sample(videos, min(n, len(videos)))

    def one(v: str) -> int:
        with tempfile.TemporaryDirectory(prefix="tarframes_check_") as tmp:
            return sum(r["sha256"] != manifest[(v, r["cap"], r["pos"])]
                       for r in extract(v, out_root=Path(tmp))["rows"])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        mismatched = sum(pool.map(one, sample))
    print(f"{len(sample)} videos re-extracted from scratch, {mismatched} frame hashes differ")
    return 1 if mismatched else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--slice", choices=("dev-eval", "tarbench", "tarbench_extra", "calib"), default="dev-eval",
                    help="tarbench: the test clips, into manifest_tarbench.jsonl; "
                         "calib: the calibration videos, into manifest_calib.jsonl")
    ap.add_argument("--verify", action="store_true", help="re-hash the cache against the manifest")
    ap.add_argument("--check-determinism", type=int, metavar="N",
                    help="re-extract N videos from scratch and compare hashes")
    args = ap.parse_args()
    if args.verify:
        return verify()
    if args.check_determinism:
        return check_determinism(args.check_determinism, args.workers)
    if args.slice == "tarbench_extra":
        return build(args.workers, ("tarbench_extra",), TARBENCH_EXTRA_MANIFEST,
                     REPORTS_DIR / "tarbench_extra_frames.json")
    if args.slice == "tarbench":
        return build(args.workers, ("tarbench",), TARBENCH_MANIFEST,
                     REPORTS_DIR / "tarbench_frames.json")
    if args.slice == "calib":
        return build(args.workers, ("calib",), CALIB_MANIFEST,
                     REPORTS_DIR / "calib_frames.json")
    return build(args.workers)


if __name__ == "__main__":
    raise SystemExit(main())
