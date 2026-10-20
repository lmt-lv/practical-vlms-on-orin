"""Fetch the TAR source videos, by tier or by source.

Delegates to the upstream `download_videos.py` where it already does the
right thing (its post-processing for so-tad, TAD and HTV is non-trivial and
correct), and takes over only for the two HuggingFace sources where the
upstream whole-repo snapshot would pull thousands of files we do not need.

Every run appends to a fetch log with timing, bytes and outcome, so 001b can
report reproduction cost without anyone watching the terminal.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .annotations import load_task
from .paths import annotations_dir, derived_dir, ensure_dirs, videos_dir
from .sources import SOURCES, Source, resolve

FETCH_LOG = "fetch_log.jsonl"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def wanted_paths(subdir: str) -> list[str]:
    """Repo-relative paths the annotations reference for one source."""
    _, items = load_task("mcq")  # one item per video: the cheapest complete list
    prefix = subdir + "/"
    return sorted({i.video_id[len(prefix):] for i in items if i.video_id.startswith(prefix)})


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def log_event(event: dict) -> None:
    derived_dir().mkdir(parents=True, exist_ok=True)
    with (derived_dir() / FETCH_LOG).open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def check_prereqs(sources: list[Source]) -> list[str]:
    """Return human-readable descriptions of anything missing."""
    problems = []
    needed = {n for s in sources for n in s.needs}
    if "ffmpeg" in needed and not have("ffmpeg"):
        problems.append("ffmpeg not on PATH (needed for transcoding/stitching)")
    if "7z" in needed and not (have("7z") or have("7za") or have("7zz")):
        problems.append(
            "7z not on PATH — needed to extract the so-tad split zip. "
            "Install p7zip-full, or use the Info-ZIP fallback "
            "(`zip -s 0 so_tad.zip --out joined.zip`, needs ~2x disk)."
        )
    if "kaggle-token" in needed and not (Path.home() / ".kaggle/kaggle.json").exists():
        problems.append("~/.kaggle/kaggle.json missing (needed for Kaggle sources)")
    return problems


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


def probe(source: Source) -> dict:
    """Cheap reachability check. No bulk payload."""
    result = {"key": source.key, "transport": source.transport}
    try:
        if source.transport == "hf":
            files = HfApi().list_repo_files(source.locator, repo_type="dataset")
            result |= {"ok": True, "repo_files": len(files)}
            if source.selective:
                wanted = wanted_paths(source.subdir)
                exact = [w for w in wanted if w in set(files)]
                swapped = [w for w in wanted
                           if w not in set(files) and w[:-4] + ".avi" in set(files)]
                result |= {
                    "wanted": len(wanted),
                    "exact_hits": len(exact),
                    "resolvable_via_avi": len(swapped),
                    "unresolvable": len(wanted) - len(exact) - len(swapped),
                }
        elif source.transport == "kaggle":
            proc = subprocess.run(
                ["kaggle", "datasets", "files", source.locator, "--csv"],
                capture_output=True, text=True, timeout=120,
            )
            ok = proc.returncode == 0 and "name" in proc.stdout
            result |= {"ok": ok, "detail": (proc.stderr or proc.stdout).strip()[:200]}
        elif source.transport == "http":
            import requests
            resp = requests.head(source.locator, allow_redirects=True, timeout=60)
            size = resp.headers.get("content-length")
            result |= {
                "ok": resp.status_code < 400,
                "status": resp.status_code,
                "content_length": size,
                "gb": round(int(size) / 1e9, 1) if size else None,
            }
        else:  # gdrive
            import requests
            # HEAD is useless here: Drive answers 200 with an empty body for both
            # live and dead ids. A GET returns either the payload or the
            # virus-scan interstitial, and the interstitial means the file is
            # there and large — which is what gdown then confirms through.
            url = f"https://drive.google.com/uc?id={source.locator}&export=download"
            resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
            head = resp.raw.read(2048, decode_content=True).decode("utf8", "replace")
            resp.close()
            ctype = resp.headers.get("content-type", "")
            interstitial = "Virus scan warning" in head or "confirm=" in head
            result |= {
                "ok": resp.status_code < 400 and (interstitial or "text/html" not in ctype),
                "status": resp.status_code,
                "content_type": ctype[:40],
                "confirm_page": interstitial,
            }
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise
        result |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def fetch_selective_hf(source: Source, dry_run: bool = False, limit: int | None = None) -> dict:
    """Fetch only the referenced files from a HuggingFace source.

    Falls back to the .avi sibling when the annotation says .mp4 but the repo
    ships .avi, then transcodes so the annotation path resolves.
    """
    dest = videos_dir() / source.subdir
    repo_files = set(HfApi().list_repo_files(source.locator, repo_type="dataset"))
    wanted = wanted_paths(source.subdir)

    direct = [w for w in wanted if w in repo_files]
    via_avi = [w for w in wanted if w not in repo_files and w[:-4] + ".avi" in repo_files]
    missing = [w for w in wanted if w not in repo_files and w[:-4] + ".avi" not in repo_files]

    if limit:  # spot-fetch: exercise both code paths without the full payload
        direct, via_avi = direct[:limit], via_avi[:limit]

    patterns = direct + [w[:-4] + ".avi" for w in via_avi]
    summary = {
        "wanted": len(wanted), "direct": len(direct),
        "via_avi": len(via_avi), "unresolvable": len(missing),
    }
    if dry_run:
        return summary | {"dry_run": True}

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=source.locator,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=patterns,
    )

    transcoded, failed = 0, []
    for target in via_avi:
        src = dest / (target[:-4] + ".avi")
        dst = dest / target
        if dst.exists():
            transcoded += 1
            continue
        if not src.exists():
            failed.append(target)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(dst)],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and dst.exists():
            transcoded += 1
            # Drop the source .avi: upstream does the same for HTV, and keeping
            # both would double-count duration in the media sweep.
            src.unlink()
        else:
            failed.append(target)

    return summary | {"transcoded": transcoded, "transcode_failed": failed}


def fetch_so_tad(source: Source, dry_run: bool = False) -> dict:
    """Fetch and extract the so-tad split zip.

    Not delegated upstream: their handler skips any source whose destination
    directory is already populated, which a resumed or partially-probed
    download trips over. Same end state as theirs — flat train/ and test/
    plus Appendix.txt, archive parts removed after a verified extraction.
    """
    dest = videos_dir() / source.subdir
    expected = wanted_paths(source.subdir)

    already = sum(1 for w in expected if (dest / w).exists())
    if already == len(expected):
        return {"skipped": "already extracted", "videos": already}
    if dry_run:
        return {"would_fetch_parts": 53, "would_extract_videos": len(expected)}

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=source.locator, repo_type="dataset", local_dir=dest,
        allow_patterns=["so_tad.z*"],
    )

    sevenz = shutil.which("7z") or shutil.which("7za") or shutil.which("7zz")
    if not sevenz:
        return {"error": "7z not on PATH; parts downloaded but not extracted"}

    proc = subprocess.run(
        [sevenz, "x", "-bso0", "-bsp0", "-y", str(dest / "so_tad.zip"), f"-o{dest}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"error": f"7z failed: {proc.stderr.strip()[:300]}"}

    extracted = sum(1 for w in expected if (dest / w).exists())
    result = {"expected": len(expected), "extracted": extracted}
    if extracted == len(expected):  # only reclaim disk once the payload is verified
        parts = sorted(dest.glob("so_tad.z*"))
        freed = sum(p.stat().st_size for p in parts)
        for p in parts:
            p.unlink()
        shutil.rmtree(dest / ".cache", ignore_errors=True)
        result |= {"parts_removed": len(parts), "freed_gb": round(freed / 1e9, 2)}
    else:
        result |= {"parts_kept": "extraction incomplete, archive retained"}
    return result


def fetch_selective_kaggle(source: Source, dry_run: bool = False,
                           limit: int | None = None) -> dict:
    """Fetch only the referenced files from a Kaggle dataset.

    Upstream pulls the whole dataset: for barbados that is 50 GB+ of competition
    data to reach 128 clips of about 1 MB each. The Kaggle API can fetch single
    files, so we ask for exactly what the annotations reference.
    """
    dest = videos_dir() / source.subdir
    wanted = wanted_paths(source.subdir)
    if limit:
        wanted = wanted[:limit]
    todo = [w for w in wanted if not (dest / w).exists()]
    if dry_run:
        return {"wanted": len(wanted), "to_fetch": len(todo), "dry_run": True}

    dest.mkdir(parents=True, exist_ok=True)
    fetched, failed = 0, []
    for n, rel in enumerate(todo, 1):
        proc = subprocess.run(
            ["kaggle", "datasets", "download", "-d", source.locator,
             "-f", rel, "-p", str((dest / rel).parent), "--unzip", "--force"],
            capture_output=True, text=True,
        )
        if (dest / rel).exists():
            fetched += 1
        else:
            failed.append({"file": rel, "error": (proc.stderr or proc.stdout).strip()[:150]})
        if n % 25 == 0:
            print(f"    {n}/{len(todo)} fetched")
    return {"wanted": len(wanted), "fetched": fetched, "failed": len(failed),
            "failures": failed[:5]}


def cleanup_tad_frames(source: Source) -> dict:
    """Remove TAD's source JPG frame tree once the stitched mp4s are verified.

    The frames are ~15 GB and are only an intermediate; upstream leaves them.
    """
    dest = videos_dir() / source.subdir
    # Kaggle unpacks to a nested TAD/TAD/frames/{normal,abnormal}/ tree; the
    # stitched mp4s land flat in TAD/ beside it.
    candidates = [d for d in (dest / "frames", dest / "TAD") if d.exists()]
    if not candidates:
        return {}
    expected = wanted_paths(source.subdir)
    resolved = sum(1 for w in expected if (dest / w).exists())
    if resolved < len(expected):
        return {"frames_kept": f"only {resolved}/{len(expected)} mp4s present"}
    freed = sum(dir_bytes(d) for d in candidates)
    for d in candidates:
        shutil.rmtree(d)
    return {"frames_removed_gb": round(freed / 1e9, 2)}


def cleanup_source_archive(source: Source) -> dict:
    """Drop a downloaded archive once its payload is on disk.

    The upstream script leaves the Drive/HTTP zips in place, which doubles the
    footprint of TAD-benchmark and would cost ~100 GB on UCF-Crime.
    """
    dest = videos_dir() / source.subdir
    if not dest.exists():
        return {}
    expected = wanted_paths(source.subdir)
    resolved = sum(1 for w in expected if (dest / w).exists())
    if not expected or resolved < len(expected):
        return {"archives_kept": f"{resolved}/{len(expected)} videos resolved"}
    freed = 0
    for archive in list(dest.glob("*.zip")) + list(dest.glob("*.tar*")):
        freed += archive.stat().st_size
        archive.unlink()
    return {"archive_freed_gb": round(freed / 1e9, 2)} if freed else {}


def fetch_upstream(source: Source, dry_run: bool = False) -> dict:
    """Delegate to the upstream downloader for one source."""
    script = annotations_dir() / "download_videos.py"
    if not script.exists():
        raise SystemExit(f"{script} missing — run tar-fetch-annotations first.")
    cmd = ["python", str(script), "--out", str(videos_dir()), "--only", source.key]
    if dry_run:
        return {"would_run": " ".join(cmd)}
    proc = subprocess.run(cmd, cwd=script.parent)
    return {"returncode": proc.returncode, "cmd": " ".join(cmd)}


def fetch(source: Source, dry_run: bool = False, limit: int | None = None) -> dict:
    dest = videos_dir() / source.subdir
    before = dir_bytes(dest) if dest.exists() else 0
    started = time.monotonic()

    if source.key == "so_tad":
        detail = fetch_so_tad(source, dry_run=dry_run)
    elif source.key == "barbados":
        detail = fetch_selective_kaggle(source, dry_run=dry_run, limit=limit)
    elif source.selective:
        detail = fetch_selective_hf(source, dry_run=dry_run, limit=limit)
    else:
        if limit:
            print(f"    (--limit ignored: {source.key} is fetched via the upstream script)")
        detail = fetch_upstream(source, dry_run=dry_run)
        if not dry_run:
            detail |= cleanup_source_archive(source)
            if source.key == "tad":
                detail |= cleanup_tad_frames(source)

    event = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source.key,
        "tier": source.tier,
        "dry_run": dry_run,
        "limit": limit,
        "seconds": round(time.monotonic() - started, 1),
        "bytes_before": before,
        "bytes_after": dir_bytes(dest) if dest.exists() else 0,
        "detail": detail,
    }
    if not dry_run:
        log_event(event)
    return event


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------


def cmd_list() -> None:
    print(f"{'key':16s} {'tier':5s} {'transport':10s} {'videos':>7s}  {'download':<26s} needs")
    for s in SOURCES:
        print(f"{s.key:16s} {s.tier:5s} {s.transport:10s} {s.expected_videos:7d}  "
              f"{s.approx_download:<26s} {','.join(s.needs) or '-'}")
    total = sum(s.expected_videos for s in SOURCES)
    for t in ("a", "b", "c"):
        n = sum(s.expected_videos for s in SOURCES if s.tier == t)
        print(f"tier {t}: {n} videos ({100 * n / total:.0f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="show the source registry")
    ap.add_argument("--probe", action="store_true", help="reachability check, no payload")
    ap.add_argument("--only", nargs="+", metavar="KEY", help="fetch specific sources")
    ap.add_argument("--tier", nargs="+", choices=["a", "b", "c"], help="fetch whole tiers")
    ap.add_argument("--dry-run", action="store_true", help="plan the fetch, download nothing")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="spot-fetch at most N files per code path (selective HF sources only)")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return 0

    ensure_dirs()
    selected = resolve(args.only, args.tier)

    if args.probe:
        for source in selected:
            print(json.dumps(probe(source)))
        return 0

    problems = check_prereqs(selected)
    if problems and not args.dry_run:
        print("Missing prerequisites:")
        for p in problems:
            print(f"  - {p}")
        return 2

    print(f"Fetching {len(selected)} source(s) into {videos_dir()}")
    for source in selected:
        print(f"\n=== {source.key} (tier {source.tier}, {source.approx_download}) ===")
        event = fetch(source, dry_run=args.dry_run, limit=args.limit)
        gained = event["bytes_after"] - event["bytes_before"]
        print(f"    {event['seconds']}s, +{gained / 1e9:.2f} GB, {event['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
