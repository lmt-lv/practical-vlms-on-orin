"""Fetch the TAR annotation repo and record exactly what we got.

The HF repo holds only JSON and helper scripts (~60 MB), so we snapshot all
of it. What matters is provenance: the commit sha and a sha256 per file, so
any later result can be tied to the exact annotations that produced it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .paths import HF_REPO_ID, annotations_dir, ensure_dirs, reports_dir


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default="main", help="HF revision to pin (default: main)")
    args = ap.parse_args()

    ensure_dirs()
    dest = annotations_dir()

    info = HfApi().dataset_info(HF_REPO_ID, revision=args.revision)
    print(f"repo     : {HF_REPO_ID}")
    print(f"revision : {args.revision} -> {info.sha}")
    print(f"dest     : {dest}")

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=dest,
    )

    files = sorted(p for p in dest.rglob("*") if p.is_file() and ".cache" not in p.parts)
    manifest = {
        "repo_id": HF_REPO_ID,
        "revision_requested": args.revision,
        "commit_sha": info.sha,
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [
            {
                "path": str(p.relative_to(dest)),
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
            for p in files
        ],
    }
    out = reports_dir() / "annotations_manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")

    total = sum(f["bytes"] for f in manifest["files"])
    print(f"\n{len(manifest['files'])} files, {total / 1e6:.1f} MB")
    print(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
