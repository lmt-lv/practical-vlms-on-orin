"""Filesystem layout for the TAR dataset.

Everything under the data root and under the report directory is
gitignored. `$TAR_ROOT` overrides the data root so the dataset
can live on another volume without touching the code.
"""

from __future__ import annotations

import os
from pathlib import Path

# tools/tar/tar_tools/paths.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]

HF_REPO_ID = "nvidia/PhysicalAI-Traffic-Anomaly-Reasoning"


def data_root() -> Path:
    """Root of all downloaded data. Never committed."""
    env = os.environ.get("TAR_ROOT")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data" / "tar"


def annotations_dir() -> Path:
    """Annotation JSON plus the upstream helper scripts, as published."""
    return data_root() / "annotations"


def videos_dir() -> Path:
    """Media root: the `--out` directory that annotation video_ids resolve against."""
    return data_root() / "videos"


def derived_dir() -> Path:
    """Generated artefacts too large to commit (ffprobe sweeps, frame samples)."""
    return data_root() / "derived"


def reports_dir() -> Path:
    """Small outputs of the tools (coverage, annotation statistics). Not committed."""
    return REPO_ROOT / "outputs" / "tar"


def ensure_dirs() -> None:
    for d in (annotations_dir(), videos_dir(), derived_dir(), reports_dir()):
        d.mkdir(parents=True, exist_ok=True)
