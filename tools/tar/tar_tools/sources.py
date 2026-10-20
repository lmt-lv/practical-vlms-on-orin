"""The eight upstream video sources, with the metadata the upstream script lacks.

Tiering is by access cost, not importance:

  A  no credentials, moderate bytes  -> 75% of the dataset
  B  needs a Kaggle token            -> +16%
  C  UCF-Crime, ~96 GB for 9%        -> deferred by default

`selective` marks sources where we can fetch only the files the annotations
reference instead of the whole upstream repo. For Vad-R1 that is 131 files
out of 8,648; for Accident-Bench, 80 out of 2,325.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    key: str                 #: upstream CLI key, passed to download_videos.py --only
    subdir: str              #: on-disk directory, equals the video_id prefix
    tier: str                #: "a" | "b" | "c"
    transport: str           #: hf | kaggle | gdrive | http
    locator: str             #: repo id, kaggle slug, drive file id, or url
    expected_videos: int     #: distinct videos the annotations reference
    approx_download: str
    selective: bool = False
    needs: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


SOURCES: tuple[Source, ...] = (
    Source(
        key="so_tad", subdir="so-tad", tier="a", transport="hf",
        locator="cccccxy/so-tad", expected_videos=2185, approx_download="~27 GB",
        needs=("7z",),
        note="51-part PKWARE split zip. 60% of the dataset behind one archive — "
             "the highest-consequence download in the task.",
    ),
    Source(
        key="tad_bench", subdir="TAD-benchmark", tier="a", transport="gdrive",
        locator="14GNlNcWLzN-sbzvmrMuSbAg_rZZ5yd26", expected_videos=366,
        approx_download="4.5 GB",
        note="Drive zip ships its own TAD-benchmark/ wrapper; upstream extracts "
             "one level up to collapse it. Drive throttling may force a manual fetch.",
    ),
    Source(
        key="vad_r1", subdir="Vad-R1", tier="a", transport="hf",
        locator="wbfwonderful/Vad-R1", expected_videos=131,
        approx_download="small (131 of 8,648 files)", selective=True,
        needs=("ffmpeg",),
        note="55 of the 131 referenced videos are annotated as .mp4 but ship as "
             ".avi (the ShanghaiTech Normal clips). We fetch the .avi and "
             "transcode; the upstream script does not, and leaves them missing.",
    ),
    Source(
        key="accident_bench", subdir="Accident-Bench", tier="a", transport="hf",
        locator="Open-Space-Reasoning/AccidentBench", expected_videos=80,
        approx_download="small (80 of 2,325 files)", selective=True,
        note="Paths map exactly; includes non-road 'planes', so it carries some "
             "non-traffic content despite its size.",
    ),
    Source(
        key="htv", subdir="HTV", tier="b", transport="kaggle",
        locator="aryashah2k/highway-traffic-videos-dataset", expected_videos=254,
        approx_download="~60 MB", needs=("kaggle-token", "ffmpeg"),
        note="CC0. 254 msmpeg4v1 .avi files transcoded to flat .mp4. Cheapest "
             "source that exercises the most machinery — our end-to-end canary.",
    ),
    Source(
        key="tad", subdir="TAD", tier="b", transport="kaggle",
        locator="nikanvasei/traffic-anomaly-dataset-tad", expected_videos=191,
        approx_download="medium", needs=("kaggle-token", "ffmpeg"),
        note="Ships as JPG frame folders; stitched to mp4 at 30 fps by "
             "stitch_tad_frames.py.",
    ),
    Source(
        key="barbados", subdir="barbados_traffic_challenge", tier="b",
        transport="kaggle", locator="kipngetichv/reencoded-barbados-traffic",
        expected_videos=128, approx_download="~130 MB selective (50 GB+ whole)",
        selective=True, needs=("kaggle-token",),
        note="Only the normanniles1/ subfolder is referenced.",
    ),
    Source(
        key="ucf_crime", subdir="UCF_Crimes", tier="c", transport="http",
        locator="http://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip",
        expected_videos=334, approx_download="~96 GB",
        note="64% of the total download for 9% of the videos, and the source "
             "carrying most of the non-traffic content. Deferred by default.",
    ),
)

BY_KEY = {s.key: s for s in SOURCES}
BY_SUBDIR = {s.subdir: s for s in SOURCES}
TIERS = ("a", "b", "c")


def tier(name: str) -> list[Source]:
    return [s for s in SOURCES if s.tier == name]


def resolve(keys: list[str] | None, tiers: list[str] | None) -> list[Source]:
    """Select sources by explicit key, by tier, or all of A and B."""
    if keys:
        unknown = set(keys) - set(BY_KEY)
        if unknown:
            raise SystemExit(f"Unknown source key(s): {sorted(unknown)}")
        return [BY_KEY[k] for k in keys]
    if tiers:
        return [s for s in SOURCES if s.tier in tiers]
    return [s for s in SOURCES if s.tier in ("a", "b")]
