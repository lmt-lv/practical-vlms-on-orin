# Obtaining TAR

TAR (Traffic Anomaly Reasoning) is the dataset of Track 3 of the AI City
Challenge 2026, published by NVIDIA as
[`nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`](https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning).
Its annotations are on Hugging Face; its videos come from eight public
datasets and are fetched from their own hosts. Nothing here redistributes
them.

Everything goes under `data/tar/` (not committed); set `TAR_ROOT` to put it
elsewhere. The frame cache (next step) goes beside it, in `data/frames/`.

## Requirements

- [uv](https://docs.astral.sh/uv/), and `ffmpeg`/`ffprobe` on `PATH`;
- `7z` (`p7zip-full`): so-tad ships as a 51-part split zip;
- for TAD, HTV and barbados (Kaggle): an API token in `~/.kaggle/kaggle.json`;
- about 40 GB of disk for the videos the study uses.

## Steps

From `tools/tar/`:

```bash
uv run tar-fetch-annotations            # the annotations and TAR's helper scripts, with a sha256 manifest
uv run tar-fetch-videos --list          # sources, tiers and expected video counts
uv run tar-fetch-videos --tier a        # so-tad, TAD-benchmark, Vad-R1, Accident-Bench (~33 GB)
uv run tar-fetch-videos --tier b        # TAD, HTV, barbados (Kaggle)
uv run tar-probe-media                  # ffprobe every video -> data/tar/derived/media_probe.jsonl
uv run tar-coverage                     # which annotated videos resolved -> outputs/tar/coverage_summary.md
```

`--only <source>` fetches one source, and `--probe` checks that every source
is reachable without downloading. The study did not use UCF-Crime (tier C,
334 videos and 103 GB, of which 133 show road incidents) or Vad-R1 (mostly
campus walkways and web violence clips); fetching Vad-R1 does no harm, as
the subsets leave it out.

`tar-probe-media` matters: the subsets, the frame cache and the runner read
every video's duration and frame rate from its output.

## TAR-Bench, the test set

TAR-Bench's 80 clips are cut from 17 YouTube videos by TAR's own script,
which comes with the annotations:

```bash
uv run python ../../data/tar/annotations/test/download_test_videos.py --out ../../data/tar/videos
uv run tar-probe-media                  # again, for the new clips
```

The clips land in `data/tar/videos/tar_test/`. One of the YouTube sources is
age-gated: if its download fails, fetch it with yt-dlp and your browser's
session (`--cookies-from-browser <browser>`, which also needs a JavaScript
runtime, `--js-runtimes node`), then rerun the script with
`--only-ytid=<id> --keep-source` pointing at the downloaded file, as the
script's help describes. The test answers are hidden; only the challenge's
server scores them ([results.md](results.md#tar-bench)).

## What the study had

The study's copy resolved 3,335 of the 3,669 annotated videos (90.9%): every
source but UCF-Crime. A copy fetched later can differ if a source has
changed or a YouTube video has gone; `tar-coverage` shows what resolved,
and [subsets.md](subsets.md) checks the subsets against it.
