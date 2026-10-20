# tar-tools

Fetching and probing the TAR (Traffic Anomaly Reasoning) dataset,
[`nvidia/PhysicalAI-Traffic-Anomaly-Reasoning`](https://huggingface.co/datasets/nvidia/PhysicalAI-Traffic-Anomaly-Reasoning).
Data lands under `data/tar/` (not committed), or `$TAR_ROOT`; the tools'
reports go to `outputs/tar/`. [docs/data.md](../../docs/data.md) has the
steps in order.

```bash
uv run tar-fetch-annotations       # annotations + TAR's helper scripts + a sha256 manifest
uv run tar-analyze-annotations     # statistics of the annotations
uv run tar-fetch-videos --list     # sources, tiers and expected video counts
uv run tar-fetch-videos --probe    # reachability check, no download
uv run tar-fetch-videos --only htv # one source
uv run tar-fetch-videos --tier a   # a whole tier
uv run tar-probe-media             # ffprobe sweep -> data/tar/derived/media_probe.jsonl
uv run tar-coverage                # which annotated videos resolved
uv run tar-timestamp-check         # questions whose timestamps point past the clip's end
```

## Sources and tiers

Tiers are by access cost. See `tar_tools/sources.py`.

| Tier | Sources | Videos | Download |
|---|---|---:|---|
| A | so-tad, TAD-benchmark, Vad-R1, Accident-Bench | 2,762 | ~33 GB |
| B | TAD, HTV, barbados | 573 | small; needs a Kaggle token |
| C | UCF-Crime | 334 | ~103 GB; not used by the study |

`tar-fetch-videos` delegates to TAR's own `download_videos.py` where it
post-processes a source (so-tad, TAD, HTV), and fetches the two Hugging
Face sources itself, file by file, instead of whole repositories.

## Prerequisites

- `ffmpeg` and `ffprobe`: probing, TAD's frame stitching, HTV's transcoding;
- `7z` (`p7zip-full`): so-tad ships as a 51-part split zip (without it,
  Info-ZIP's `zip -s 0 so_tad.zip --out joined.zip` works, with twice the
  disk);
- Kaggle credentials in `~/.kaggle/kaggle.json` for tier B.

`gdown` and `yt-dlp` come in as Python dependencies.
