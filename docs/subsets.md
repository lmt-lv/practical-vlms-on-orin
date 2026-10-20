# The subsets and the frame cache

Only TAR's training split has public answers, so the study drew its
evaluation data from it. The subsets are frozen in `subsets/`:

| File | Videos | Questions | Used for |
|---|---:|---:|---|
| `eval.jsonl` | 1,000 | 11,771 | quality: every baseline on all 1,000; the deployment sweep on the first 250 |
| `dev.jsonl` | 100 | 1,181 | development, thinking on and off, and every measurement on the Orin |
| `tarbench.jsonl` | 79 clips | 948 | TAR-Bench, the test set (answers hidden) |
| `tarbench_extra.jsonl` | 1 clip | 12 | TAR-Bench's age-gated clip, obtained later |

Each row is one question: `item_id`, the subset, the video's `priority`
(its position in the subset's fixed order), `video_id`, `source`, whether
the clip is anomalous, its duration, the task, and TAR's question and
answer.

## How they were drawn

From every source but UCF-Crime and Vad-R1 (six sources of road CCTV, 3,203
videos), stratified by source and by whether TAR's own summary of the clip
describes an anomaly:

- half of each subset is anomalous, against 22% of the available videos;
- the two sources filmed by a single camera, HTV and barbados, are capped at
  40 videos per stratum;
- questions whose timestamps point past the end of their clip are left out
  (689 of them);
- one so-tad clip, which TAR counts twice with two partly contradictory sets
  of questions (`so-tad/test/47.mp4`), is left out;
- the evaluation subset's order makes every prefix, the first 250 videos
  included, a balanced sample of the whole.

The seed and the rules are in `eval/tar_eval/slices.py`.

## Checking them against your copy of TAR

With the annotations and videos fetched and probed ([data.md](data.md)),
from `eval/`:

```bash
uv run tar-eval-slices --check     # rebuilds into outputs/eval/subsets/, compares with subsets/
```

It prints `identical to subsets/<name>.jsonl` for the development and
evaluation subsets when your copy matches the study's. A missing or changed
video changes the draw. In that case use the frozen subsets as they are:
every run reads `subsets/`, and the frame cache needs only the subsets'
videos.

## The frame cache

Every model sees the same frames: 32 positions per clip, of which the 16-,
8- and 4-frame settings take nested subsets, scaled so that the longer side
is at most 448 or 896 pixels. They are extracted once, from `eval/`:

```bash
uv run tar-eval-frames                       # the development and evaluation subsets (~18 GB)
uv run tar-eval-frames --slice tarbench      # TAR-Bench's clips
uv run tar-eval-frames --slice tarbench_extra
uv run tar-eval-frames --verify              # re-hash the cache against its manifest
uv run tar-eval-frames --check-determinism 5 # re-extract 5 videos and compare, bit for bit
```

The cache goes to `data/frames/` (beside `TAR_ROOT`) with a manifest per
subset; every run records the manifest's sha256.
