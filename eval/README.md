# tar-eval

Subsets, the frame cache, the scorer, statistics, and the paper's figures and
tables. A separate `uv` project from the runner: the scorer pins
`transformers==4.57.0` for BERTScore parity with TAR's evaluation script,
while vLLM needs transformers 5. The two talk through JSONL files only.

```bash
uv run tar-eval-slices --check             # rebuild the subsets, compare with ../subsets/
uv run tar-eval-frames                     # the frame cache for the development and evaluation subsets
uv run tar-eval-frames --slice tarbench    # ... and TAR-Bench's clips (also tarbench_extra)
uv run tar-eval-frames --verify            # re-hash the cache against its manifest
uv run tar-eval-score --slice eval --pred <run>/predictions.jsonl --out <run>/scored [--prefix N]
uv run tar-eval-compare <ref>/scored/items.jsonl <run>/scored/items.jsonl --out <dir>   # paired, per task
uv run tar-eval-results check              # the paper's quality numbers from ../results/scores/
uv run tar-eval-results score <run>/scored/items.jsonl --like <config id>
uv run tar-eval-figures                    # the paper's figures and tables, into ../outputs/
uv run tar-eval-baselines                  # the trivial baselines (random, most common answer, whole clip)
uv run tar-eval-tarbench export --pred <run>/predictions.jsonl --out submission.csv
uv run --group dev pytest -q
```

Needs TAR's annotations and probed videos from `tools/tar`
([docs/data.md](../docs/data.md)) for the subsets and the frame cache; the
results commands need only `../results/`.

## Subsets

`../subsets/*.jsonl`, one row per question, self-contained:

| Field | |
|---|---|
| `item_id` | 16 hex digits: sha256 of the task, video id and question |
| `priority` | the video's rank; the first N videos form a stratified sample |
| `video_id`, `source`, `anomalous`, `duration_s` | |
| `task`, `shortlist` | all ten question types |
| `cites_time` | whether the question names a moment in the clip |
| `question`, `answer` | verbatim from TAR (TAR-Bench: answer empty) |

Score a sample with `--prefix N` (the first N videos by priority).

## Frame cache

`data/frames/c{448,896}/<video_id without extension>/{00..31}.png`, with
`data/frames/manifest*.jsonl` recording, per frame, its decoded index, its
presentation time `pts_s` (from the first frame), the target time, size and
sha256. The models see each frame with its time.

The frame counts are nested subsets of the 32 positions
(`tar_eval.common.FRAME_SETS`):

| Frames | Positions |
|---:|---|
| 32 | 0–31 |
| 16 | the odd positions |
| 8 | 1, 5, 9, …, 29 |
| 4 | 5, 13, 21, 29 |

The cache is about 15 GB; copy it to another machine rather than rebuild it.

## Predictions and scores

A run writes JSONL with at least `item_id` and `prediction` (the raw
reply). `tar-eval-score` writes `items.jsonl` (per question: correctness on
the evaluation script's reading and on the final answer, IoU, BERTScore,
how the answer parsed) and `summary.json`. The first run downloads the
BERTScore model. `tar-eval-compare` takes two `items.jsonl` files and gives
each task's paired change with a 95% video-bootstrap CI.
