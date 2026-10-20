# The results

`results/` holds the study's results, under CC BY 4.0 ([LICENSE](../results/LICENSE)).
The models' full text replies are not included.

| Path | What |
|---|---|
| `index.json` | one entry per configuration id: its results file and the run it came from |
| `scores/<id>.csv.gz` | per-question scores of a configuration's quality run |
| `runs/<id>.json` | that run's record: configuration, model revision, versions, subset and frame hashes |
| `thinking_dev/<id>.json` | score summaries of the thinking runs on the development subset |
| `tarbench/<id>.csv` | the TAR-Bench submissions, in the challenge's format |
| `tarbench/server_scores.json` | the evaluation server's scores for each submission |
| `orin/bench.csv` | the Orin's measurements: medians and interquartile ranges per configuration and power mode |
| `results.json` | the aggregate behind every number in the paper |

## Per-question scores

One row per question of the configuration's subset (or of its first 250
videos, for the sweep):

| Column | Meaning |
|---|---|
| `item_id`, `video_id`, `task` | the question, as in `subsets/` |
| `correct` | 1 or 0 for the exactly scored types (yes/no, multiple choice, and the yes/no or letter of the two types with an explanation), read as TAR's evaluation script reads them: the answer at the start of the reply, or else the first yes, no or option letter in it |
| `correct_final` | the same, read on the final answer: the last yes, no or letter in the reply. The paper uses it for Qwen3.5-9B and Qwen3.8-27B, which explain before they answer |
| `iou`, `iou_final` | temporal localization: the intersection over union of the predicted and true intervals; empty when the reply did not parse (the paper counts those as zero) |
| `bertscore_f1` | the open answers' BERTScore, as TAR's evaluation script computes it |
| `parse` | how the answer was read: `strict`, `lenient` or `fail`; `n/a` for the open types |

The **exact-answer score** (`closed_form` in the code) is the mean of three accuracies: yes/no,
multiple choice, and the letter of multiple choice with explanation. Its
95% confidence interval comes from 2,000 bootstrap draws of the videos.

```bash
cd eval
uv run tar-eval-results check                  # every exact-answer score and sweep change, against results.json
uv run tar-eval-results check f32.qwen3-vl-8b  # one configuration
```

## The paper's tables and figures

```bash
cd eval
uv run tar-eval-figures       # outputs/figures/*.pdf, outputs/tables/*.csv
```

| Paper | Output | From |
|---|---|---|
| Table 2 (TAR's sources) | `tables/sources.csv` | `results.json`: `dataset/sources` |
| Table 3 (the models) | `tables/models.csv` | `models/*` |
| Table 4 (baselines) | `tables/baselines.csv` | `anchor.*`, `pivot.*`, `noframes.*` |
| Table 5 (speed and energy on the Orin) | `tables/board.csv` | `orin/bench.csv`, MAXN |
| Table 6 (the deployment settings) | `tables/settings.csv` | the sweep's scores; `orin/bench.csv` |
| Table 7 (runtimes) | `tables/runtimes.csv` | `llama.*`, `edgellm.*` and their Orin rows |
| Table 8 (TAR-Bench) | `tables/tarbench.csv` | `tarbench/server_scores.json` |
| Figure 1 (power modes) | `figures/power_modes.pdf` | `orin/bench.csv`, three modes |
| Figure 2 (quality against speed) | `figures/pareto.pdf` | `pivot.*`, `orin/bench.csv` |
| Figure 3 (each setting, per task) | `figures/sensitivity.pdf` | the sweep's scores against `anchor.*` |

In the CSV tables, a trailing `*` marks the paper's bold (the best model not
trained on TAR in its size class), `^a` a model trained on TAR's training
data and `^b` a model scored on its final answer.

## The Orin's measurements

`orin/bench.csv` has one row per configuration and power mode (`maxn`,
`30w`, `15w`), measured on the development subset after three warm-up
videos, with the clocks pinned:

| Column | Meaning |
|---|---|
| `videos` | measured videos |
| `e2e_s` | time per video, from the frames to the last of its 12 answers |
| `prefill_tok_s` | prompt tokens, frames included, over the time to process them |
| `decode_tok_s` | single-stream decode: 256 tokens for one question, the video's prefix cached |
| `throughput_tok_s` | the 12 questions' output tokens over their wall time |
| `energy_j`, `energy_per_item_j` | module energy per video and per question |
| `mean_module_w`, `idle_w` | mean module power during the run, and at idle before it |
| `peak_mem_mb` | used memory at its peak, above the idle baseline (for vLLM, with its reservation) |
| `*_iqr_low`, `*_iqr_high` | the interquartile range over the measured videos |

Module power is the sum of the module's three input rails, read from the
board's INA3221 monitors at 10 Hz; the carrier board and the power supply
are not included.

## TAR-Bench

TAR-Bench's answers are hidden, so only the challenge's server scores it.
`tarbench/<id>.csv` are the files we submitted (960 rows: `item_index,
prediction`); `tar-eval-tarbench export` (in `eval/`) writes one from a
run's predictions. The evaluation script cannot score a blank answer, which
a reply cut off by the token budget leaves: the thinking-on submission was
resubmitted with each blank replaced by a placeholder that parses as no
answer (`*.resubmitted.csv`). Server scores are on a 0–1 scale; the server's
mean leaves out temporal localization.
