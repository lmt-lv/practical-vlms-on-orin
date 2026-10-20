# Practical vision-language models on the Jetson AGX Orin

Code and results for the paper *Practical Vision-Language Models on the
Jetson AGX Orin: A Traffic Video Case Study*. Ten open vision-language
models (VLMs) answer questions about traffic video from the Traffic Anomaly
Reasoning (TAR) dataset on an NVIDIA Jetson AGX Orin, with three runtimes
(vLLM, llama.cpp, TensorRT Edge-LLM), three power modes and a sweep of
deployment settings: weight precision, frame count, resolution, decoding,
input mode and thinking.

Quality was measured on a data-center GPU, whose answers reproduce the
Orin's; time, power and energy on the Orin. This repository has what is
needed to:

1. obtain TAR ([docs/data.md](docs/data.md));
2. rebuild and check our frozen subsets of it ([docs/subsets.md](docs/subsets.md));
3. run any model and setting of the study by its id, on a discrete GPU or a
   Jetson AGX Orin ([docs/running.md](docs/running.md), [docs/models.md](docs/models.md));
4. score runs, check the paper's numbers against the published per-question
   scores, and regenerate the paper's figures and tables
   ([docs/results.md](docs/results.md)).

## Quickstart: the paper's numbers, figures and tables

No GPU and no data download needed: the published results are in `results/`.
Requires [uv](https://docs.astral.sh/uv/).

```bash
cd eval
uv run tar-eval-results check     # recompute every quality number from the per-question scores
uv run tar-eval-figures           # the figures (outputs/figures/) and tables (outputs/tables/*.csv)
```

## Running a configuration

Every configuration in the paper has an id in
[`configs/configs.tsv`](configs/configs.tsv), for example
`pivot.qwen3-vl-8b` (Qwen3-VL-8B at 4 bits on vLLM) or `f32.qwen3-vl-8b`
(the same with 32 frames):

```bash
cd runner
uv run tar-config list                         # every configuration
uv run tar-config weights pivot.qwen3-vl-8b    # its weights, at pinned revisions
uv run tar-config run pivot.qwen3-vl-8b --videos 20   # quality: the first 20 evaluation videos
uv run tar-config bench pivot.qwen3-vl-8b      # on a Jetson AGX Orin: time, power, energy
```

The data, the frames and the order of the steps are in
[docs/running.md](docs/running.md).

## Layout

| Path | What |
|---|---|
| `tools/tar/` | uv project: fetch TAR's annotations and videos, probe the media |
| `eval/` | uv project: subsets, frames, scoring, statistics, figures and tables |
| `runner/` | uv project: `tar-run`, `tar-bench`, `tar-config`, the backends, the quantizer; side environments for Edge-LLM and the HF reference |
| `configs/configs.tsv` | every configuration in the paper |
| `subsets/` | the frozen evaluation (1,000 videos), development (100) and TAR-Bench (80 clips) subsets |
| `results/` | per-question scores, TAR-Bench submissions and server scores, Orin measurements, and `results.json`, the aggregate behind the paper's numbers |
| `docs/` | how to do each step |

Data, frames, weights, runs and outputs are written under `data/`,
`models/`, `runs/` and `outputs/`, none of which are committed.

## Names in the code

The code keeps the names it was written with:

| Code | Paper |
|---|---|
| anchor | the model's BF16 reference |
| pivot | its INT4 configuration |
| slice | subset (`eval`, `dev`, `tarbench`) |
| noframes | question only (no video) |
| closed-form | the mean accuracy over yes/no, multiple-choice and multiple-choice-with-explanation (letter) questions |

## Licenses

The code is licensed under the [Apache License 2.0](LICENSE). The results in
`results/` are licensed under [CC BY 4.0](results/LICENSE). TAR's
annotations, and the questions and answers in `subsets/`, are NVIDIA's,
released under CC BY 4.0; its videos come from eight third-party datasets
under their own terms and are not redistributed here. Model weights are
their publishers', under their licenses.

## Citation

If you use this code or these results, please cite the paper (the
reference will be added on publication) and TAR:

> Traffic Anomaly Reasoning (TAR), AI City Challenge 2026, Track 3.
> `nvidia/PhysicalAI-Traffic-Anomaly-Reasoning` on Hugging Face.
