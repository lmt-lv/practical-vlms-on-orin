# Running a configuration

Every configuration in the paper is one row of
[`configs/configs.tsv`](../configs/configs.tsv), with an id such as
`pivot.qwen3-vl-8b`. `tar-config` (in `runner/`) turns an id into the
command the study ran. Everything here runs on one machine: a Linux PC with
a discrete NVIDIA GPU, or a Jetson AGX Orin.

## The machines

| | Discrete GPU | Jetson AGX Orin |
|---|---|---|
| Study | NVIDIA A100 80 GB (quality); RTX PRO 4000 Blackwell 24 GB (checks) | AGX Orin Developer Kit, 32 GB, JetPack 7.2.1 (Jetson Linux 39.2.1, CUDA 13.2, TensorRT 10.16.2) |
| Python environment | `runner/uv.lock`: vLLM 0.30.0, PyTorch 2.13 with its own CUDA 13.0 | the same lock (PyPI's aarch64 wheels) |
| Runtimes | vLLM, llama.cpp | vLLM, llama.cpp, TensorRT Edge-LLM |
| Measures | quality (`tar-config run`) | time, power and energy (`tar-config bench`), and quality |

A configuration's quality does not depend on the machine within the
study's margins: answers on the A100 and the Orin were identical for
98.8–100% of the exactly scored questions with vLLM, and 97.1% with
llama.cpp. The BF16 references of Nemotron-3-Nano-Omni and Qwen3.8-27B, and
the 16-bit sweep settings, need more memory than the Orin has.

## Steps

1. TAR, the subsets and the frame cache: [data.md](data.md), [subsets.md](subsets.md).
2. The runner's environment, from `runner/`:
   ```bash
   uv sync
   ```
3. A configuration's weights, at their pinned revisions, into `models/`:
   ```bash
   uv run tar-config show pivot.qwen3-vl-8b      # what it runs
   uv run tar-config weights pivot.qwen3-vl-8b   # its BF16 model and its 4-bit build
   ```
   Gated models need `uvx --from huggingface_hub hf auth login` first. Our
   own builds (tags with `own-`) are made locally; `tar-config weights`
   prints the command ([models.md](models.md)).
4. Run it:
   ```bash
   uv run tar-config run pivot.qwen3-vl-8b                 # all 1,000 evaluation videos
   uv run tar-config run pivot.qwen3-vl-8b --videos 20     # the first 20, a sample
   ```
   Predictions go to `runs/<group>/<run name>/predictions.jsonl`, with
   `run.json` (the configuration, versions and hashes) and `videos.jsonl`
   (per-video timing). A run resumes where it stopped, and a finished one
   returns at once. `--videos-per-batch 8` speeds up long replies (thinking)
   on vLLM without changing them.
5. Score it, from `eval/`, and compare it with the published scores:
   ```bash
   uv run tar-eval-score --slice eval --prefix 20 \
       --pred ../runs/baseline/<run name>/predictions.jsonl --out ../runs/baseline/<run name>/scored_p20
   uv run tar-eval-results score ../runs/baseline/<run name>/scored_p20/items.jsonl --like pivot.qwen3-vl-8b
   ```
   Score a sample with `--prefix N`, the number of videos it ran. The first
   scoring run downloads the BERTScore model; prefix the command with
   `CUDA_VISIBLE_DEVICES=` to score on the CPU while the GPU is busy.

`tar-config list --group <group>` lists the configurations of one group:
`baseline` (BF16 references, INT4 configurations, question only),
`sweep` (the three models' settings, on the first 250 evaluation videos),
`tarbench`, `thinking` (on and off, development subset) and `runtime`
(Edge-LLM).

## Runtimes

**vLLM** is the default and needs nothing beyond `uv sync`. Runs use eager
mode and, for the models that support them, batch-invariant kernels, so a
run's answers do not depend on how its requests are batched. That favors
reproducibility over speed.

**llama.cpp** runs through `llama-server`, built at the study's commit:

```bash
CUDA_ARCH=87 runner/scripts/build_llama_cpp.sh    # 87: Orin; 80: A100; 120: Blackwell
```

It builds into `build/llama.cpp/`, where the runner finds it (or set
`TAR_LLAMA_SERVER`). The configurations with `backend` `llamacpp` start and
stop the server themselves. Their checkpoints are GGUF files
(`tar-config weights` fetches them).

**TensorRT Edge-LLM** (Orin only, version 0.11.0) has its own environment,
`runner/edgellm/` (`uv sync` there), and C++ tools built on the board:

```bash
runner/scripts/build_edgellm.sh
runner/scripts/quantize_edgellm.sh models/Qwen3-VL-8B-Instruct qwen3-vl-8b.own-edgellm-awq4   # ~1.5 h on the Orin
runner/scripts/edgellm_engines.sh qwen3-vl-8b.own-edgellm-awq4
uv run tar-config run edgellm.qwen3-vl-8b
```

Edge-LLM supports x86 GPUs for development only, so the study ran it on
the Orin alone. Its engines are per power mode: build them in the mode you
measure in (`ENGINES_DIR=models/edgellm/<tag>.engines-30w` when building,
`TAR_EDGELLM_ENGINES=` the same path when running). `edgellm_b12.qwen3-vl-8b`,
a video's 12 questions as one request, needs engines built for a batch:
`MAX_BATCH=12 ENGINES_DIR=models/edgellm/qwen3-vl-8b.own-edgellm-awq4.engines-b12
runner/scripts/edgellm_engines.sh qwen3-vl-8b.own-edgellm-awq4`. Edge-LLM's
answers vary slightly from run to run.

## Timing on the Jetson AGX Orin

`tar-config bench` runs a configuration on the development subset under
`tar-bench`, which samples the module's power rails at 10 Hz and records
time, token rates, power, energy and peak memory per video:

```bash
sudo nvpmodel -m 0          # MAXN; -m 2 for 30 W, -m 1 for 15 W
sudo jetson_clocks          # pin the clocks at the mode's caps; again after every reboot
uv run tar-config bench pivot.qwen3-vl-8b                 # 23 videos, the first 3 as warm-up
uv run tar-config bench pivot.qwen3-vl-8b --videos 53     # as for the paper's MAXN and 30 W tables
```

Results go to `runs/bench_<group>/<run name>/` (`bench.json` holds the
medians). As in the study:

- run headless (no desktop session), with nothing else on the board;
- `tar-config bench` sets the memory options the 32 GB board needs: vLLM's
  engine in process, a smaller multimodal cache, fewer glibc arenas, a
  memory share of 0.65 (0.72 and a fixed 0.4 GiB KV cache for
  Nemotron-3-Nano-Omni);
- run Nemotron-3-Nano-Omni right after a reboot: vLLM leaves some memory
  behind after each run, and Nemotron fits only with all of it free;
- for long runs, `systemd-run --user -p OOMPolicy=continue …` keeps the job
  alive after you log out and records an out-of-memory kill.

The study measured the nine vLLM INT4 configurations in all three modes,
Qwen3.8-27B on llama.cpp, Edge-LLM, and the sweep settings that change
speed, at MAXN; the `orin` column of `configs.tsv` lists each one's modes.
At 32 frames or the 896-pixel cap, Nemotron-3-Nano-Omni ran out of memory
partway through.

## Notes

- The BF16 reference configurations (`anchor.*`) were run on the A100 80 GB;
  on a smaller GPU, the larger ones need more memory than it has.
- `tar-run --help` lists every option; `tar-config show <id>` prints the
  exact `tar-run` command of a configuration.
- [`runner/README.md`](../runner/README.md) documents the input contract,
  the adapters and the backends.
