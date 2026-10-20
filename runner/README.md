# tar-runner

Runs a model over a subset and writes predictions that the scorer in
`eval/` reads, and times it on a Jetson. A separate `uv` project: vLLM 0.30
pins torch 2.13 and needs transformers 5.x, while the scorer pins
transformers 4.57.0 for BERTScore parity. The two talk through JSONL only.

```bash
uv run tar-config list                       # the study's configurations (../configs/configs.tsv)
uv run tar-config show pivot.qwen3-vl-8b     # one configuration's tar-run command
uv run tar-config weights pivot.qwen3-vl-8b  # its weights, at pinned revisions
uv run tar-config run pivot.qwen3-vl-8b [--videos N]
uv run tar-config bench pivot.qwen3-vl-8b    # on a Jetson AGX Orin, under tar-bench
```

[docs/running.md](../docs/running.md) has the steps. `tar-config` builds a
`tar-run` command; `tar-run` itself takes any combination:

```bash
uv run tar-run --model qwen3-vl-8b --backend vllm --mode image_list \
    --frames 16 --cap 448 --slice dev --tasks all
uv run tar-run ... --limit-videos 3          # the first 3 videos: a smoke test or a sample
uv run tar-run ... --weights <tag>           # another checkpoint (checkpoints.json)
uv run tar-run ... --thinking on             # models with a thinking mode
uv run tar-run ... --decoding vendor         # the model card's sampling settings
uv run tar-run ... --batch-invariant         # bit-reproducible vLLM runs
uv run tar-run ... --dtype float16           # FP16 activations
uv run tar-run ... --constrained             # vLLM: yes/no, A-D and the interval JSON enforced
uv run tar-run ... --mode native_video       # the frames as one video, where the model supports it
uv run tar-run ... --backend llamacpp --weights <gguf tag> --kv-cache-dtype q8_0   # 8-bit KV cache
uv run tar-probe --model <key>               # no GPU: the prompt, vision tokens per frame
```

**A run's name doesn't say how many videos it covers.** `--limit-videos N`
runs the subset's first N videos by priority and isn't part of the
configuration, so a 250-video run has the same name as the full run of its
configuration, and rerunning without the limit extends it. `run.json`'s
`coverage` says what a run holds. Score a limited run with
`tar-eval-score --prefix N`.

**Other checkpoints** of a model (INT4, INT8, GGUF) are listed in
`checkpoints.json` and run with `--weights <tag>` after
`python -m tar_runner.weights fetch <tag>`. The run uses the checkpoint's
weights and config and the BF16 model's own input side (tokenizer,
processor, chat template), so only the weights differ; the tag becomes part
of the run's configuration and name.

**llama.cpp**: `--backend llamacpp --weights <gguf tag>` runs a GGUF
checkpoint through `llama-server`, built by `scripts/build_llama_cpp.sh`
(or `TAR_LLAMA_SERVER=<path>`). The prompt is the model's own chat template,
with each image placeholder replaced by the server's media marker; frames
go at the size the model's processor gives them, so prompt tokens match
vLLM's. Requests run one at a time on one slot, so a video's questions
reuse its cached prefix.

**llama.cpp and recurrent models.** A hybrid model's recurrent layers
(Gated DeltaNet, short convolutions, Mamba) keep a state that llama.cpp
can't rewind, so a video's questions can reuse its frames only from a
context checkpoint, which llama-server makes only a few tokens before a
prompt's end. For these models the backend probes each video with its
frames plus exactly n_ubatch + 4 filler tokens, which puts a checkpoint
where the questions part. On LFM2.5 that took cache misses from 11 of 12
questions to none, and halved the time per video.

`--videos-per-batch N` submits N videos' questions per generate call, so
long replies (thinking) don't leave the GPU idle; batch-invariant models
give the same replies either way. It drops the per-video prefill probe, so
leave it at 1 for timing runs.

Needs the subsets (`../subsets/`), the frame cache (`data/frames/`, built by
`tar-eval-frames`) and the weights under `models/`.

## Input contract

Fixed in `tar_runner/contract.py`, identical for every model, so that only
the model and its serving configuration vary between runs.

| Mode | Visual input | Notes |
|---|---|---|
| `image_list` | each frame as an image, preceded by `Frame at MM:SS.ss:` | **default**; every model and runtime accepts images |
| `native_video` | the frames as one video, with frame indices and source fps; resampling off | where supported; a sweep setting |
| `no_frames` | none; one fixed line says the video is unavailable | the question-only dataset ablation |

- Frames come from the cache at the run's cap (448 or 896 long side), at
  the positions of the frame-count set (4/8/16/32), in time order, with each
  frame's actual presentation time.
- TAR's question follows the visual input, verbatim, so all questions about
  one video share a prefix and its encoding is reused.
- No system prompt, the model's own chat template, greedy unconstrained
  decoding. Output cap: 1,024 tokens for every task (shorter caps cut off
  Qwen3.5-9B, which explains before it answers even with thinking off). Each
  prediction records whether it hit its cap.
- Thinking off by default. `--thinking on` (models with a thinking mode)
  gives every task one budget for trace and answer (8,192 tokens, or the
  model's own: 16,384 for Qwen3.5, Qwen3.8 and Nemotron); the answer
  after the trace is the prediction, and the trace is kept beside it.
  With thinking off, a reply carrying a trace is flagged (`leaked_trace`).

## Adding a model

Write one adapter class in `tar_runner/adapters.py` and register it in
`MODELS`. The contract, backends and run command are shared.

| Member | Purpose |
|---|---|
| `key`, `hf_id`, `revision` | identity; weights pinned to a repo revision |
| `supported_modes` | which input modes the model accepts |
| `thinking`, `thinking_kwarg`, `thinking_instruction` | how thinking is switched: a chat-template kwarg, or an instruction in the prompt |
| `vendor_sampling` | the model card's generation settings per thinking mode (`--decoding vendor`) |
| `max_model_len` | context for thinking-off runs; raised where 16 frames at the model's default tiling need more |
| `batch_invariant` | whether vLLM's batch-invariant kernels support the model |
| `prompt(content, thinking)` | chat template → prompt string (the default uses the model's processor) |
| `split_reasoning(text)` | reply → (reasoning trace, answer) |
| `vllm_engine_kwargs(mode, n_frames, cap)` | engine settings: multimodal limits, processor kwargs |
| `vllm_mm_data(frames, mode, video)` | `multi_modal_data` for one request |
| `hf_model_class()` | the transformers class for the reference backend |
| `hf_processor_inputs(prompt, frames, mode, video, n_frames, cap)` | processor kwargs for one request |

Watch for processor defaults that resize frames. Qwen3-VL is the worked
example: its `max_pixels` limit is per frame for images but a budget for
the whole clip for video, so a single value silently shrank video frames to
a sixteenth of their size until the two were set separately.

### Batch invariance

vLLM is not reproducible run to run on its default kernels: two identical
dev runs of Qwen3-VL-8B differed on 20 of 59 replies, all long generative
ones. `--batch-invariant` (`VLLM_BATCH_INVARIANT=1`) makes runs
bit-identical, at ~30% more time, and every quality run uses it where it
can. vLLM implements it only for plain-attention backends; models with
Mamba, short-convolution or linear-attention (Gated DeltaNet) layers are
refused, so their run-to-run noise is measured with a repeated run instead.

| Supported | Not supported |
|---|---|
| Qwen3-VL-8B, Molmo2-8B, Cosmos3-Nano, GLM-4.6V-Flash, Cosmos3-Edge, North Micro | LFM2.5-VL-3B (short convolution), Qwen3.5-9B and Qwen3.8-27B (Gated DeltaNet), Nemotron (Mamba2) |

### Compatibility shims

`tar_runner/compat.py` restores one transformers-4 behaviour that Molmo2's
Hub processor relies on (`optional_attributes`), for classes that declare it
only; it is applied on import and, through vLLM's general-plugin entry
point, in every vLLM process. No model code or numerics change.

## Outputs

Under `runs/<group>/<model>-<backend>-<mode>-<frames>f<cap>-...-<subset>-<config hash>/`:

| File | Contents |
|---|---|
| `predictions.jsonl` | per item: `item_id`, `prediction` (raw reply), finish reason, `hit_cap`, prompt / cached / output tokens, first-token top-2 logprobs |
| `videos.jsonl` | per video: one-token prefill probe (≈ video encoding), time for all its questions, token totals |
| `run.json` | manifest: config, model revision, backend and version, engine settings, subset and frame-manifest hashes, git commit, GPU |

The config hash changes whenever anything that affects outputs changes, so
a directory name identifies a configuration. Runs resume where they
stopped.

## Known issues

- vLLM's first-run compilation stalled for over 15 minutes on the Blackwell
  workstation, so eager mode is the default (`--no-eager` to try graphs).
- vLLM's engine process can outlive the interpreter; the run command shuts
  it down explicitly and exits hard once its outputs are flushed.
