# The models

Ten open VLMs, each pinned to a repository revision in
`runner/tar_runner/adapters.py` (the BF16 reference) and
`runner/checkpoints.json` (every other build). `tar-config weights <id>`
fetches what a configuration needs.

| Model key | BF16 reference (revision) | INT4 configuration | Its 4-bit build |
|---|---|---|---|
| `lfm2.5-vl-3b` | [LiquidAI/LFM2.5-VL-3B](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) `35a118d938` | `lfm2.5-vl-3b.sahilchachra-awq`, vLLM, FP16 activations | sahilchachra/LFM2.5-VL-3B-AWQ |
| `north-micro` | [CohereLabs/North-Micro-Vision-Instruct](https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct) `46b719694e` | `north-micro.own-w4a16`, vLLM, FP16 | ours (RTN) |
| `cosmos3-edge` | [nvidia/Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge) `344d602b12` | `cosmos3-edge.own-w4a16`, vLLM, FP16 | ours (RTN) |
| `qwen3-vl-8b` | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) `0c351dd01e` | `qwen3-vl-8b.cyankiwi-awq4`, vLLM, FP16 | cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit |
| `qwen3.5-9b` | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) `c202236235` | `qwen3.5-9b.cyankiwi-awq4`, vLLM, FP16 | cyankiwi/Qwen3.5-9B-AWQ-4bit |
| `molmo2-8b` | [allenai/Molmo2-8B](https://huggingface.co/allenai/Molmo2-8B) `e28fa28597` | `molmo2-8b.own-w4a16`, vLLM, BF16 | ours (RTN) |
| `glm-4.6v-flash` | [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) `411bb4d771` | `glm-4.6v-flash.cyankiwi-awq4`, vLLM, FP16 | cyankiwi/GLM-4.6V-Flash-AWQ-4bit |
| `cosmos3-nano` | [nvidia/Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano) `e59a53c259` | `cosmos3-nano.nathanth-w4a16`, vLLM, FP16 | NathanTH141/Cosmos3-Nano-Reasoner-W4A16 |
| `nemotron-3-nano-omni` | [nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) `e5e9932441` | `nemotron-3-nano-omni.drawais-w4a16`, vLLM, BF16 | drawais/Nemotron-3-Nano-Omni-30B-A3B-W4A16 |
| `qwen3.8-27b` | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) `1d4bf0f2ff` | `qwen3.8-27b.unsloth-gguf`, llama.cpp | unsloth/Qwen3.8-27B-GGUF (Q4_K_M) |

`uv run python -m tar_runner.weights list` (from `runner/`) lists every
build, including the 8-bit ones and the GGUF files of the llama.cpp
settings.

## Our own builds

Where no community 4-bit build loaded in vLLM, we made one with data-free
round-to-nearest (RTN) quantization: 4-bit weights in groups of 32,
symmetric, the language model's linear layers only (the vision tower, the
projector, the embeddings and the output head stay 16-bit). The 8-bit
builds of the sweep (`*.own-w8a16`) use the same method at 8 bits. From
`runner/quantize/`:

```bash
uv run python own_w4a16.py north-micro.own-w4a16     # writes models/own/<tag>/ with quantize.json
cd .. && uv run python -m tar_runner.weights fetch north-micro.own-w4a16
```

`quantize.json` records the tool versions and each output file's sha256.
The BF16 model must be fetched first (`uv run python -m tar_runner.weights
base north-micro`).

The Edge-LLM build of Qwen3-VL-8B (`qwen3-vl-8b.own-edgellm-awq4`) is made
on the Orin with NVIDIA's quantizer and its defaults (INT4 AWQ, 128
calibration samples, the vision tower in FP16); see
[running.md](running.md#runtimes).

## Notes per model

- **Cosmos3-Nano and Cosmos3-Edge were trained on TAR's training data**
  (NVIDIA's model cards), from which our subsets are drawn. The paper reports
  them but does not rank them.
- **Qwen3.5-9B and Qwen3.8-27B explain before they answer** even with
  thinking off. TAR's evaluation script reads the first yes, no or letter in
  a reply, which is often in the explanation, so the paper scores these two
  on their final answer (`correct_final` in the scores) and reports the
  script's reading beside it.
- **Batch-invariant kernels** (bit-reproducible vLLM runs) work for
  Qwen3-VL-8B, Molmo2-8B, Cosmos3-Nano, GLM-4.6V-Flash, Cosmos3-Edge and
  North Micro. LFM2.5-VL-3B (short convolutions), Qwen3.5-9B and Qwen3.8-27B
  (Gated DeltaNet) and Nemotron-3-Nano-Omni (Mamba2) have layers vLLM's
  batch-invariant mode does not cover; their answers vary slightly between
  runs.
- **Thinking**: Qwen3.5-9B, Qwen3.8-27B, GLM-4.6V-Flash, Nemotron-3-Nano-Omni,
  Cosmos3-Nano and Cosmos3-Edge have a thinking mode (`--thinking on`).
  With it on, each reply has one budget of 8,192 tokens (16,384 for
  Nemotron-3-Nano-Omni and the two Qwen3.5/3.8 models) for its trace and its
  answer, and the vendors' sampling settings (`--decoding vendor` alone sets
  those with thinking off).
- **Native video** input (`--mode native_video`, one video instead of
  separate frames) is supported for the Qwen models and Cosmos3-Nano; the
  sweep tested it on Qwen3-VL-8B.
- **Nemotron-3-Nano-Omni and Molmo2-8B run with BF16 activations** at 4 bits:
  vLLM 0.30's Nemotron code needs BF16, and Molmo2 overflows in FP16, even
  unquantized, and degenerates.
- **Qwen3.8-27B** does not fit the 32 GB Orin in vLLM even at 4 bits; on the
  Orin it runs on llama.cpp from a GGUF file. Its BF16 reference ran on the
  A100.
- `kimi-vl-a3b` has an adapter but is not part of the study.
