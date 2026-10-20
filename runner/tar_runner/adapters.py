"""Model adapters: everything model-specific, and nothing else.

An adapter knows where the weights are, how to turn the contract's chat
content into the model's prompt, and how to hand frames to each backend.
Adding a model means adding one adapter class here and registering it in
MODELS; the contract, backends and run command are shared.

Interface (see `Adapter`):
  key, hf_id, revision, local_dir     identity, pinned by repo revision
  supported_modes                     which input modes the model accepts
  thinking, thinking_modes            how the model's thinking mode is switched
  vendor_sampling                     the model card's recommended generation settings
  prompt(content, thinking)           chat template -> prompt string
  split_reasoning(text)               reply -> (reasoning trace, answer)
  vllm_engine_kwargs(mode, n, cap)    engine settings for a run
  vllm_mm_data(frames, mode, video)   multi_modal_data for one request
  hf_model_class()                    transformers class for the reference backend
  hf_processor_inputs(...)            processor kwargs for one request
  llamacpp_placeholder()              the chat template's placeholder for one image
  llamacpp_image(frame, cap)          one frame as llama.cpp gets it (the processor's size)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .data import MODELS_ROOT, Frame

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


@dataclass
class Adapter:
    key: str
    hf_id: str
    revision: str
    supported_modes: tuple[str, ...] = ("image_list", "no_frames")
    #: none      no thinking mode
    #: template  a chat-template kwarg switches it (`thinking_kwarg`)
    #: prompt    an instruction in the prompt turns it on (`thinking_instruction`)
    thinking: str = "none"
    thinking_kwarg: str = "enable_thinking"
    thinking_instruction: str = ""
    #: The model card's recommended generation settings, per thinking mode
    #: ("off"/"on"), in vLLM SamplingParams terms; used with --decoding vendor.
    vendor_sampling: dict = field(default_factory=dict)
    trust_remote_code: bool = False
    #: Context for thinking-off runs; thinking-on runs add the thinking budget.
    #: Raised per model when 16 frames at its default tiling need more, never
    #: met by shrinking the model's input.
    max_model_len: int = 8192
    #: vLLM batch-invariant kernels (bit-reproducible runs) exist only for
    #: plain-attention backends; hybrid models with Mamba, short-convolution
    #: or linear-attention (GDN) layers are refused. Tested on 2026-09-28.
    batch_invariant: bool = True
    #: Recurrent layers beside attention (Gated DeltaNet, short convolutions,
    #: Mamba): a state that can't be rewound to an earlier position, unlike a
    #: KV cache. llama.cpp then reuses a video's frames across its questions
    #: only from a checkpoint, which the backend's prefix probe places.
    recurrent: bool = False
    #: Tokens for trace and answer together with thinking on (None: the
    #: contract's 8,192). Raised where traces run long: Qwen3.5-architecture
    #: models hit 8,192 on 19% of dev replies, so they get 16K.
    thinking_budget: int | None = None
    #: The HF reference backend's environment, when the runner's transformers
    #: is known to compute this model wrongly: a runner/<hf_env>/ uv project.
    hf_env: str | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def local_dir(self) -> str:
        return str(MODELS_ROOT / self.hf_id.split("/")[-1])

    @property
    def thinking_modes(self) -> tuple[str, ...]:
        return ("off",) if self.thinking == "none" else ("off", "on")

    @cached_property
    def processor(self):
        from pathlib import Path

        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(self.local_dir, trust_remote_code=self.trust_remote_code)
        # Once vLLM has loaded a remote-code processor (Molmo2), later loads in
        # the same process come back without the repo's chat template. It is
        # the same file either way: read it directly.
        template = Path(self.local_dir) / "chat_template.jinja"
        if getattr(proc, "chat_template", None) is None and template.exists():
            proc.chat_template = template.read_text()
        # Processor attributes fixed at construction (an experiment's token
        # layout); vLLM gets the same through mm_processor_kwargs.
        for k, v in self.extra.get("processor_attrs", {}).items():
            setattr(proc, k, v)
        return proc

    def template_kwargs(self, thinking: str) -> dict:
        """Chat-template kwargs for a thinking mode; recorded in the run config."""
        if self.thinking == "template":
            return {self.thinking_kwarg: thinking == "on"}
        return {}

    def prompt(self, content: list[dict], thinking: str = "off") -> str:
        if thinking == "on" and self.thinking == "prompt":
            content = content + [{"type": "text", "text": self.thinking_instruction}]
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            **self.template_kwargs(thinking))

    def split_reasoning(self, text: str) -> tuple[str | None, str]:
        """(reasoning trace, answer), or (None, text) when the reply has no
        closing tag; backends.reply_fields decides what that means (a trace
        cut off by the cap, or an answer given without closing the trace)."""
        if THINK_CLOSE in text:
            head, _, tail = text.rpartition(THINK_CLOSE)
            return head.replace(THINK_OPEN, "").strip(), tail.strip()
        return None, text

    def extract_answer(self, text: str) -> str:
        """The answer the model itself delimits. Default: the text as it is.
        Adapters strip model-specific structural markup around the answer
        (GLM's answer box); they never parse free text — that is the
        official extractors' job, in the scorer."""
        return text

    def leaked_trace(self, text: str) -> bool:
        """With thinking off: does the reply carry a reasoning trace anyway?"""
        return THINK_OPEN in text or THINK_CLOSE in text

    def vllm_engine_kwargs(self, mode: str, n_frames: int, cap: int) -> dict:
        raise NotImplementedError

    def vllm_mm_data(self, frames: list[Frame], mode: str, video: dict) -> dict | None:
        raise NotImplementedError

    def hf_model_class(self):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText

    def hf_processor_inputs(self, prompt: str, frames: list[Frame], mode: str, video: dict,
                            n_frames: int, cap: int) -> dict:
        raise NotImplementedError

    def hf_generate_inputs(self, inputs) -> dict:
        """The processor outputs the model's generate() takes (default: all)."""
        return dict(inputs)

    def llamacpp_placeholder(self) -> str:
        """The rendered chat template's text for one image, which the llama.cpp
        backend replaces with the server's media marker: the model's own
        template then renders the prompt on every backend. llama.cpp's mtmd
        adds the model's image delimiters itself."""
        raise NotImplementedError(f"{self.key}: no llama.cpp image placeholder yet")

    def llamacpp_media(self, marker: str) -> str:
        """What replaces one placeholder: the server's media marker, wrapped
        in the model's image delimiters where mtmd does not add them."""
        return marker

    def llamacpp_image(self, frame: Frame, cap: int):
        """A frame as llama.cpp gets it. Default: as it is. Adapters whose
        processor resizes frames resize them here to the same size, since
        mtmd rounds sizes its own way."""
        return frame.image()

    def _grid_sized(self, frame: Frame, processor_kwargs: dict):
        """The frame at the size the processor's patch grid covers
        (image_grid_thw x patch size), for processors that report one."""
        from PIL import Image
        img = frame.image()
        key = (img.size, json.dumps(processor_kwargs, sort_keys=True))
        sizes = self.__dict__.setdefault("_llamacpp_sizes", {})
        if key not in sizes:
            ip = self.processor.image_processor
            grid = ip(images=[img], **processor_kwargs)["image_grid_thw"][0]
            sizes[key] = (int(grid[2]) * ip.patch_size, int(grid[1]) * ip.patch_size)
        return img if img.size == sizes[key] else img.resize(sizes[key], Image.BICUBIC)


class ImageListAdapter(Adapter):
    """A model whose processor takes the frames as a list of images and needs
    no per-mode pixel settings: the default for the families that follow."""

    def vllm_engine_kwargs(self, mode: str, n_frames: int, cap: int) -> dict:
        kw = {"limit_mm_per_prompt": {"image": n_frames if mode == "image_list" else 0,
                                      **self.extra.get("mm_limits", {})}}
        if self.trust_remote_code:
            kw["trust_remote_code"] = True
        mm_kwargs = self.extra.get("mm_processor_kwargs", {}) | self.extra.get("processor_attrs", {})
        if mm_kwargs:
            kw["mm_processor_kwargs"] = mm_kwargs
        return kw | self.extra.get("engine_kwargs", {})

    def vllm_mm_data(self, frames: list[Frame], mode: str, video: dict) -> dict | None:
        if mode == "image_list" and frames:
            return {"image": [f.image() for f in frames]}
        return None

    def hf_processor_inputs(self, prompt, frames, mode, video, n_frames, cap) -> dict:
        kw = {"text": [prompt], "return_tensors": "pt"}
        if mode == "image_list" and frames:
            kw |= {"images": [f.image() for f in frames], **self.extra.get("mm_processor_kwargs", {})}
        return kw


class NemotronOmniAdapter(ImageListAdapter):
    """Nemotron-3-Nano-Omni's Hub code registers only AutoModel and
    AutoModelForCausalLM, and its own generate() takes pixel_values,
    input_ids and attention_mask: the processor's num_patches, num_tokens and
    imgs_sizes would reach the language model's generate() and be rejected
    (its HF reference failed on the class first)."""

    def hf_model_class(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM

    def hf_generate_inputs(self, inputs) -> dict:
        return {k: inputs[k] for k in ("input_ids", "attention_mask", "pixel_values") if k in inputs}

    # llama.cpp: the chat template puts every image first ("<image 1><image>
    # <image 2><image>..."), and the processor turns each "<image>" into
    # "<img>" + tokens + "</img>"; mtmd's nemotron_v2_vl adds no delimiters.
    # Frames go as they are: mtmd resizes to its fixed size, where the
    # processor tiles each 448 x 336 frame to 608 x 480 (285 tokens), so the
    # prompt tokens differ from vLLM's.
    def llamacpp_placeholder(self) -> str:
        return "<image>"

    def llamacpp_media(self, marker: str) -> str:
        return f"<img>{marker}</img>"


class GlmAdapter(ImageListAdapter):
    """GLM-4.6V puts its final answer between <|begin_of_box|> and
    <|end_of_box|> — on every closed-form dev item (bcq 200/200, mcq 100/100,
    temporal_localization 99/99), after a sentence of explanation. The
    official JSON extractor cannot see through the markers, so the boxed
    content is the answer; the raw reply is kept beside it."""

    BOX = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)

    def extract_answer(self, text: str) -> str:
        boxes = self.BOX.findall(text)
        return boxes[-1].strip() if boxes else text

    # llama.cpp: mtmd adds <|begin_of_image|> and <|end_of_image|> itself.
    # The processor keeps 448 x 336 as it is (a 24 x 32 grid of 14-pixel
    # patches), which is also aligned to mtmd's 28.
    def llamacpp_placeholder(self) -> str:
        return "<|begin_of_image|><|image|><|end_of_image|>"

    def llamacpp_image(self, frame: Frame, cap: int):
        return self._grid_sized(frame, self.extra.get("mm_processor_kwargs", {}))


class Lfm2VlAdapter(ImageListAdapter):
    """LFM2.5-VL. llama.cpp: the template's "<image>" becomes
    <|image_start|> ... <|image_end|>, which mtmd adds itself. A 448 x 336
    frame stays one tile, resized by the processor to its patch grid
    (spatial_shapes, 16-pixel patches: 448 x 320, 140 tokens); frames go to
    llama.cpp at that size."""

    def llamacpp_placeholder(self) -> str:
        return "<image>"

    def llamacpp_image(self, frame: Frame, cap: int):
        from PIL import Image
        img = frame.image()
        sizes = self.__dict__.setdefault("_llamacpp_sizes", {})
        if img.size not in sizes:
            ip = self.processor.image_processor
            h, w = (int(x) for x in ip(images=[img])["spatial_shapes"][0])
            sizes[img.size] = (w * ip.encoder_patch_size, h * ip.encoder_patch_size)
        size = sizes[img.size]
        return img if img.size == size else img.resize(size, Image.BICUBIC)


class Qwen3VLAdapter(Adapter):
    """Qwen3-VL. Pixel limits differ by mode: for images `max_pixels` is per
    frame, for video it is a budget for the whole clip. Frames arrive already
    capped, so both limits are set only to stop the processor from shrinking
    them; the video path's per-frame capping is pinned off because its
    default is due to change in transformers 5.22.

    llama.cpp: the processor rounds each side to the nearest multiple of 32
    (448 x 336 -> 448 x 320), mtmd rounds up (-> 448 x 352, 14 more tokens a
    frame). Frames go to llama.cpp at the processor's size, which gives the
    same prompt tokens as vLLM (2,502 on the first dev video)."""

    def llamacpp_placeholder(self) -> str:
        return "<|vision_start|><|image_pad|><|vision_end|>"

    def llamacpp_image(self, frame: Frame, cap: int):
        return self._grid_sized(frame, self._processor_kwargs("image_list", 1, cap))

    def _processor_kwargs(self, mode: str, n_frames: int, cap: int) -> dict:
        if mode == "native_video":
            return {"max_pixels": cap * cap * n_frames, "cap_pixels_per_frame": False}
        return {"max_pixels": cap * cap}

    def vllm_engine_kwargs(self, mode: str, n_frames: int, cap: int) -> dict:
        return {
            "limit_mm_per_prompt": {"image": n_frames if mode == "image_list" else 0,
                                    "video": 1 if mode == "native_video" else 0},
            "mm_processor_kwargs": self._processor_kwargs(mode, n_frames, cap),
        }

    @staticmethod
    def _video(frames: list[Frame], video: dict) -> tuple[np.ndarray, dict]:
        arr = np.stack([np.asarray(f.image()) for f in frames])
        meta = {"fps": video["fps"], "frames_indices": [f.frame_index for f in frames],
                "total_num_frames": video["total_num_frames"], "duration": video["duration_s"],
                "do_sample_frames": False}
        return arr, meta

    def vllm_mm_data(self, frames: list[Frame], mode: str, video: dict) -> dict | None:
        if mode == "image_list":
            return {"image": [f.image() for f in frames]}
        if mode == "native_video":
            return {"video": self._video(frames, video)}
        return None

    def hf_processor_inputs(self, prompt, frames, mode, video, n_frames, cap) -> dict:
        kw = {"text": [prompt], "return_tensors": "pt"}
        if mode == "image_list" and frames:
            kw |= {"images": [f.image() for f in frames], **self._processor_kwargs(mode, n_frames, cap)}
        elif mode == "native_video":
            arr, meta = self._video(frames, video)
            kw |= {"videos": [arr], "video_metadata": [meta], "do_sample_frames": False,
                   **self._processor_kwargs(mode, n_frames, cap)}
        return kw


#: Cosmos3-Reasoner prompt guide (github.com/nvidia/cosmos,
#: cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md): sampling without
#: and with reasoning, and the instruction that turns reasoning on.
COSMOS3_SAMPLING = {
    "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
            "repetition_penalty": 1.0},
    "on": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0,
           "repetition_penalty": 1.0},
}
COSMOS3_THINK_INSTRUCTION = (
    "Answer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\n"
    "Write your final answer immediately after the </think> tag.")


MODELS: dict[str, Adapter] = {
    "qwen3-vl-8b": Qwen3VLAdapter(
        key="qwen3-vl-8b",
        hf_id="Qwen/Qwen3-VL-8B-Instruct",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        supported_modes=("image_list", "native_video", "no_frames"),
        notes="Instruct variant: no thinking mode.",
    ),
    # Qwen3.5 uses Qwen3-VL's processor, so the same pixel settings apply.
    # Model card, "general tasks" settings for each mode.
    "qwen3.5-9b": Qwen3VLAdapter(
        key="qwen3.5-9b",
        hf_id="Qwen/Qwen3.5-9B",
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        supported_modes=("image_list", "native_video", "no_frames"),
        thinking="template",
        vendor_sampling={
            "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 1.5, "repetition_penalty": 1.0},
            "on": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                   "presence_penalty": 1.5, "repetition_penalty": 1.0},
        },
        batch_invariant=False,   # Gated DeltaNet layers (GDN_ATTN)
        recurrent=True,
        thinking_budget=16384,
        notes="Thinking defaults to on in its chat template (enable_thinking).",
    ),
    "qwen3.8-27b": Qwen3VLAdapter(
        key="qwen3.8-27b",
        hf_id="Qwen/Qwen3.8-27B",
        revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        supported_modes=("image_list", "native_video", "no_frames"),
        thinking="template",
        vendor_sampling={
            "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 1.5, "repetition_penalty": 1.0},
            "on": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                   "presence_penalty": 0.0, "repetition_penalty": 1.0},
        },
        batch_invariant=False,   # Gated DeltaNet layers, as Qwen3.5-9B
        recurrent=True,
        thinking_budget=16384,
        notes="Same architecture and processor as Qwen3.5-9B. BF16 needs the A100.",
    ),
    # The reasoner tower of Cosmos3-Nano is Qwen3-VL-8B trained further and
    # uses Qwen3-VL's processor. Reasoning is turned on by an instruction in
    # the prompt (Cosmos3-Reasoner prompt guide); off means no instruction.
    "cosmos3-nano": Qwen3VLAdapter(
        key="cosmos3-nano",
        hf_id="nvidia/Cosmos3-Nano",
        revision="e59a53c25979a090fa8706c9acc0c254a6e89b92",
        supported_modes=("image_list", "native_video", "no_frames"),
        thinking="prompt",
        thinking_instruction=COSMOS3_THINK_INSTRUCTION,
        vendor_sampling=COSMOS3_SAMPLING,
        notes="Reasoner tower only; the generator is never loaded. Trained on TAR train.",
    ),
    # GLM-4.6V: patch 14, merge 2, so a 448x336 frame maps to 16x12 = 192
    # tokens with no resize. Model's generation_config sampling (the card
    # gives none); thinking defaults to on (enable_thinking -> /nothink).
    "glm-4.6v-flash": GlmAdapter(
        key="glm-4.6v-flash",
        hf_id="zai-org/GLM-4.6V-Flash",
        revision="411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
        thinking="template",
        vendor_sampling={"off": {"temperature": 0.8, "top_p": 0.6, "top_k": 2},
                         "on": {"temperature": 0.8, "top_p": 0.6, "top_k": 2}},
        notes="Cleared on dev (thinking off: no parse failures, no loops). Thinking on needs "
              "the A100: 20.6 GB of weights leave no KV cache for a 16K context in 24 GB.",
    ),
    # Molmo2 ships its own processing code. Weights are stored FP32; the
    # backends load them in the run's dtype. No sampling settings on the card
    # or in generation_config, so no vendor decoding.
    "molmo2-8b": ImageListAdapter(
        key="molmo2-8b",
        hf_id="allenai/Molmo2-8B",
        revision="e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b",
        trust_remote_code=True,
        # ~537 vision tokens per frame with its token layout: 16 frames are
        # ~8.7K prompt tokens, plus a 1,024-token reply. 16K left no room for
        # the KV cache in 24 GB.
        max_model_len=12288,
        # Video is off: we never send it, and vLLM would otherwise size its
        # memory profile for a maximum-size video (16,351 tokens).
        # Token layout: the one transformers 4.57.1 — the version Molmo2's
        # quickstart names — builds, where these flags come back None: no
        # column tokens, no single-crop start token. transformers 5.17 would
        # apply processor_config.json's values instead (224 more tokens per
        # 16-frame item). With this layout vLLM agrees with the HF reference
        # on 94.8% (0 substantive) instead of 88.0%, and scores a little higher.
        extra={"mm_limits": {"video": 0},
               "processor_attrs": {"image_use_col_tokens": False,
                                   "use_single_crop_col_tokens": False,
                                   "use_single_crop_start_token": False,
                                   "video_use_col_tokens": False,
                                   "use_frame_special_tokens": False}},
        # Its Hub modelling code calls ROPE_INIT_FUNCTIONS["default"], gone in
        # transformers 5: the HF reference runs on 4.57.1, the version it names.
        hf_env="hf-transformers-4.57",
        notes="Each image is tiled into up to 8 crops of 378 px plus a global view. Overflows in "
              "FP16 activations, even unquantized: runs BF16 on the Orin.",
    ),
    # The MoE slot; BF16 needs the A100. Mamba2-Transformer hybrid, so
    # no batch-invariant kernels. Video and audio off: we send neither, and
    # vLLM would otherwise size its memory profile for them. Model card,
    # "Sampling Parameters": instruct mode temperature=0.2, top_k=1; thinking
    # mode temperature=0.6, top_p=0.95.
    "nemotron-3-nano-omni": NemotronOmniAdapter(
        key="nemotron-3-nano-omni",
        hf_id="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        revision="e5e9932441de940c9a62185c870ea5bcd4cd24e2",
        trust_remote_code=True,
        thinking="template",
        vendor_sampling={"off": {"temperature": 0.2, "top_k": 1},
                         "on": {"temperature": 0.6, "top_p": 0.95}},
        batch_invariant=False,
        recurrent=True,          # Mamba-2 layers
        # At 8,192, 22.9% of its thinking-on dev replies hit the cap mid-trace;
        # 16K as for the Qwen3.5 architecture.
        thinking_budget=16384,
        extra={"mm_limits": {"video": 0, "audio": 0}},
        # transformers warns its tokenizer has "an incorrect regex pattern"
        # (fix_mistral_regex=True): a false positive, raised for any local
        # tokenizer whose config.json lacks transformers_version. The shipped
        # tokenizer.json already has the fixed regex; the fix changes none of
        # the dev prompts or answers.
        notes="Upsamples small images to >= 1,024 patches (~300 tokens per frame). Its "
              "tokenizer's regex warning is a false positive.",
    ),
    # Fallback for the MoE slot, only if Nemotron fails its gates.
    # Instruct variant: no thinking mode. Model card: temperature 0.2 for
    # Instruct models. MLA attention; batch invariance untested (A100).
    "kimi-vl-a3b": ImageListAdapter(
        key="kimi-vl-a3b",
        hf_id="moonshotai/Kimi-VL-A3B-Instruct",
        revision="398eede0903cd983a2bfa0cc634e9ac1d843f375",
        trust_remote_code=True,
        vendor_sampling={"off": {"temperature": 0.2}},
        notes="Fallback for the MoE slot. BF16 needs the A100.",
    ),
    "lfm2.5-vl-3b": Lfm2VlAdapter(
        key="lfm2.5-vl-3b",
        hf_id="LiquidAI/LFM2.5-VL-3B",
        revision="35a118d938ce6d123ac2d371649f24a8efb69058",
        # Model card: temperature=0.2, top_k=50, repetition_penalty=1.0.
        vendor_sampling={"off": {"temperature": 0.2, "top_k": 50, "repetition_penalty": 1.0}},
        batch_invariant=False,   # short-convolution layers (SHORT_CONV_ATTN)
        recurrent=True,
        notes="Non-reasoning. A 448-px frame stays below the 512-px tile, so no splitting.",
    ),
    "north-micro": ImageListAdapter(
        key="north-micro",
        hf_id="CohereLabs/North-Micro-Vision-Instruct",
        revision="46b719694e3bad142f3e931774f4622f1024009e",
        # Model card, vLLM section.
        vendor_sampling={"off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                 "presence_penalty": 1.5, "repetition_penalty": 1.0}},
        # transformers 5.17.0's cohere_compass disagrees with vLLM on 71% of
        # dev mcq items; 5.16.0, the card's version, agrees with vLLM.
        hf_env="hf-transformers-5.16",
        notes="Not a reasoning model. Native resolution; min 65,536 px per image.",
    ),
    "cosmos3-edge": ImageListAdapter(
        key="cosmos3-edge",
        hf_id="nvidia/Cosmos3-Edge",
        revision="344d602b128d1bbdacb43b08d0a3626f46343e29",
        thinking="template",
        # The Edge card gives no sampling settings; these are the
        # Cosmos3-Reasoner prompt guide's.
        vendor_sampling=COSMOS3_SAMPLING,
        notes="Thinking defaults to on in its chat template (enable_thinking).",
    ),
}
