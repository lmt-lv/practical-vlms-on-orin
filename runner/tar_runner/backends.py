"""Inference backends: vLLM (the real runner), llama.cpp (GGUF checkpoints,
through llama-server), TensorRT Edge-LLM (its Python runtime) and
HF transformers (the semantic reference vLLM is checked against).

Both take one video's requests at a time and return one result per
request. vLLM reuses the video's encoding across its questions through its
prefix cache; to measure that split, each video first gets a one-token
probe of its first question (≈ the cost of encoding the video), then all of
its questions run together and hit the cache.
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .adapters import Adapter
from .data import Frame


@dataclass
class Request:
    item_id: str
    task: str
    prompt: str
    frames: list[Frame]
    video: dict
    max_tokens: int
    #: The contract's chat content (backends that render the template
    #: themselves, Edge-LLM, rebuild the message from it and check the result
    #: against `prompt`).
    content: list | None = None
    #: The same prompt without its images (tar-bench's --text-probe): a
    #: one-token probe of it, after the video's questions, times the prompt's
    #: text alone, so that the image probe minus it is the images' cost.
    text_prompt: str | None = None
    #: tar-bench's single-stream decode probe: after the questions, the
    #: first question again for 1 token and for DECODE_PROBE_TOKENS tokens
    #: with EOS ignored, its prefix cached; the rate is the extra tokens over
    #: the extra time.
    decode_probe: bool = False


def top2(pairs: list[tuple[str, float]]) -> list[list]:
    """[[token, logprob], [token, logprob]] sorted by logprob."""
    return [[t, round(lp, 4)] for t, lp in sorted(pairs, key=lambda x: -x[1])[:2]]


#: Constrained decoding (a deployment-sweep setting): the answer format
#: enforced where the contract asks for a fixed one; the other tasks stay
#: free text. A reply then can't fail to parse. Thinking off only: a
#: constraint would cover the reasoning trace too.
TIMESTAMP = r"^[0-9]{2}:[0-9]{2}(\.[0-9]{1,2})?$"
CONSTRAINTS = {
    "bcq": {"choice": ["Yes", "No"]},
    "mcq": {"choice": ["A", "B", "C", "D"]},
    "temporal_localization": {"json": {
        "type": "object", "required": ["start", "end"], "additionalProperties": False,
        "properties": {"start": {"type": "string", "pattern": TIMESTAMP},
                       "end": {"type": "string", "pattern": TIMESTAMP}}}},
}


def reply_fields(adapter: Adapter, text: str, thinking: str, hit_cap: bool) -> dict:
    """The scored `prediction` and the trace bookkeeping for one reply.

    Thinking off: the raw reply is the prediction, flagged if it carries a
    reasoning trace anyway. Thinking on: the answer after the trace is the
    prediction; the trace and the raw reply are kept, so a wrong split can be
    redone without re-running (tar_runner.resplit). In both modes the
    adapter's extract_answer strips the model's own answer markup (GLM's
    box), keeping the raw reply when it changes anything. With no closing tag, a
    reply that hit the cap is a cut-off trace (all reasoning, no answer); one
    that stopped on its own is the model answering without closing its trace
    — the text is scored as it stands and flagged `unclosed_trace`."""
    if thinking == "off":
        answer = adapter.extract_answer(text)
        out = {"prediction": answer, "leaked_trace": adapter.leaked_trace(text)}
        return out | ({"raw": text} if answer != text else {})
    reasoning, answer = adapter.split_reasoning(text)
    if reasoning is not None:
        return {"prediction": adapter.extract_answer(answer), "reasoning": reasoning, "raw": text}
    if hit_cap:
        return {"prediction": "", "reasoning": text, "raw": text}
    return {"prediction": adapter.extract_answer(text), "reasoning": None, "raw": text,
            "unclosed_trace": True}


class VLLMBackend:
    name = "vllm"

    def __init__(self, adapter: Adapter, mode: str, n_frames: int, cap: int, *,
                 dtype: str, thinking: str, decoding: str,
                 enforce_eager: bool, gpu_memory_utilization: float, max_model_len: int,
                 cudagraphs: bool = False, model_dir: str | None = None,
                 constrained: bool = False, speculative: dict | None = None):
        import vllm
        from vllm import LLM
        self.adapter, self.mode, self.n_frames, self.cap = adapter, mode, n_frames, cap
        self.thinking = thinking
        self.constrained = constrained
        self.sampling = sampling_settings(adapter, thinking, decoding)
        self.version = vllm.__version__
        self.engine_kwargs = {
            "dtype": dtype, "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enable_prefix_caching": True, "enforce_eager": enforce_eager and not cudagraphs,
            "seed": 0, **adapter.vllm_engine_kwargs(mode, n_frames, cap),
        }
        # TAR_VLLM_MM_CACHE_GB: the multimodal processor cache, held once in
        # the front end and once in the engine (vLLM's default, 4 GiB each).
        # On a 32 GB Orin it grew a Nemotron run past the board's memory.
        # It only saves CPU preprocessing, so outputs are unchanged and
        # it is not part of a run's configuration.
        if os.environ.get("TAR_VLLM_MM_CACHE_GB"):
            self.engine_kwargs["mm_processor_cache_gb"] = float(os.environ["TAR_VLLM_MM_CACHE_GB"])
        # TAR_VLLM_KV_CACHE_GB: a fixed KV cache instead of one derived from
        # the memory share. On the Orin, vLLM's startup accounting can count
        # the page cache filled by loading the weights as consumed: Nemotron
        # (20.5 GB of weights) got a negative KV budget on a fresh boot.
        # It sizes only the KV cache: not part of a run's configuration.
        if os.environ.get("TAR_VLLM_KV_CACHE_GB"):
            self.engine_kwargs["kv_cache_memory_bytes"] = int(float(os.environ["TAR_VLLM_KV_CACHE_GB"]) * 2**30)
        if speculative:   # MTP: the model's own multi-token-prediction head
            self.engine_kwargs["speculative_config"] = dict(speculative)
        if cudagraphs:
            # CUDA graphs for decode only, without torch.compile: the compile
            # step is what stalled on this GPU; graphs cut per-token launch
            # overhead, which dominates small-batch decode in eager mode.
            self.engine_kwargs["compilation_config"] = {
                "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "max_cudagraph_capture_size": 32}
        # model_dir: another checkpoint of the same model (tar_runner.weights),
        # with the adapter's own input side linked in.
        self.llm = LLM(model=model_dir or adapter.local_dir, **self.engine_kwargs)

    def _inputs(self, r: Request) -> dict:
        mm = self.adapter.vllm_mm_data(r.frames, self.mode, r.video)
        return {"prompt": r.prompt, **({"multi_modal_data": mm} if mm else {})}

    def _params(self, r: Request):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        extra = {}
        if self.constrained and r.task in CONSTRAINTS:
            extra["structured_outputs"] = StructuredOutputsParams(**CONSTRAINTS[r.task])
        return SamplingParams(**self.sampling, max_tokens=r.max_tokens, logprobs=2, seed=0, **extra)

    def run_video(self, requests: list[Request]) -> tuple[list[dict], dict]:
        from vllm import SamplingParams
        inputs = [self._inputs(r) for r in requests]

        t0 = time.perf_counter()
        self.llm.generate([inputs[0]], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
        prefill_s = time.perf_counter() - t0

        params = [self._params(r) for r in requests]
        t1 = time.perf_counter()
        outs = self.llm.generate(inputs, params, use_tqdm=False)
        questions_s = time.perf_counter() - t1
        t_answered = time.time()   # tar-bench: the probes after this are not the video's time

        # The decode probe (tar-bench), single stream, the video's prefix cached.
        decode = None
        if requests[0].decode_probe:
            t = time.perf_counter()
            self.llm.generate([inputs[0]], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
            t1 = time.perf_counter() - t
            t = time.perf_counter()
            out = self.llm.generate([inputs[0]], SamplingParams(
                temperature=0.0, max_tokens=DECODE_PROBE_TOKENS, ignore_eos=True), use_tqdm=False)
            decode = decode_rate(t1, time.perf_counter() - t, len(out[0].outputs[0].token_ids))

        # After the questions, so that it cannot touch this video's cache.
        text_s = text_tokens = None
        if requests[0].text_prompt is not None:
            t = time.perf_counter()
            tout = self.llm.generate([{"prompt": requests[0].text_prompt}],
                                     SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
            text_s = time.perf_counter() - t
            text_tokens = len(tout[0].prompt_token_ids)

        results = [self._result(r, out) for r, out in zip(requests, outs, strict=True)]
        timing = {"prefill_probe_s": round(prefill_s, 4), "questions_s": round(questions_s, 4),
                  "n_questions": len(requests), "t_answered": round(t_answered, 3),
                  "prompt_tokens_first": results[0]["prompt_tokens"],
                  "cached_tokens": sum(x["cached_tokens"] for x in results),
                  "output_tokens": sum(x["output_tokens"] for x in results)}
        if text_s is not None:
            timing |= {"text_probe_s": round(text_s, 4), "text_probe_tokens": text_tokens}
        if decode is not None:
            timing["decode_probe"] = decode
        return results, timing

    def run_videos(self, videos: list[list[Request]]) -> tuple[list[list[dict]], dict]:
        """Several videos' questions in one generate call, so continuous
        batching keeps the GPU busy while long replies finish (thinking mode).
        No per-video prefill probe: timing is per batch, for quality runs."""
        reqs = [r for v in videos for r in v]
        inputs = [self._inputs(r) for r in reqs]
        params = [self._params(r) for r in reqs]
        t1 = time.perf_counter()
        outs = self.llm.generate(inputs, params, use_tqdm=False)
        batch_s = time.perf_counter() - t1
        flat = [self._result(r, out) for r, out in zip(reqs, outs, strict=True)]
        results, i = [], 0
        for v in videos:
            results.append(flat[i:i + len(v)])
            i += len(v)
        return results, {"batch_s": round(batch_s, 4), "batch_videos": len(videos),
                         "batch_questions": len(reqs)}

    def _result(self, r: Request, out) -> dict:
        gen = out.outputs[0]
        first = gen.logprobs[0] if gen.logprobs else {}
        return {
            "item_id": r.item_id,
            **reply_fields(self.adapter, gen.text, self.thinking, gen.finish_reason == "length"),
            "finish_reason": gen.finish_reason,
            "hit_cap": gen.finish_reason == "length",
            "prompt_tokens": len(out.prompt_token_ids),
            "cached_tokens": out.num_cached_tokens or 0,
            "output_tokens": len(gen.token_ids),
            "first_token_top2": top2([(lp.decoded_token, lp.logprob) for lp in first.values()]),
        }

    def shutdown(self) -> None:
        self.llm.llm_engine.engine_core.shutdown()


class HFBackend:
    """Batch size 1, no caching: slow and simple, which is the point."""
    name = "hf"

    def __init__(self, adapter: Adapter, mode: str, n_frames: int, cap: int, *,
                 dtype: str, thinking: str, decoding: str, **_):
        import torch
        import transformers
        if decoding != "greedy":
            raise SystemExit("the HF reference backend decodes greedily only")
        self.torch = torch
        self.adapter, self.mode, self.n_frames, self.cap = adapter, mode, n_frames, cap
        self.thinking = thinking
        self.version = transformers.__version__
        self.engine_kwargs = {"dtype": dtype, "device_map": "cuda", "batch_size": 1}
        self.model = adapter.hf_model_class().from_pretrained(
            adapter.local_dir, dtype=getattr(torch, dtype), device_map="cuda",
            trust_remote_code=adapter.trust_remote_code).eval()
        self.tokenizer = adapter.processor.tokenizer

    def run_video(self, requests: list[Request]) -> tuple[list[dict], dict]:
        torch = self.torch
        results = []
        t0 = time.perf_counter()
        for r in requests:
            kw = self.adapter.hf_processor_inputs(r.prompt, r.frames, self.mode, r.video,
                                                  self.n_frames, self.cap)
            inputs = self.adapter.processor(**kw).to("cuda")
            with torch.inference_mode():
                out = self.model.generate(**self.adapter.hf_generate_inputs(inputs),
                                          max_new_tokens=r.max_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          output_scores=True, return_dict_in_generate=True)
            prompt_len = inputs["input_ids"].shape[1]
            # Models whose generate() builds on inputs_embeds (Nemotron's Hub
            # code) may return only the new tokens. A reply is capped far below
            # any prompt's length, so the two cases cannot be confused.
            seq = out.sequences[0]
            new = (seq[prompt_len:] if seq.shape[0] > prompt_len else seq).cpu()
            eos_id = self.model.generation_config.eos_token_id
            eos = {eos_id} if isinstance(eos_id, int) else set(eos_id or [])
            hit_cap = len(new) >= r.max_tokens and (int(new[-1]) not in eos)
            logp = torch.log_softmax(out.scores[0][0].float(), dim=-1)
            vals, idx = torch.topk(logp, 2)
            results.append({
                "item_id": r.item_id,
                **reply_fields(self.adapter, self.tokenizer.decode(new, skip_special_tokens=True),
                               self.thinking, hit_cap),
                "finish_reason": "length" if hit_cap else "stop", "hit_cap": hit_cap,
                "prompt_tokens": prompt_len, "cached_tokens": 0, "output_tokens": len(new),
                "first_token_top2": top2([(self.tokenizer.decode([int(i)]), float(v))
                                          for v, i in zip(vals, idx, strict=True)]),
            })
            del inputs, out
            torch.cuda.empty_cache()   # long prompts fragment memory over a run (Molmo2)
        timing = {"questions_s": round(time.perf_counter() - t0, 4), "n_questions": len(requests)}
        return results, timing

    def shutdown(self) -> None:
        pass


#: Tokens of tar-bench's single-stream decode probe.
DECODE_PROBE_TOKENS = 256


def decode_rate(t1: float, tn: float, n: int) -> dict:
    """The decode probe's record: n tokens in tn s against 1 token in t1 s."""
    rate = (n - 1) / (tn - t1) if n > 1 and tn > t1 else None
    return {"tokens": n, "t1_s": round(t1, 4), "tn_s": round(tn, 4),
            "tok_s": round(rate, 2) if rate else None}


#: llama-server's samplers set to vLLM's neutral defaults: its own defaults
#: (top_k 40, top_p 0.95, min_p 0.05, and repeat_penalty in older builds)
#: would change greedy and vendor-sampled replies alike.
LLAMACPP_NEUTRAL = {"top_k": 0, "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0,
                    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
                    "dry_multiplier": 0.0, "xtc_probability": 0.0}
#: vLLM SamplingParams names -> llama-server's.
LLAMACPP_NAMES = {"repetition_penalty": "repeat_penalty"}


def llamacpp_server_bin() -> Path:
    """TAR_LLAMA_SERVER, else the one build under build/llama.cpp/
    (runner/scripts/build_llama_cpp.sh)."""
    if os.environ.get("TAR_LLAMA_SERVER"):
        return Path(os.environ["TAR_LLAMA_SERVER"])
    root = Path(__file__).resolve().parents[2] / "build" / "llama.cpp"
    found = sorted(glob.glob(str(root / "build-*" / "bin" / "llama-server")))
    if len(found) != 1:
        raise SystemExit(f"llama-server: found {found or 'none'} under {root}; set TAR_LLAMA_SERVER")
    return Path(found[0])


def llamacpp_version(binary: Path) -> dict:
    """{"build": ..., "commit": ...} from `llama-server --version`."""
    out = subprocess.run([str(binary), "--version"], capture_output=True, text=True)
    line = next((l for l in (out.stdout + out.stderr).splitlines() if "commit" in l), "")
    # version: 0.5.0-dev (build 11302, commit 05af0d2b1)
    build = line.split("build", 1)[-1].split(",")[0].strip() if "build" in line else ""
    commit = line.rsplit("commit", 1)[-1].strip(" )") if "commit" in line else ""
    return {"build": build, "commit": commit}


def llamacpp_server_args(max_model_len: int, kv_cache_dtype: str | None = None) -> list[str]:
    """Server settings that can change outputs; part of a run's configuration.
    One slot, so requests run one at a time and each reuses the previous
    prompt's cache (a video's frames, across its questions; the backend
    empties it between videos). No host-RAM
    prompt cache (default 8 GiB): on unified memory it is the GPU's memory.
    kv_cache_dtype (a deployment-sweep setting): the KV cache in that type,
    keys and values; a quantized V cache needs flash attention."""
    args = ["-ngl", "999", "-c", str(max_model_len), "-np", "1", "--seed", "0",
            "--cache-ram", "0", "--no-webui"]
    if kv_cache_dtype:
        args += ["-ctk", kv_cache_dtype, "-ctv", kv_cache_dtype, "-fa", "on"]
    return args


class LlamaCppBackend:
    """llama.cpp's llama-server, one process per run, on a GGUF checkpoint and
    its mmproj (`--weights <tag>`, a llama.cpp entry in checkpoints.json).

    The prompt is the adapter's own, rendered with the model's HF chat
    template as for vLLM, with each image placeholder replaced by the
    server's media marker; frames go at the size the model's processor
    would give them (Adapter.llamacpp_image). Requests go to /completion one
    at a time: the first question of a video is probed with one token (the
    video's encoding), then its questions run and reuse the cached prefix."""
    name = "llamacpp"

    def __init__(self, adapter: Adapter, mode: str, n_frames: int, cap: int, *,
                 thinking: str, decoding: str, max_model_len: int, model_dir: str | None = None,
                 kv_cache_dtype: str | None = None, **_):
        if mode not in ("image_list", "no_frames"):
            raise SystemExit(f"the llama.cpp backend takes image_list and no_frames, not {mode}")
        if model_dir is None:
            raise SystemExit("the llama.cpp backend runs a GGUF checkpoint: pass --weights <tag>")
        self.adapter, self.mode, self.n_frames, self.cap = adapter, mode, n_frames, cap
        self.thinking = thinking
        self.sampling = sampling_settings(adapter, thinking, decoding)
        rec = json.loads((Path(model_dir) / "gguf.json").read_text())
        names = list(rec["files"])
        model = next(n for n in names if "mmproj" not in n.lower())
        mmproj = next((n for n in names if "mmproj" in n.lower()), None)
        self.binary = llamacpp_server_bin()
        v = llamacpp_version(self.binary)
        self.version = f"{v['build']} ({v['commit']})"
        with socket.socket() as sk:
            sk.bind(("127.0.0.1", 0))
            self.port = sk.getsockname()[1]
        args = llamacpp_server_args(max_model_len, kv_cache_dtype)
        self.engine_kwargs = {"binary": str(self.binary), **v, "server_args": args,
                              "model": model, "mmproj": mmproj}
        # --slot-save-path only enables the slot actions (the erase after a
        # text probe); nothing is saved. Not an output setting, so not hashed.
        cmd = [str(self.binary), "-m", str(Path(model_dir) / model), *args,
               "--host", "127.0.0.1", "--port", str(self.port),
               "--slot-save-path", "/tmp/tar-llama-slots"]
        if mmproj:
            cmd += ["--mmproj", str(Path(model_dir) / mmproj)]
        env = {k: val for k, val in os.environ.items() if k != "CUDA_SCALE_LAUNCH_QUEUES"}
        Path("/tmp/tar-llama-slots").mkdir(exist_ok=True)   # /tmp does not survive a reboot
        self.log_path = Path(os.environ.get("TAR_LLAMA_LOG", "/tmp/tar-llama-server.log"))
        self.log = self.log_path.open("w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=env)
        self._wait_ready()
        self.marker = self._get("/props")["media_marker"]
        self.tokenizer = adapter.processor.tokenizer
        # The server adds the model's BOS to a /completion prompt; the chat
        # template's rendering may already start with it (LFM2.5's
        # <|startoftext|>), which would double it (one extra prompt token).
        added = self._post("/tokenize", {"content": "", "add_special": True, "with_pieces": True})
        self.bos = added["tokens"][0]["piece"] if added.get("tokens") else None

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self._url(path), timeout=30) as r:
            return json.load(r)

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self._url(path), json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.load(r)

    def _wait_ready(self, timeout_s: float = 900) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise SystemExit(f"llama-server exited ({self.proc.returncode}); see {self.log_path}")
            try:
                if self._get("/health").get("status") == "ok":
                    return
            except OSError:
                pass
            time.sleep(1)
        raise SystemExit(f"llama-server not ready after {timeout_s} s; see {self.log_path}")

    def _prompt(self, r: Request, images: list[str]) -> dict | str:
        text = r.prompt
        if self.bos and text.startswith(self.bos):
            text = text[len(self.bos):]
        if self.mode == "no_frames" or not images:
            return text
        ph = self.adapter.llamacpp_placeholder()
        if text.count(ph) != len(images):
            raise RuntimeError(f"{r.item_id}: {text.count(ph)} placeholders for {len(images)} frames")
        return {"prompt_string": text.replace(ph, self.adapter.llamacpp_media(self.marker)),
                "multimodal_data": images}

    def _probe_prompt(self, prompts: list):
        """The video's one-token probe. Default: its first question's prompt.
        For a recurrent model (adapter.recurrent), the frames the questions
        share plus exactly n_ubatch + 4
        filler tokens: llama-server then checkpoints the model's state where
        the questions part (it checkpoints n_ubatch + 4 and 4 tokens before a
        prompt's end, never at a batch that starts with an image), and every
        question resumes from there instead of re-encoding the frames."""
        first = prompts[0]
        if not self.adapter.recurrent or not isinstance(first, dict):
            return first
        marker = self.adapter.llamacpp_media(self.marker)
        text = first["prompt_string"]
        cut = text.rfind(marker) + len(marker)   # the end of the last frame
        if any(not (isinstance(q, dict) and q["prompt_string"][:cut] == text[:cut]) for q in prompts):
            return first
        return {"prompt_string": text[:cut] + self._filler(), "multimodal_data": first["multimodal_data"]}

    def _filler(self) -> str:
        if getattr(self, "_filler_text", None) is None:
            n_ub = int(os.environ.get("TAR_LLAMA_UBATCH", "512"))   # llama-server's default -ub
            k, text = n_ub + 4, ""
            for _ in range(8):   # " the" is one token in these tokenizers; check and adjust
                text = " the" * k
                n = len(self._post("/tokenize", {"content": text, "add_special": False})["tokens"])
                if n == n_ub + 4:
                    break
                k += n_ub + 4 - n
            self._filler_text = text
        return self._filler_text

    def _body(self, prompt, n_predict: int) -> dict:
        sampling = {LLAMACPP_NAMES.get(k, k): v for k, v in self.sampling.items()}
        return {"prompt": prompt, "n_predict": n_predict, **LLAMACPP_NEUTRAL, **sampling,
                "seed": 0, "n_probs": 2, "cache_prompt": True, "return_tokens": True}

    def _text(self, out: dict) -> str:
        """The reply, decoded from its token ids with the model's own HF
        tokenizer as vLLM decodes (skip_special_tokens): llama-server's
        `content` drops every control token, GLM's answer box included."""
        return self.tokenizer.decode(out["tokens"], skip_special_tokens=True)

    def run_video(self, requests: list[Request]) -> tuple[list[dict], dict]:
        images = []
        if self.mode == "image_list":
            for f in requests[0].frames:
                buf = io.BytesIO()
                self.adapter.llamacpp_image(f, self.cap).save(buf, format="PNG")
                images.append(base64.b64encode(buf.getvalue()).decode())
        prompts = [self._prompt(r, images) for r in requests]

        # Each video starts from an empty cache. llama.cpp's numerics depend
        # on how a prefix was batched: a video that reused the previous
        # video's cached header replied differently (6 of 12 items) from the
        # same video computed fresh, so a run would depend on its video order
        # and on where it resumed. Within the video, its questions reuse its
        # frames' cache as before.
        self._post("/slots/0?action=erase", {})
        t0 = time.perf_counter()
        probe = self._post("/completion", self._body(self._probe_prompt(prompts), 1))
        prefill_s = time.perf_counter() - t0

        results, server = [], {"prompt_ms": 0.0, "predicted_ms": 0.0}
        t1 = time.perf_counter()
        for r, prompt in zip(requests, prompts, strict=True):
            out = self._post("/completion", self._body(prompt, r.max_tokens))
            hit_cap = out.get("stop_type") == "limit"
            probs = out.get("completion_probabilities") or [{}]
            server["prompt_ms"] += out["timings"]["prompt_ms"]
            server["predicted_ms"] += out["timings"]["predicted_ms"]
            results.append({
                "item_id": r.item_id,
                **reply_fields(self.adapter, self._text(out), self.thinking, hit_cap),
                "finish_reason": "length" if hit_cap else "stop",
                "hit_cap": hit_cap,
                "prompt_tokens": out["tokens_evaluated"],
                # tokens_cached counts the slot's whole cache, earlier replies
                # included; the prompt's cached part is what it did not evaluate.
                "cached_tokens": out["tokens_evaluated"] - out["timings"]["prompt_n"],
                "output_tokens": out["tokens_predicted"],
                "first_token_top2": top2([(t["token"], t["logprob"])
                                          for t in probs[0].get("top_logprobs", [])]),
            })
        questions_s = time.perf_counter() - t1
        t_answered = time.time()   # tar-bench: the probes after this are not the video's time

        # The decode probe (tar-bench), after the replies: llama-server's own
        # per-request rate is kept beside the probe's. EOS ignored.
        decode = None
        if requests[0].decode_probe:
            t = time.perf_counter()
            self._post("/completion", self._body(prompts[0], 1))
            t1 = time.perf_counter() - t
            t = time.perf_counter()
            out = self._post("/completion", self._body(prompts[0], DECODE_PROBE_TOKENS) | {"ignore_eos": True})
            decode = decode_rate(t1, time.perf_counter() - t, out["tokens_predicted"])
            decode["server_tok_s"] = round(out["timings"]["predicted_per_second"], 2)

        # The text-only probe (tar-bench) runs last, when the video's replies
        # are done; the next video starts from an empty cache anyway. (A
        # probe run first changed 12 of 12 replies.)
        text = None
        if requests[0].text_prompt is not None:
            t = time.perf_counter()
            tp = requests[0].text_prompt
            if self.bos and tp.startswith(self.bos):
                tp = tp[len(self.bos):]
            text = self._post("/completion", self._body(tp, 1))
            text_s = time.perf_counter() - t
        timing = {"prefill_probe_s": round(prefill_s, 4), "questions_s": round(questions_s, 4),
                  "n_questions": len(requests), "t_answered": round(t_answered, 3),
                  "prompt_tokens_first": results[0]["prompt_tokens"],
                  "cached_tokens": sum(x["cached_tokens"] for x in results),
                  "output_tokens": sum(x["output_tokens"] for x in results),
                  "server": {"probe_prompt_ms": round(probe["timings"]["prompt_ms"], 2),
                             "prompt_ms": round(server["prompt_ms"], 2),
                             "predicted_ms": round(server["predicted_ms"], 2)}}
        if text is not None:
            timing["text_probe_s"] = round(text_s, 4)
            timing["text_probe_tokens"] = text["tokens_evaluated"]
            timing["server"]["text_probe_prompt_ms"] = round(text["timings"]["prompt_ms"], 2)
        if decode is not None:
            timing["decode_probe"] = decode
        return results, timing

    def shutdown(self) -> None:
        # SIGKILL, not SIGTERM: on SIGTERM llama-server once hung in its
        # cleanup ("cleaning up before exit...", waiting on a futex) and
        # outlived the run holding 9 GB. It keeps nothing we need.
        self.proc.kill()
        self.proc.wait(timeout=60)
        self.log.close()


class EdgeLLMBackend:
    """TensorRT Edge-LLM's Python runtime (the published wheel's native
    payload), on a ModelOpt checkpoint and the engines built for it
    (`--weights <tag>`, an edge-llm entry in checkpoints.json; engines by
    runner/scripts/edgellm_engines.sh). Runs in runner/edgellm/'s
    environment (JetPack's TensorRT), with PYTHONPATH pointing at runner/.

    Edge-LLM renders the chat template itself, from the contract's content
    parts; every request checks the rendering against the adapter's prompt
    and fails on a difference. Frames go at the size the model's processor
    gives them (Adapter.llamacpp_image), with Edge-LLM's resize off. One
    request at a time: a video's first question is probed with one token,
    with the context cache bypassed (each video starts fresh, as with
    llama.cpp); its questions then reuse that prefix from the cache. Replies
    are decoded from token ids with the HF tokenizer, as vLLM decodes.

    With engines built for a batch (edgellm.json's --maxBatchSize > 1, the
    batched measurement), a video's questions go as one request instead, up
    to that many at a time, under one cap (the contract's caps are equal with
    thinking off; the largest otherwise, each reply cut back to its own)."""
    name = "edgellm"

    def __init__(self, adapter: Adapter, mode: str, n_frames: int, cap: int, *,
                 thinking: str, decoding: str, model_dir: str | None = None, **_):
        import importlib.metadata
        import tempfile
        from tensorrt_edgellm import runtime
        if mode not in ("image_list", "no_frames"):
            raise SystemExit(f"the Edge-LLM backend takes image_list and no_frames, not {mode}")
        if model_dir is None:
            raise SystemExit("the Edge-LLM backend runs an Edge-LLM checkpoint: pass --weights <tag>")
        self.adapter, self.mode, self.n_frames, self.cap = adapter, mode, n_frames, cap
        self.thinking = thinking
        self.sampling = sampling_settings(adapter, thinking, decoding)
        extra = {k: v for k, v in self.sampling.items()
                 if k not in ("temperature", "top_p", "top_k")
                 and v not in (0.0, 1.0)}
        if extra:
            raise SystemExit(f"Edge-LLM has no {sorted(extra)} sampling setting")
        self.m = m = runtime.load()
        m.set_profiling_enabled(True)   # prefill metrics: the reused (cached) tokens
        # TAR_EDGELLM_ENGINES: engines built in another power mode. TensorRT
        # plans are per device, and MODE_30W's power-gated GPU (8 of 16 SMs)
        # is another device: MAXN engines fail there.
        engines = Path(os.environ.get("TAR_EDGELLM_ENGINES") or f"{model_dir}.engines")
        rec = json.loads((engines / "edgellm.json").read_text())
        cc = m.ContextCacheConfig()
        cc.enabled = True
        cc.max_records = 64
        self.rt = m.LLMRuntime(engine_dir=str(engines / "llm"),
                               multimodal_engine_dir=str(engines / "visual") if mode == "image_list" else "",
                               checkpoint_dir=str(model_dir), context_cache_config=cc)
        self.version = importlib.metadata.version("tensorrt-edgellm")
        args = rec.get("llm_build_args", [])
        self.batch = int(args[args.index("--maxBatchSize") + 1]) if "--maxBatchSize" in args else 1
        self.engine_kwargs = {"engines": str(engines), "context_cache": "per video",
                              **{k: v for k, v in rec.items() if k != "tag"}}
        self.tokenizer = adapter.processor.tokenizer
        self.tmp = tempfile.TemporaryDirectory(prefix="tar-edgellm-")

    def _request(self, r: Request, images: list, n: int, fresh: bool, text_only: bool = False):
        m = self.m
        content = r.content
        if text_only:
            content = [c for c in content if c["type"] != "image"]
        if self.thinking == "on" and self.adapter.thinking == "prompt":
            content = content + [{"type": "text", "text": self.adapter.thinking_instruction}]
        msg = m.Message("user", [m.MessageContent(c["type"], c.get("text", "")) for c in content])
        s = self.sampling
        greedy = s.get("temperature", 0.0) == 0.0
        g = m.create_generation_request(
            [[msg]], temperature=s.get("temperature", 0.0), top_p=1.0 if greedy else s.get("top_p", 1.0),
            top_k=1 if greedy else s.get("top_k", 0), max_generate_length=n, num_logprobs=2,
            enable_thinking=bool(self.adapter.template_kwargs(self.thinking).get(self.adapter.thinking_kwarg)))
        g.sampling_seed = 0
        g.context_cache_lookup_policy = (m.ContextCacheLookupPolicy.BYPASS if fresh
                                         else m.ContextCacheLookupPolicy.USE_CACHE)
        g.context_cache_commit_policy = m.ContextCacheCommitPolicy.PREFILL_STATE_ONLY
        if images and not text_only:
            reqs = list(g.requests)
            reqs[0].image_buffers = images
            g.requests = reqs
        return g

    def _batch_request(self, rs: list[Request], images: list, n: int):
        """One request for several questions, each with the video's frames."""
        gs = [self._request(r, images, n, fresh=False) for r in rs]
        g = self._request(rs[0], images, n, fresh=False)
        g.requests = [x.requests[0] for x in gs]
        return g

    def _run(self, g, expected: str | None):
        before = self.rt.get_prefill_metrics().reused_tokens
        t = time.perf_counter()
        resp = self.rt.handle_request(g)
        dt = time.perf_counter() - t
        if expected is not None:
            got = g.formatted_requests[0].formatted_complete_request
            if got != expected:
                raise RuntimeError("Edge-LLM rendered a different prompt from the adapter's: "
                                   f"{got[:200]!r} vs {expected[:200]!r}")
        return resp, dt, self.rt.get_prefill_metrics().reused_tokens - before

    def run_video(self, requests: list[Request]) -> tuple[list[dict], dict]:
        m = self.m
        images = []
        if self.mode == "image_list":
            for i, f in enumerate(requests[0].frames):
                p = Path(self.tmp.name) / f"{i:02d}.png"
                self.adapter.llamacpp_image(f, self.cap).save(p)
                im = m.load_image_from_path(str(p))
                im.do_resize = False
                images.append(im)

        _, prefill_s, _ = self._run(self._request(requests[0], images, 1, fresh=True), requests[0].prompt)

        results = []
        t1 = time.perf_counter()
        if self.batch > 1:
            answered = []
            for i in range(0, len(requests), self.batch):
                rs = requests[i:i + self.batch]
                g = self._batch_request(rs, images, max(r.max_tokens for r in rs))
                resp, _, reused = self._run(g, None)
                for j, r in enumerate(rs):
                    got = g.formatted_requests[j].formatted_complete_request
                    if got != r.prompt:
                        raise RuntimeError("Edge-LLM rendered a different prompt from the adapter's: "
                                           f"{got[:200]!r} vs {r.prompt[:200]!r}")
                    answered.append((r, resp, j, reused if j == 0 else 0))
        else:
            answered = []
            for r in requests:
                resp, _, reused = self._run(self._request(r, images, r.max_tokens, fresh=False), r.prompt)
                answered.append((r, resp, 0, reused))
        for r, resp, j, reused in answered:
            ids = list(resp.output_ids[j])
            hit_cap = resp.finish_reasons[j].name == "LENGTH"
            if len(ids) > r.max_tokens:   # a batch under a larger cap than this reply's
                ids, hit_cap = ids[:r.max_tokens], True
            first = resp.logprobs[j][0] if resp.logprobs and resp.logprobs[j] else []
            results.append({
                "item_id": r.item_id,
                **reply_fields(self.adapter, self.tokenizer.decode(ids, skip_special_tokens=True),
                               self.thinking, hit_cap),
                "finish_reason": "length" if hit_cap else "stop",
                "hit_cap": hit_cap,
                "prompt_tokens": int(resp.prompt_token_counts[j]),
                "cached_tokens": int(reused),
                "output_tokens": len(ids),
                "first_token_top2": top2([(bytes(e.piece).decode("utf-8", "replace"), e.logprob)
                                          for e in first]),
            })
        questions_s = time.perf_counter() - t1
        t_answered = time.time()   # tar-bench: the probes after this are not the video's time
        timing = {"prefill_probe_s": round(prefill_s, 4), "questions_s": round(questions_s, 4),
                  "n_questions": len(requests), "t_answered": round(t_answered, 3),
                  "prompt_tokens_first": results[0]["prompt_tokens"],
                  "cached_tokens": sum(x["cached_tokens"] for x in results),
                  "output_tokens": sum(x["output_tokens"] for x in results)}
        # The decode probe (tar-bench). Edge-LLM has no ignore-EOS setting, so
        # it decodes the question with the video's longest reply, up to
        # DECODE_PROBE_TOKENS, and the rate uses the tokens it produced.
        if requests[0].decode_probe:
            k = max(range(len(results)), key=lambda i: results[i]["output_tokens"])
            _, t1, _ = self._run(self._request(requests[k], images, 1, fresh=False), requests[k].prompt)
            resp, tn, _ = self._run(self._request(requests[k], images, DECODE_PROBE_TOKENS, fresh=False),
                                    requests[k].prompt)
            timing["decode_probe"] = decode_rate(t1, tn, len(resp.output_ids[0]))
        # The text-only probe (tar-bench) runs last; the next video bypasses the cache.
        if requests[0].text_prompt is not None:
            resp, text_s, _ = self._run(self._request(requests[0], [], 1, fresh=True, text_only=True), None)
            timing["text_probe_s"] = round(text_s, 4)
            timing["text_probe_tokens"] = int(resp.prompt_token_counts[0])
        return results, timing

    def shutdown(self) -> None:
        del self.rt
        self.tmp.cleanup()


def sampling_settings(adapter: Adapter, thinking: str, decoding: str) -> dict:
    """SamplingParams kwargs (less max_tokens, logprobs and seed): greedy, or the
    model card's recommended settings for this thinking mode."""
    if decoding == "greedy":
        return {"temperature": 0.0}
    settings = adapter.vendor_sampling.get(thinking)
    if not settings:
        raise SystemExit(f"{adapter.key} records no vendor sampling settings for thinking {thinking}")
    return dict(settings)


BACKENDS = {"vllm": VLLMBackend, "llamacpp": LlamaCppBackend, "edgellm": EdgeLLMBackend,
            "hf": HFBackend}
