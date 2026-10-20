"""Run a model over a slice: adapter × backend × serving config → predictions.

    uv run tar-run --model qwen3-vl-8b --backend vllm --mode image_list \\
        --frames 16 --cap 448 --slice dev --tasks all

Writes, under runs/<task-dir>/<name>-<config hash>/:
  predictions.jsonl   one row per item: item_id, prediction (raw reply),
                      finish reason, token counts, first-token top-2
  videos.jsonl        per-video timing: prefill probe, questions, tokens
  run.json            the manifest: model and revision, backend and version,
                      engine settings, input contract, slice and frame hashes,
                      git commit, GPU

Runs resume: videos already fully present in predictions.jsonl are skipped.
Score with the eval project: `tar-eval-score --pred .../predictions.jsonl`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path

from . import contract, weights
from .adapters import MODELS
from .backends import (
    BACKENDS, Request, llamacpp_server_args, llamacpp_server_bin, llamacpp_version,
    sampling_settings,
)

#: The checkpoint runtime (checkpoints.json) each backend runs with --weights.
WEIGHTS_RUNTIME = {"vllm": "vllm", "llamacpp": "llama.cpp", "edgellm": "edge-llm"}
from .data import (
    RUNS_ROOT, frame_manifest_sha256, frames_for, load_rows, sha256_file, slice_file,
    source_video,
)


def git_state() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def config_of(args) -> dict:
    adapter = MODELS[args.model]
    cfg = {
        "model": adapter.key, "hf_id": adapter.hf_id, "revision": adapter.revision,
        "backend": args.backend, "mode": args.mode,
        "frames": 0 if args.mode == "no_frames" else args.frames,
        "cap": 0 if args.mode == "no_frames" else args.cap,
        "slice": Path(slice_file(args.slice)).stem, "slice_sha256": sha256_file(slice_file(args.slice)),
        "tasks": args.tasks, "prefix": args.prefix,
        "max_tokens": contract.MAX_TOKENS,
        "decoding": "greedy, constrained" if args.constrained else "greedy, unconstrained",
        "contract": {"image_list_header": contract.IMAGE_LIST_HEADER,
                     "no_frames_preamble": contract.NO_FRAMES_PREAMBLE,
                     "style": args.style,
                     "style_line": contract.STYLE_LINE if args.style == "plain" else None},
        "enforce_eager": args.eager and not args.cudagraphs, "cudagraphs": args.cudagraphs,
        "max_model_len": args.max_model_len,
    }
    # Added later. Recorded only when they differ from the settings every
    # earlier run had, so that earlier configurations keep their hashes.
    if args.dtype != "bfloat16":
        cfg["dtype"] = args.dtype
    if args.thinking != "off" or adapter.thinking != "none":
        cfg["thinking"] = args.thinking
        cfg["chat_template_kwargs"] = adapter.template_kwargs(args.thinking)
        if args.thinking == "on":
            cfg["max_tokens"] = adapter.thinking_budget or contract.MAX_TOKENS_THINKING
            if adapter.thinking == "prompt":
                cfg["contract"]["thinking_instruction"] = adapter.thinking_instruction
    if args.decoding != "greedy":
        cfg["sampling"] = {"vendor": sampling_settings(adapter, args.thinking, args.decoding)}
    if args.batch_invariant:
        cfg["batch_invariant"] = True
    if args.weights:
        cfg["weights"] = weights.run_config(args.weights)
    if args.constrained:   # what the constraint enforces, per task
        from .backends import CONSTRAINTS
        cfg["constraints"] = CONSTRAINTS
    if args.speculative:   # MTP
        cfg["speculative"] = args.speculative
    if args.backend == "llamacpp":
        # The build and server settings change outputs as an engine version
        # does; vLLM's version is pinned by the lock, llama.cpp's by a build.
        v = llamacpp_version(llamacpp_server_bin())
        cfg["llamacpp"] = {"commit": v["commit"],
                           "server_args": llamacpp_server_args(args.max_model_len, args.kv_cache_dtype),
                           "cache": "per video", "detokenize": "hf tokenizer, skip_special_tokens"}
        if adapter.recurrent:   # the probe that lets its questions reuse the frames
            cfg["llamacpp"]["cache"] = "per video, prefix probe"
    if args.backend == "edgellm":
        # Version, engine limits and checkpoint hash are in cfg["weights"]
        # (the engines' edgellm.json).
        cfg["edgellm"] = {"cache": "per video", "detokenize": "hf tokenizer, skip_special_tokens",
                          "images": "processor size, do_resize false"}
    # Adapter settings that change the model's inputs (Molmo2's token layout).
    for key in ("processor_attrs", "mm_processor_kwargs"):
        if adapter.extra.get(key):
            cfg.setdefault("adapter", {})[key] = adapter.extra[key]
    if args.backend == "hf" and adapter.hf_env:
        import transformers
        cfg["hf_env"] = {"name": adapter.hf_env, "transformers": transformers.__version__}
    return cfg


def backend_options(args) -> dict:
    """The deployment sweep's settings, passed only to the backend that has them."""
    if args.backend == "vllm":
        return {k: v for k, v in (("constrained", args.constrained), ("speculative", args.speculative)) if v}
    if args.backend == "llamacpp" and args.kv_cache_dtype:
        return {"kv_cache_dtype": args.kv_cache_dtype}
    return {}


def run_name(cfg: dict) -> str:
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    shape = "noframes" if cfg["mode"] == "no_frames" else f"{cfg['frames']}f{cfg['cap']}"
    prefix = f"-p{cfg['prefix']}" if cfg["prefix"] else ""
    style = "-plain" if cfg["contract"]["style"] == "plain" else ""
    think = "-think" if cfg.get("thinking") == "on" else ""
    vendor = "-vendor" if "sampling" in cfg else ""
    fp16 = "-fp16" if cfg.get("dtype") == "float16" else ""
    bi = "-bi" if cfg.get("batch_invariant") else ""
    quant = f"-{cfg['weights']['tag'].split('.', 1)[1]}" if cfg.get("weights") else ""
    # The deployment sweep's settings, in the readable part too (they are already in the hash).
    constrained = "-constrained" if "constraints" in cfg else ""
    mtp = f"-mtp{cfg['speculative']['num_speculative_tokens']}" if "speculative" in cfg else ""
    kv = next((f"-kv{a[i + 1]}" for a in [cfg.get("llamacpp", {}).get("server_args", [])]
               for i, x in enumerate(a) if x == "-ctk"), "")
    return (f"{cfg['model']}-{cfg['backend']}-{cfg['mode']}-{shape}{style}{think}{vendor}{fp16}{bi}"
            f"{constrained}{mtp}{kv}{quant}-{cfg['slice']}{prefix}-{h}")


def requests_for(video_id: str, vrows: list[dict], args, adapter) -> list[Request]:
    """One video's requests: the contract's content and the prompt rendered
    from it, and with --text-probe the first prompt without its images."""
    frames = [] if args.mode == "no_frames" else frames_for(video_id, args.frames, args.cap)
    video = source_video(video_id)
    contents = [contract.user_content(r["question"], frames, args.mode, r["task"], args.style)
                for r in vrows]
    reqs = [Request(r["item_id"], r["task"], adapter.prompt(c, args.thinking),
                    frames, video,
                    contract.max_tokens(r["task"], args.thinking, adapter.thinking_budget),
                    content=c)
            for r, c in zip(vrows, contents, strict=True)]
    if args.text_probe and args.mode == "image_list":
        text_only = [p for p in contents[0] if p["type"] != "image"]
        reqs[0].text_prompt = adapter.prompt(text_only, args.thinking)
    if args.decode_probe:
        reqs[0].decode_probe = True
    return reqs


def run_batched(todo, args, adapter, backend, pf, vf, t_run) -> None:
    """--videos-per-batch N: N videos per generate call. A killed run loses at
    most the batch in progress; resuming skips every video already written."""
    n_done = 0
    for i in range(0, len(todo), args.videos_per_batch):
        chunk = todo[i:i + args.videos_per_batch]
        t_batch = time.time()
        results, timing = backend.run_videos([requests_for(v, vr, args, adapter) for v, vr in chunk])
        timing |= {"t_start": round(t_batch, 3), "t_end": round(time.time(), 3)}
        for (video_id, vrows), vres in zip(chunk, results, strict=True):
            for r, res in zip(vrows, vres, strict=True):
                pf.write(json.dumps(res | {"task": r["task"], "video_id": video_id},
                                    ensure_ascii=False) + "\n")
            vf.write(json.dumps({"video_id": video_id, **timing}) + "\n")
        pf.flush()
        vf.flush()
        n_done += len(chunk)
        el = time.perf_counter() - t_run
        print(f"  {n_done}/{len(todo)} videos  {el / n_done:.2f} s/video  "
              f"eta {el / n_done * (len(todo) - n_done) / 60:.1f} min", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--backend", default="vllm", choices=sorted(BACKENDS))
    ap.add_argument("--mode", default="image_list", choices=contract.INPUT_MODES)
    ap.add_argument("--frames", type=int, default=16, choices=(4, 8, 16, 32))
    ap.add_argument("--cap", type=int, default=448, choices=(448, 896))
    ap.add_argument("--slice", default="dev")
    ap.add_argument("--tasks", default="all", help="all | shortlist | comma-separated list")
    ap.add_argument("--prefix", type=int, help="first N videos by priority")
    ap.add_argument("--limit-videos", type=int, help="only the first N videos by priority (a sample, or a smoke test)")
    ap.add_argument("--style", default="plain", choices=contract.STYLES,
                    help="plain (default): one fixed answer-style line on the generative tasks")
    ap.add_argument("--task-dir", default="adhoc", help="runs/<task-dir>/")
    ap.add_argument("--eager", action=argparse.BooleanOptionalAction, default=True,
                    help="vLLM eager mode (default on: first-run compilation stalls on this GPU)")
    ap.add_argument("--cudagraphs", action="store_true",
                    help="vLLM: CUDA graphs for decode, without torch.compile")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int,
                    help="default: the adapter's (8192 for most), plus the thinking budget "
                         "with thinking on")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"),
                    help="activation dtype (float16 for the A100/Orin comparison)")
    ap.add_argument("--thinking", default="off", choices=("off", "on"))
    ap.add_argument("--decoding", choices=("greedy", "vendor"),
                    help="greedy, or vendor: the model card's recommended sampling (seeded). "
                         "Default: greedy with thinking off, vendor with thinking on, since "
                         "greedy decoding loops in thinking mode")
    ap.add_argument("--print-run-dir", action="store_true",
                    help="print the run directory this configuration maps to, and exit")
    ap.add_argument("--batch-invariant", action="store_true",
                    help="vLLM batch-invariant kernels: bit-reproducible runs, ~30%% slower")
    ap.add_argument("--text-probe", action="store_true",
                    help="tar-bench: also probe each video's first prompt without its images "
                         "(videos.jsonl text_probe_s); does not change outputs or the run's name")
    ap.add_argument("--decode-probe", action="store_true",
                    help="tar-bench: after each video's questions, a single-stream decode probe "
                         "(videos.jsonl decode_probe); does not change outputs or the run's name")
    ap.add_argument("--videos-per-batch", type=int, default=1, metavar="N",
                    help="vLLM: submit N videos' questions per generate call, so long replies "
                         "(thinking mode) don't leave the GPU idle. Not part of the "
                         "configuration: batch-invariant models give identical replies, and "
                         "the others vary within their noise band either way. N > 1 drops "
                         "the per-video prefill probe, so use 1 for timing runs")
    ap.add_argument("--weights", metavar="TAG",
                    help="run another checkpoint of the model (runner/checkpoints.json), e.g. an "
                         "INT4 build, fetched with `python -m tar_runner.weights fetch TAG`")
    ap.add_argument("--constrained", action="store_true",
                    help="vLLM, thinking off: enforce the answer format (bcq Yes/No, mcq A-D, "
                         "temporal_localization's JSON) with structured outputs")
    ap.add_argument("--kv-cache-dtype", choices=("q8_0",),
                    help="llama.cpp: the KV cache's type, keys and values, with flash attention")
    ap.add_argument("--speculative", metavar="mtp:N",
                    help="vLLM: speculative decoding with the model's own MTP head, N tokens")
    args = ap.parse_args()
    if args.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"   # read by vLLM when the engine starts
    adapter = MODELS[args.model]
    if args.decoding is None:
        args.decoding = "vendor" if args.thinking == "on" else "greedy"
    if args.max_model_len is None:
        args.max_model_len = adapter.max_model_len + (
            (adapter.thinking_budget or contract.MAX_TOKENS_THINKING) if args.thinking == "on" else 0)
    if args.mode not in adapter.supported_modes:
        raise SystemExit(f"{args.model} does not support {args.mode}")
    if args.thinking not in adapter.thinking_modes:
        raise SystemExit(f"{args.model} has no thinking mode")
    if args.weights:
        e = weights.entry(args.weights)
        if e["model"] != args.model or e["runtime"] != WEIGHTS_RUNTIME.get(args.backend):
            raise SystemExit(f"{args.weights} is a {e['runtime']} checkpoint of {e['model']}; "
                             f"--backend {args.backend} runs "
                             f"{WEIGHTS_RUNTIME.get(args.backend, 'no')} checkpoints of --model")
    if args.backend in ("llamacpp", "edgellm"):
        if not args.weights:
            raise SystemExit(f"--backend {args.backend} runs a quantized checkpoint: pass --weights <tag>")
        if args.dtype != "bfloat16" or args.batch_invariant:
            raise SystemExit("--dtype and --batch-invariant are vLLM settings; the checkpoint and its "
                             "engines fix their own types")
    if args.constrained and (args.backend != "vllm" or args.thinking != "off"):
        raise SystemExit("--constrained needs --backend vllm and thinking off: a constraint would "
                         "cover the reasoning trace too")
    if args.kv_cache_dtype and args.backend != "llamacpp":
        raise SystemExit("--kv-cache-dtype is a llama.cpp setting: the Orin has no FP8 for vLLM's")
    if args.speculative:
        method, _, n = args.speculative.partition(":")
        if args.backend != "vllm" or method != "mtp" or not n.isdigit():
            raise SystemExit("--speculative takes mtp:N, with --backend vllm")
        args.speculative = {"method": "mtp", "num_speculative_tokens": int(n)}
    if args.backend == "hf" and adapter.hf_env and os.environ.get("TAR_HF_ENV") != adapter.hf_env:
        raise SystemExit(f"{args.model}'s HF reference runs in runner/{adapter.hf_env}/ "
                         f"(TAR_HF_ENV={adapter.hf_env}); see its adapter")

    cfg = config_of(args)
    out_dir = RUNS_ROOT / args.task_dir / run_name(cfg)
    if args.print_run_dir:
        print(out_dir)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path, vid_path = out_dir / "predictions.jsonl", out_dir / "videos.jsonl"

    rows = load_rows(args.slice, args.tasks, args.prefix)
    by_video: OrderedDict[str, list[dict]] = OrderedDict()
    for r in sorted(rows, key=lambda r: r["priority"]):
        by_video.setdefault(r["video_id"], []).append(r)
    done = set()
    if pred_path.exists():
        done = {json.loads(l)["item_id"] for l in pred_path.open() if l.strip()}
    # --limit-videos N: the first N videos by priority, so a rerun of the same
    # command finds them done instead of running N more.
    videos = list(by_video.items())[:args.limit_videos] if args.limit_videos else list(by_video.items())
    todo = [(v, rs) for v, rs in videos if not all(r["item_id"] in done for r in rs)]
    n_done = len(videos) - len(todo)
    scope = f"the first {len(videos)} of {len(by_video)}" if args.limit_videos else f"{len(by_video)}"
    print(f"{out_dir.name}\n{scope} videos ({len(rows)} items in all); "
          f"{n_done} videos already done, {len(todo)} to run")
    if not todo and (out_dir / "run.json").exists():
        # A finished run: loading the model only to rewrite run.json would
        # replace the timing and host of the session that produced it.
        return 0

    started = datetime.now(UTC).isoformat(timespec="seconds")
    t_start = time.perf_counter()
    backend = BACKENDS[args.backend](adapter, args.mode, args.frames, args.cap,
                                     dtype=args.dtype, thinking=args.thinking,
                                     decoding=args.decoding,
                                     enforce_eager=args.eager,
                                     gpu_memory_utilization=args.gpu_memory_utilization,
                                     max_model_len=args.max_model_len,
                                     cudagraphs=args.cudagraphs,
                                     model_dir=str(weights.local_path(args.weights))
                                     if args.weights else None,
                                     **backend_options(args))
    load_s = time.perf_counter() - t_start

    t_run = time.perf_counter()
    if args.videos_per_batch > 1 and not hasattr(backend, "run_videos"):
        raise SystemExit(f"--videos-per-batch needs the vLLM backend, not {args.backend}")
    with pred_path.open("a") as pf, vid_path.open("a") as vf:
        if args.videos_per_batch > 1:
            run_batched(todo, args, adapter, backend, pf, vf, t_run)
            todo_loop = []
        else:
            todo_loop = todo
        for n, (video_id, vrows) in enumerate(todo_loop, 1):
            reqs = requests_for(video_id, vrows, args, adapter)
            t_video = time.time()
            results, timing = backend.run_video(reqs)
            # Wall-clock bounds, for tar-bench to align power samples with videos.
            timing |= {"t_start": round(t_video, 3), "t_end": round(time.time(), 3)}
            for r, res in zip(vrows, results, strict=True):
                pf.write(json.dumps(res | {"task": r["task"], "video_id": video_id},
                                    ensure_ascii=False) + "\n")
            vf.write(json.dumps({"video_id": video_id, **timing}) + "\n")
            pf.flush()
            vf.flush()
            if n % 25 == 0 or n == len(todo):
                el = time.perf_counter() - t_run
                print(f"  {n}/{len(todo)} videos  {el / n:.2f} s/video  "
                      f"eta {el / n * (len(todo) - n) / 60:.1f} min", flush=True)

    import torch
    # How much of the slice the run covers. A run's name is its configuration,
    # and --limit-videos is not part of it: a 250-video cell has its
    # full-slice run's name. Read these before using a run.
    done = {json.loads(l)["item_id"] for l in pred_path.open() if l.strip()}
    covered = sum(1 for rs in by_video.values() if all(r["item_id"] in done for r in rs))
    manifest = {
        "name": out_dir.name, "config": cfg, "started": started,
        "finished": datetime.now(UTC).isoformat(timespec="seconds"),
        "coverage": {"limit_videos": args.limit_videos, "videos_done": covered,
                     "videos_in_slice": len(by_video)},
        "videos_run_this_session": len(todo), "load_s": round(load_s, 1),
        "videos_per_batch": args.videos_per_batch,
        "run_s": round(time.perf_counter() - t_run, 1),
        "backend": {"name": backend.name, "version": backend.version,
                    "engine_kwargs": backend.engine_kwargs},
        "frame_manifest_sha256": frame_manifest_sha256(args.slice),
        "git": git_state(),
        "host": {"gpu": torch.cuda.get_device_name(0), "python": platform.python_version(),
                 "torch": torch.__version__, "cuda_home": os.environ.get("CUDA_HOME")},
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"wrote {out_dir}")

    backend.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)   # vLLM's engine process can otherwise keep the interpreter alive


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:   # vLLM's engine process would otherwise keep a failed run alive, holding the GPU
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
