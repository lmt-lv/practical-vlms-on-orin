"""Run any configuration of the study by its id (configs/configs.tsv).

    uv run tar-config list [--group sweep]          # the configurations
    uv run tar-config show pivot.qwen3-vl-8b        # its tar-run command
    uv run tar-config weights pivot.qwen3-vl-8b     # fetch (or say how to build) its weights
    uv run tar-config run pivot.qwen3-vl-8b [--videos 20] [--videos-per-batch 8]
    uv run tar-config bench pivot.qwen3-vl-8b [--videos 23] [--warmup 3]   # on a Jetson AGX Orin

`run` measures quality: it runs the configuration over its subset (or the
first N videos with --videos) and writes predictions under
runs/<group>/<run name>/; score them with `tar-eval-score` (eval/). `bench`
measures time, power and energy on a Jetson: it runs the configuration on
the development subset under `tar-bench`, with the memory settings the
study used on the 32 GB Orin.

Every configuration runs as in the study: greedy decoding and thinking off
unless its arguments say otherwise, all ten question types, and vLLM's
batch-invariant kernels where the model supports them.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .adapters import MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs" / "configs.tsv"
RUNNER = REPO_ROOT / "runner"

#: The Orin runs' memory settings (32 GB shared by CPU and GPU). vLLM's
#: multiprocess engine segfaults in NVML with pinned clocks, so it runs in
#: process; glibc's per-thread arenas and vLLM's processor cache (4 GiB,
#: held twice) otherwise grow the process past the board's memory.
ORIN_ENV = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "MALLOC_ARENA_MAX": "2", "TAR_VLLM_MM_CACHE_GB": "1"}
#: vLLM's memory share on the Orin; the share sizes only the KV cache.
ORIN_GPU_UTIL = {"nemotron-3-nano-omni": 0.72}
ORIN_GPU_UTIL_DEFAULT = 0.65
#: Nemotron's KV cache is fixed instead: derived from the share it came out
#: negative on a fresh boot, and 1.5 GiB left the board thrashing.
ORIN_KV_CACHE_GB = {"nemotron-3-nano-omni": "0.4"}


def load() -> dict[str, dict]:
    rows, cols = {}, None
    for line in CONFIGS.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split("\t")
        if cols is None:
            cols = f
            continue
        r = dict(zip(cols, f + [""] * (len(cols) - len(f))))
        rows[r["id"]] = r
    return rows


def get(cid: str) -> dict:
    rows = load()
    if cid not in rows:
        raise SystemExit(f"unknown configuration {cid!r}; `tar-config list` shows them")
    return rows[cid]


def run_args(r: dict, videos: int | None, subset: str | None = None, task_dir: str | None = None) -> list[str]:
    """tar-run's arguments for a configuration, as the study ran it."""
    a = ["--model", r["model"], "--backend", r["backend"], "--slice", subset or r["subset"],
         "--tasks", "all", "--task-dir", task_dir or r["group"]]
    if r["dtype"] != "-":
        a += ["--dtype", r["dtype"]]
    a += shlex.split(r["args"])
    if r["backend"] == "vllm" and MODELS[r["model"]].batch_invariant:
        a.append("--batch-invariant")
    n = videos if videos is not None else (None if r["videos"] == "-" else int(r["videos"]))
    if n is not None:
        a += ["--limit-videos", str(n)]
    return a


def weights_tag(r: dict) -> str | None:
    a = shlex.split(r["args"])
    return a[a.index("--weights") + 1] if "--weights" in a else None


def python_for(r: dict) -> list[str]:
    """Edge-LLM runs in its own environment (runner/edgellm), with the runner on PYTHONPATH."""
    if r["backend"] == "edgellm":
        return [str(RUNNER / "edgellm" / ".venv" / "bin" / "python")]
    return [sys.executable]


def env_for(r: dict, extra: dict | None = None) -> dict:
    env = dict(os.environ)
    if r["backend"] == "edgellm":
        env["PYTHONPATH"] = str(RUNNER) + os.pathsep + env.get("PYTHONPATH", "")
        if r["id"].startswith("edgellm_b12.") and "TAR_EDGELLM_ENGINES" not in env:
            # the batched engines: MAX_BATCH=12 ENGINES_DIR=<these> runner/scripts/edgellm_engines.sh <tag>
            env["TAR_EDGELLM_ENGINES"] = str(REPO_ROOT / "models" / "edgellm" / f"{weights_tag(r)}.engines-b12")
    env.update(extra or {})
    return env


def shown(env: dict) -> str:
    """The environment settings a command adds, as a shell prefix."""
    return "".join(f"{k}={v} " for k, v in env.items() if os.environ.get(k) != v and k != "PYTHONPATH")


def cmd_list(group: str | None) -> int:
    for r in load().values():
        if group and r["group"] != group:
            continue
        print(f"{r['id']:40s} {r['group']:9s} {r['backend']:8s} {r['subset']:8s} "
              f"{r['videos']:>4s}  {r['args']}")
    return 0


def cmd_show(cid: str) -> int:
    r = get(cid)
    print("quality: uv run tar-run " + shlex.join(run_args(r, None)))
    if r["orin"] != "-":
        print("timing:  uv run tar-config bench " + cid + f"   # timed in: {r['orin']}")
    tag = weights_tag(r)
    print(f"weights: {tag or MODELS[r['model']].hf_id + ' (BF16)'}")
    return 0


def cmd_weights(cid: str) -> int:
    r = get(cid)
    from . import weights
    weights.fetch_base(r["model"])
    tag = weights_tag(r)
    if tag:
        if ".own-edgellm" in tag:
            print(f"{tag}: built on the Orin with runner/scripts/quantize_edgellm.sh and "
                  "runner/scripts/edgellm_engines.sh (docs/running.md, Edge-LLM)")
        elif ".own-" in tag:
            print(f"{tag}: our own build; make it with\n"
                  f"  (cd quantize && uv run python own_w4a16.py {tag})\n"
                  f"then: uv run python -m tar_runner.weights fetch {tag}")
        else:
            weights.main(["fetch", tag])
    return 0


def cmd_run(cid: str, videos: int | None, vpb: int | None, dry: bool) -> int:
    r = get(cid)
    a = run_args(r, videos)
    if vpb and r["backend"] == "vllm":
        a += ["--videos-per-batch", str(vpb)]
    cmd = [*python_for(r), "-m", "tar_runner.run", *a]
    env = env_for(r)
    print("$ " + shown(env) + shlex.join(cmd), flush=True)
    return 0 if dry else subprocess.call(cmd, cwd=RUNNER, env=env)


def cmd_bench(cid: str, videos: int, warmup: int, idle_s: int, dry: bool) -> int:
    r = get(cid)
    a = run_args(r, videos, subset="dev", task_dir=f"bench_{r['group']}")
    extra = dict(ORIN_ENV)   # as the study's board runs, whatever the backend
    if r["backend"] == "vllm":
        a += ["--gpu-memory-utilization", str(ORIN_GPU_UTIL.get(r["model"], ORIN_GPU_UTIL_DEFAULT))]
        if r["model"] in ORIN_KV_CACHE_GB:
            extra["TAR_VLLM_KV_CACHE_GB"] = ORIN_KV_CACHE_GB[r["model"]]
    cmd = [*python_for(r), "-m", "tar_runner.bench", "--warmup", str(warmup), "--idle-s", str(idle_s),
           "--", *a]
    env = env_for(r, extra)
    print("$ " + shown(env) + shlex.join(cmd), flush=True)
    return 0 if dry else subprocess.call(cmd, cwd=RUNNER, env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--group", choices=("baseline", "sweep", "tarbench", "thinking", "runtime"))
    for name in ("show", "weights"):
        sub.add_parser(name).add_argument("id")
    p = sub.add_parser("run")
    p.add_argument("id")
    p.add_argument("--videos", type=int, help="only the subset's first N videos by priority")
    p.add_argument("--videos-per-batch", type=int,
                   help="vLLM: videos per generate call (faster for long replies; same answers)")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("bench")
    p.add_argument("id")
    p.add_argument("--videos", type=int, default=23, help="development videos, warm-up included (study: 23 or 53)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--idle-s", type=int, default=30, help="idle power baseline before the run")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        return cmd_list(a.group)
    if a.cmd == "show":
        return cmd_show(a.id)
    if a.cmd == "weights":
        return cmd_weights(a.id)
    if a.cmd == "run":
        return cmd_run(a.id, a.videos, a.videos_per_batch, a.dry_run)
    return cmd_bench(a.id, a.videos, a.warmup, a.idle_s, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
