"""Check an adapter without the GPU: render one video's prompt and count the
tokens the processor gives it.

    uv run tar-probe --model lfm2.5-vl-3b [--thinking on] [--show-prompt]

Reports the prompt's token count with 16 frames and with none, and so the
vision tokens per frame; the processed image size where the processor
exposes it; and the tail of the rendered prompt, where the thinking switch
lands. This is the evidence for "the processor does not shrink our frames"
and the vision-token figures in outputs/runner/models.json.
"""

from __future__ import annotations

import argparse
import json

from . import contract
from .adapters import MODELS
from .data import frames_for, load_rows, source_video


def token_count(adapter, prompt: str, frames, mode: str, video: dict, n: int, cap: int) -> int:
    kw = adapter.hf_processor_inputs(prompt, frames, mode, video, n, cap)
    out = adapter.processor(**kw)
    return int(out["input_ids"].shape[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--cap", type=int, default=448)
    ap.add_argument("--thinking", default="off", choices=("off", "on"))
    ap.add_argument("--show-prompt", action="store_true")
    args = ap.parse_args()

    adapter = MODELS[args.model]
    row = next(r for r in load_rows("dev") if r["task"] == "mcq")
    frames = frames_for(row["video_id"], args.frames, args.cap)
    video = source_video(row["video_id"])

    with_frames = adapter.prompt(
        contract.user_content(row["question"], frames, "image_list", row["task"]), args.thinking)
    without = adapter.prompt(
        contract.user_content(row["question"], [], "image_list", row["task"]), args.thinking)
    n_with = token_count(adapter, with_frames, frames, "image_list", video, args.frames, args.cap)
    n_without = token_count(adapter, without, [], "image_list", video, args.frames, args.cap)

    size = None
    try:
        proc = adapter.processor.image_processor(images=[frames[0].image()], return_tensors="pt")
        for key in ("image_sizes", "image_grid_thw", "spatial_shapes", "pixel_values"):
            if key in proc:
                size = {key: proc[key].tolist() if key != "pixel_values" else list(proc[key].shape)}
                break
    except Exception as e:  # noqa: BLE001 - a report, not a gate
        size = {"error": repr(e)[:200]}

    report = {
        "model": adapter.key, "video_id": row["video_id"], "frames": args.frames, "cap": args.cap,
        "frame_size": frames[0].image().size, "thinking": args.thinking,
        "prompt_tokens_with_frames": n_with, "prompt_tokens_text_only": n_without,
        "vision_tokens_per_frame": round((n_with - n_without) / args.frames, 1),
        "processed_image": size,
        "prompt_tail": with_frames[-160:],
    }
    print(json.dumps(report, indent=2, default=str))
    if args.show_prompt:
        print(with_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
