"""The input contract every model follows.

Fixed here, once, so that across models and runtimes only the model and
its serving configuration vary:

  image_list    every frame as an image, each preceded by its presentation
                time as text ("Frame at 00:03.75:"), then the question.
                The default for every model: all models and all Orin
                runtimes accept multiple images.
  native_video  the frames as one video, with the frame indices and source
                fps the model needs to place them in time; resampling off.
                Where supported; a deployment-sweep setting.
  no_frames     the question-only ablation: no visual input, one fixed line
                telling the model the video is unavailable.

In every mode the question is TAR's, verbatim, after the visual input, so
all questions about one video share a prefix and its encoding can be
reused. No system prompt; the model's own chat template; greedy,
unconstrained decoding with per-task token caps chosen to rarely bind.
"""

from __future__ import annotations

from .data import Frame

INPUT_MODES = ("image_list", "native_video", "no_frames")

IMAGE_LIST_HEADER = "The video is given as frames, each labelled with its timestamp."
NO_FRAMES_PREAMBLE = "The video is not available. Give your best guess, in the requested format."

#: Answer-style line for the generative tasks (style "plain", the default).
#: Without it, Qwen3-VL answered with markdown lists 2-3x the ground truth's
#: length and BERTScore scored the form rather than the content (dev:
#: summarization 0.05 -> 0.21 with the line). Not applied to the
#: bcq/mcq_openended pair: there it overrode the question's "letter
#: followed by an explanation" instruction, and 27% of mcq_openended
#: replies lost their letter. Same line for every model.
STYLE_LINE = "Answer in one paragraph of plain prose, without headings, lists or bold text."
STYLED_TASKS = frozenset({
    "open_qa", "scene_description", "video_summarization",
    "temporal_description", "causal_linkage",
})
STYLES = ("plain", "none")

#: Output token caps per task, thinking off; record whether a reply hit its
#: cap. Caps are there to stop runaways, not to shape answers, so they should
#: rarely bind. Caps of 64/256/512, first set on Qwen3-VL-8B, failed
#: Qwen3.5-9B, which explains before it answers even with thinking off: 64
#: tokens cut off 97 of 99 of its temporal_localization replies before the
#: JSON. One cap of 1,024 for every task: a model that answers sooner is
#: unaffected.
MAX_TOKENS = {task: 1024 for task in (
    "bcq", "mcq", "temporal_localization",
    "bcq_openended", "mcq_openended",
    "open_qa", "scene_description", "video_summarization",
    "temporal_description", "causal_linkage",
)}


#: With thinking on, one budget for every task: the trace and the answer
#: together. Replies that hit it are counted; above 5% on dev the budget is
#: raised rather than replies truncated.
MAX_TOKENS_THINKING = 8192


def max_tokens(task: str, thinking: str, budget: int | None = None) -> int:
    """A reply's cap: the task's with thinking off; with thinking on, one budget
    for trace and answer (the adapter's, if it sets one)."""
    return (budget or MAX_TOKENS_THINKING) if thinking == "on" else MAX_TOKENS[task]


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def user_content(question: str, frames: list[Frame], mode: str,
                 task: str = "", style: str = "none") -> list[dict]:
    """Chat content parts in the generic {"type": ...} form chat templates accept."""
    tail = [{"type": "text", "text": STYLE_LINE}] if style == "plain" and task in STYLED_TASKS else []
    if mode == "image_list":
        parts: list[dict] = [{"type": "text", "text": IMAGE_LIST_HEADER}]
        for f in frames:
            parts += [{"type": "text", "text": f"Frame at {mmss(f.pts_s)}:"}, {"type": "image"}]
        return parts + [{"type": "text", "text": question}] + tail
    if mode == "native_video":
        return [{"type": "video"}, {"type": "text", "text": question}] + tail
    if mode == "no_frames":
        return [{"type": "text", "text": f"{NO_FRAMES_PREAMBLE}\n\n{question}"}] + tail
    raise ValueError(f"unknown input mode {mode!r}")
