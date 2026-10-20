"""Frame extraction and contact sheets.

Used by the viewer and by the 001c audit: a human (or a model) judging whether
a question is answerable needs to see the video, and a labelled grid of frames
with timestamps is the cheapest faithful substitute for playback.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from .paths import derived_dir, videos_dir
from .probe_media import PROBE_FILE, load_existing

_DURATIONS: dict[str, float] | None = None


def duration_of(video_id: str) -> float | None:
    global _DURATIONS
    if _DURATIONS is None:
        _DURATIONS = {
            k: v["duration_s"]
            for k, v in load_existing(derived_dir() / PROBE_FILE).items()
            if v.get("ok") and v.get("duration_s")
        }
    return _DURATIONS.get(video_id)


def extract_frames(video_id: str, n: int = 8, width: int = 320) -> list[tuple[float, Image.Image]]:
    """Sample n frames evenly across the video, skipping the very edges."""
    path = videos_dir() / video_id
    dur = duration_of(video_id)
    if not path.exists() or not dur:
        return []
    # Inset by half a step so the first frame is not a black lead-in.
    step = dur / n
    times = [round(step * (i + 0.5), 3) for i in range(n)]
    out = []
    for t in times:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path),
             "-frames:v", "1", "-vf", f"scale={width}:-1", "-f", "image2pipe",
             "-vcodec", "png", "-"],
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout:
            import io
            try:
                out.append((t, Image.open(io.BytesIO(proc.stdout)).convert("RGB")))
            except OSError:
                continue
    return out


def contact_sheet(
    frames: list[tuple[float, Image.Image]],
    cols: int = 4,
    label: str = "",
    cell_width: int = 320,
) -> Image.Image | None:
    """Grid of frames, each captioned with its timestamp."""
    if not frames:
        return None
    cell_h = max(im.height for _, im in frames)
    caption_h = 16
    rows = (len(frames) + cols - 1) // cols
    header_h = 18 if label else 0
    sheet = Image.new(
        "RGB", (cols * cell_width, header_h + rows * (cell_h + caption_h)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    if label:
        draw.text((4, 4), label[:140], fill="white")
    for idx, (t, im) in enumerate(frames):
        r, c = divmod(idx, cols)
        x, y = c * cell_width, header_h + r * (cell_h + caption_h)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + cell_h + 2), f"{int(t // 60):02d}:{t % 60:05.2f}", fill="yellow")
    return sheet


def grid_of_videos(
    video_ids: list[str], cols: int = 5, cell_width: int = 256, label: str = ""
) -> Image.Image | None:
    """One representative frame per video, captioned with an index and source.

    For the camera-type survey, where the question is "what kind of camera is
    this" rather than "what happens in this clip".
    """
    cells = []
    for n, vid in enumerate(video_ids):
        frames = extract_frames(vid, n=1, width=cell_width)
        if frames:
            cells.append((n, vid, frames[0][1]))
    if not cells:
        return None
    cell_h = max(im.height for _, _, im in cells)
    caption_h = 16
    rows = (len(cells) + cols - 1) // cols
    header_h = 18 if label else 0
    sheet = Image.new(
        "RGB", (cols * cell_width, header_h + rows * (cell_h + caption_h)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    if label:
        draw.text((4, 4), label[:140], fill="white")
    for pos, (idx, vid, im) in enumerate(cells):
        r, c = divmod(pos, cols)
        x, y = c * cell_width, header_h + r * (cell_h + caption_h)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + cell_h + 2), f"[{idx}] {vid.split('/')[0][:22]}", fill="yellow")
    return sheet


def write_sheet(image: Image.Image, name: str) -> Path:
    out_dir = derived_dir() / "sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    image.save(path)
    return path
