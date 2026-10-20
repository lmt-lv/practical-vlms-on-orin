"""Redo the answer derivation of a run from its stored raw replies, with
the current adapter rules: the trace/answer split (thinking on) and the
adapter's answer extraction (both modes). The original file is kept as
predictions.jsonl.orig the first time.

    uv run python -m tar_runner.resplit runs/<task-dir>/<run>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .adapters import MODELS
from .backends import reply_fields

KEYS = ("prediction", "reasoning", "raw", "unclosed_trace", "leaked_trace")


def main() -> int:
    run = Path(sys.argv[1])
    cfg = json.loads((run / "run.json").read_text())["config"]
    thinking = cfg.get("thinking", "off")
    adapter = MODELS[cfg["model"]]
    pred = run / "predictions.jsonl"
    orig = run / "predictions.jsonl.orig"
    if not orig.exists():
        orig.write_bytes(pred.read_bytes())
    rows, changed = [], 0
    for r in map(json.loads, orig.open()):
        raw = r.get("raw", r["prediction"])
        new = reply_fields(adapter, raw, thinking, r["hit_cap"])
        out = {k: v for k, v in r.items() if k not in KEYS} | new
        changed += out.get("prediction") != r.get("prediction")
        rows.append(out)
    pred.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{run.name}: {len(rows)} replies, {changed} predictions changed, "
          f"{sum(bool(r.get('unclosed_trace')) for r in rows)} unclosed traces, "
          f"{sum('raw' in r and r['raw'] != r['prediction'] for r in rows)} answers extracted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
