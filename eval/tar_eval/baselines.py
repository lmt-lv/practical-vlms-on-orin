"""Floor baselines: what the metrics give for answers that ignore the video.

Two baselines, each covering every task type, scored on the eval slice:

  random  bcq/mcq: a uniformly random answer. temporal: a random interval
          inside the clip. open-ended: a real ground-truth answer to the same
          task, drawn from a different (dev) video — fluent, on-topic, wrong.
  prior   bcq/mcq: the dev slice's most common answer. temporal: the whole
          clip. open-ended: one fixed generic answer per task, the dev
          slice's medoid under BERTScore.

The prior baseline's open-ended scores are the "generic-answer floor": how
much BERTScore pays for traffic prose that says nothing about the clip.
Everything used to build the baselines comes from dev, so eval stays clean.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from tar_tools.annotations import OPEN_ENDED_TASKS, TASKS, extract_letter, gt_yesno

from .common import REPORTS_DIR, RUNS_ROOT, SEED, read_jsonl, slice_path, write_jsonl
from .score import aggregate, bertscore_f1, score_items

OUT = RUNS_ROOT / "baselines"
MEDOID_SAMPLE = 80


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def medoid(answers: list[str]) -> str:
    """The answer with the highest mean BERTScore F1 against all the others."""
    n = len(answers)
    cands, refs, owner = [], [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                cands.append(answers[i])
                refs.append(answers[j])
                owner.append(i)
    scores = bertscore_f1(cands, refs)
    totals = [0.0] * n
    for i, s in zip(owner, scores, strict=True):
        totals[i] += s
    return answers[max(range(n), key=lambda i: totals[i])]


def build(dev: list[dict], eval_rows: list[dict]) -> tuple[dict, dict, dict]:
    rng = random.Random(SEED)
    dev_by_task: dict[str, list[dict]] = {}
    for r in dev:
        dev_by_task.setdefault(r["task"], []).append(r)

    majority_yesno = Counter(gt_yesno(r["answer"]) for r in dev_by_task["bcq"]).most_common(1)[0][0]
    majority_letter = Counter(extract_letter(r["answer"]) for r in dev_by_task["mcq"]).most_common(1)[0][0]
    generic = {}
    for task in sorted(OPEN_ENDED_TASKS):
        pool = [r["answer"] for r in dev_by_task[task]]
        pool = random.Random(SEED).sample(pool, min(MEDOID_SAMPLE, len(pool)))
        generic[task] = medoid(pool)

    rand, prior = {}, {}
    for r in eval_rows:
        task, iid, dur = r["task"], r["item_id"], r["duration_s"]
        if task in ("bcq", "bcq_openended"):
            rand[iid] = rng.choice(["Yes", "No"])
            prior[iid] = majority_yesno.capitalize() if task == "bcq" else generic[task]
        elif task in ("mcq", "mcq_openended"):
            rand[iid] = rng.choice("ABCD")
            prior[iid] = majority_letter if task == "mcq" else generic[task]
        elif task == "temporal_localization":
            a, b = sorted(rng.uniform(0, dur) for _ in range(2))
            rand[iid] = json.dumps({"start": mmss(a), "end": mmss(b)})
            prior[iid] = json.dumps({"start": mmss(0), "end": mmss(dur)})
        else:
            rand[iid] = rng.choice(dev_by_task[task])["answer"]
            prior[iid] = generic[task]
    info = {"majority_yesno": majority_yesno, "majority_letter": majority_letter,
            "generic_answers": generic}
    return rand, prior, info


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    dev = read_jsonl(slice_path("dev"))
    eval_rows = read_jsonl(slice_path("eval"))
    rand, prior, info = build(dev, eval_rows)

    results = {}
    for name, preds in (("random", rand), ("prior", prior)):
        write_jsonl(OUT / name / "predictions.jsonl",
                    [{"item_id": k, "prediction": v} for k, v in preds.items()])
        items = score_items(eval_rows, preds)
        write_jsonl(OUT / name / "items.jsonl", items)
        results[name] = aggregate(items)
        (OUT / name / "summary.json").write_text(json.dumps(results[name], indent=2) + "\n")

    report = {"slice": "eval", "built_from": "dev", "info": info,
              "results": {k: {"official": v["official"], "extra": v["extra"]}
                          for k, v in results.items()}}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baselines.json").write_text(json.dumps(report, indent=2) + "\n")
    (REPORTS_DIR / "baselines.md").write_text(render(report))
    print(render(report))
    return 0


def render(report: dict) -> str:
    r, p = report["results"]["random"], report["results"]["prior"]
    keys = list(r["official"]) + list(r["extra"])
    lines = [
        "# Floor baselines (eval slice)",
        "",
        "Answers that ignore the video, built from the dev slice only; see "
        "`eval/tar_eval/baselines.py`. BERTScore is rescaled, so 0 is roughly "
        "\"unrelated text\" and can go negative.",
        "",
        "| Metric | random | prior |",
        "|---|---:|---:|",
    ]
    for k in keys:
        rv = {**r["official"], **r["extra"]}[k]
        pv = {**p["official"], **p["extra"]}[k]
        lines.append(f"| {k} | {rv:.4f} | {pv:.4f} |")
    lines += ["", f"Prior answers: bcq = **{report['info']['majority_yesno']}**, "
              f"mcq = **{report['info']['majority_letter']}**, temporal = whole clip.", "",
              "## Generic answers (dev medoids)", ""]
    for task, text in report["info"]["generic_answers"].items():
        lines += [f"**{task}**", "", f"> {text}", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
