"""The paper's figures and tables, from the published results.

    uv run tar-eval-figures [tables|figures|all]

Reads results/results.json and writes:
  - outputs/figures/*.pdf and *.png: the paper's four figures;
  - outputs/tables/*.csv: the paper's tables, as data. Bold (the best model
    not trained on TAR in its size class) is marked with a trailing "*";
    superscripts a (trained on TAR's training data) and b (scored on its
    final answer) follow the model's name as "^a" and "^b".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import REPO_ROOT


def tex_escape(s: str) -> str:
    return re.sub(r"([_%&#$])", r"\\\1", s)

STORE = REPO_ROOT / "results" / "results.json"
OUT = REPO_ROOT / "outputs"

# --- figures ---------------------------------------------------------------------

#: The chart palette (dataviz reference instance): ink, chrome, the three
#: picks' categorical slots, the de-emphasis gray, and the diverging pair.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a"]
GRAY = "#a3a29c"
SURFACE = "#ffffff"
TASK_LABEL = {"bcq": "bcq", "bcq_openended": "bcq open", "mcq": "mcq", "mcq_openended": "mcq open",
              "temporal_localization": "TL IoU"}
KIND_LABEL = {"pivot": "INT4 configuration", "f8": "8 frames", "f32": "32 frames", "cap896": "896-px cap",
              "w16": "16-bit weights", "w8": "W8A16", "llama": "llama.cpp", "vendor": "vendor sampling",
              "constrained": "constrained decoding", "video": "native video", "kvq8": "llama.cpp, 8-bit KV cache",
              "q8mmproj": "llama.cpp, 8-bit vision encoder", "think_anchor": "thinking on (BF16)",
              "think_pivot": "thinking on (INT4)"}
KIND_ORDER = list(KIND_LABEL)


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlesize": 9,
        "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "pdf.fonttype": 42,
    })
    return plt


def pick_color(store: dict, m: str) -> str | None:
    picks = store["picks"]
    return SLOTS[picks.index(m)] if m in picks else None


def score_of(rec: dict, task: str) -> float | None:
    t = rec["per_task"].get(task)
    if not t:
        return None
    return t.get("final", t["score"]) if task in ("bcq", "bcq_openended", "mcq", "mcq_openended",
                                                    "temporal_localization") else t["score"]


def fig_pareto(store: dict, plt) -> str:
    """Quality (the INT4 configuration's exact-answer score) against time per video on
    the Orin at MAXN, with the Pareto frontier of the TAR-clean models."""
    models = store["models"]
    pts = []
    for m, r in models.items():
        q = r.get("pivot") or r.get("anchor")
        pts.append((m, r["orin"]["maxn"]["video_s"], q["closed_form"]["value"], q["closed_form"]["ci95"]))
    clean = [(x, y, m) for m, x, y, _ in pts if models[m]["tar_exposure"] != "confirmed"]
    front = sorted(p for p in clean if not any(q[0] <= p[0] and q[1] >= p[1] and q != p for q in clean))

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot([f[0] for f in front], [f[1] for f in front], "-", color=INK2, lw=1.0, alpha=0.5, zorder=1,
            label="Pareto frontier (TAR-clean models)")
    offsets = {"north-micro": (6, -10), "cosmos3-edge": (6, 9), "lfm2.5-vl-3b": (7, -3),
               "glm-4.6v-flash": (7, -9), "qwen3.8-27b": (-7, 7), "cosmos3-nano": (7, 2)}
    for m, x, y, ci in pts:
        r = models[m]
        c = pick_color(store, m) or GRAY
        exposed = r["tar_exposure"] == "confirmed"
        ax.errorbar(x, y, yerr=[[y - ci[0]], [ci[1] - y]], fmt="none", ecolor=c, elinewidth=1.0, capsize=2.5,
                    zorder=2)
        ax.scatter(x, y, s=70 if r["picked"] else 42, zorder=3, marker="D" if m == "qwen3.8-27b" else "o",
                   facecolors=SURFACE if exposed else c, edgecolors=c if exposed else SURFACE,
                   linewidths=1.5)
        text = r["label"] + (r"$^{\mathrm{a}}$" if exposed else "") + (r"$^{\mathrm{b}}$" if r["explains_first"] else "")
        if m == "qwen3.8-27b":
            text += "\n(BF16 score, llama.cpp time)"
        dx, dy = offsets.get(m, (7, 3))
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha="right" if dx < 0 else "left",
                    fontsize=7.5, color=INK if r["picked"] else INK2, fontweight="bold" if r["picked"] else "normal")
    ax.set_xscale("log")
    ax.set_xticks([8, 10, 15, 20, 30, 50, 100, 200])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xlim(7, 300)
    ax.set_xlabel("Seconds per video on the Jetson AGX Orin (MAXN; 16 frames, 12 questions)")
    ax.set_ylabel("Exact-answer score (%)")
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return save(fig, "pareto")


def fig_sensitivity(store: dict, plt) -> str:
    """The three models' sensitivity maps: each setting's paired delta against
    the BF16 reference, per exact-answer task and temporal localization."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    cmap = LinearSegmentedColormap.from_list(
        "div", ["#a8302f", "#e34948", "#f4b8b6", "#f0efec", "#b7d3f6", "#3987e5", "#1c5cab"])
    lim = 10.0
    cols = [("bcq", "bcq:correct"), ("bcq_openended", "bcq_openended:correct"), ("mcq", "mcq:correct"),
            ("mcq_openended", "mcq_openended:correct"), ("temporal_localization", "temporal_localization:iou0")]
    sweep = store["sweep"]
    picks = store["picks"]
    rows_of = {m: [k for k in KIND_ORDER if k in sweep[m]] for m in picks}
    heights = [len(rows_of[m]) + 1.2 for m in picks]
    fig, axes = plt.subplots(len(picks), 1, figsize=(5.6, 0.22 * sum(heights) + 0.3),
                             gridspec_kw={"height_ratios": heights})
    for ax, m in zip(axes, picks):
        kinds = rows_of[m]
        vals = np.full((len(kinds), len(cols)), np.nan)
        sig = np.zeros_like(vals, dtype=bool)
        for i, k in enumerate(kinds):
            q = sweep[m][k]["quality_vs_anchor"]
            for j, (_, key) in enumerate(cols):
                key = key.replace("iou0", "iou (unparseable = 0)").replace(":", "/")
                if key in q:
                    vals[i, j] = q[key]["delta"]
                    lo, hi = q[key]["ci95"]
                    sig[i, j] = lo > 0 or hi < 0
        ax.imshow(np.clip(vals, -lim, lim), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        for i in range(len(kinds)):
            for j in range(len(cols)):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                dark = abs(v) > 0.6 * lim
                ax.text(j, i, f"{v:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if dark else INK, fontweight="bold" if sig[i, j] else "normal")
        partial = {k for k in kinds if (sweep[m][k].get("orin") or {}).get("partial")}
        ax.set_yticks(range(len(kinds)), [KIND_LABEL[k] + (" \u00b9" if k in partial else "") for k in kinds],
                      fontsize=7)
        ax.set_xticks(range(len(cols)), [TASK_LABEL[t] for t, _ in cols], fontsize=7)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title(store["models"][m]["label"], loc="left", fontsize=8.5, pad=14, color=INK)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(kinds)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    # Bold: the 95% CI excludes zero; colours clipped at ±10 points; ¹: partial on the Orin (memory).
    fig.tight_layout()
    return save(fig, "sensitivity")


def fig_power(store: dict, plt) -> str:
    """Time against energy per question at MAXN, 30 W and 15 W, per pivot."""
    models = store["models"]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    marks = {"maxn": ("o", "MAXN"), "30w": ("s", "30 W"), "15w": ("^", "15 W")}
    for m, r in models.items():
        o = r["orin"]
        if not all(k in o for k in marks):
            continue
        c = pick_color(store, m) or GRAY
        xs = [o[k]["video_s"] for k in marks]
        ys = [o[k]["j_per_question"] for k in marks]
        ax.plot(xs, ys, "-", color=c, lw=1.2 if r["picked"] else 0.9, alpha=0.9, zorder=2)
        for (k, (mk, _)), x, y in zip(marks.items(), xs, ys):
            ax.scatter(x, y, marker=mk, s=40 if r["picked"] else 26, facecolors=c, edgecolors=SURFACE,
                       linewidths=1.2, zorder=3)
        dx, dy = {"lfm2.5-vl-3b": (-6, 6), "north-micro": (-6, -2), "cosmos3-edge": (-6, -4)}.get(m, (-6, 0))
        ax.annotate(r["label"] + (r"$^{\mathrm{a}}$" if r["tar_exposure"] == "confirmed" else ""), (xs[0], ys[0]),
                    xytext=(dx, dy), textcoords="offset points", ha="right", va="center", fontsize=7,
                    color=INK if r["picked"] else INK2, fontweight="bold" if r["picked"] else "normal")
    for k, (mk, label) in marks.items():
        ax.scatter([], [], marker=mk, s=30, facecolors=MUTED, edgecolors=SURFACE, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.set_yticks([20, 50, 100, 200])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.minorticks_off()
    ax.set_xlim(4, 300)
    ax.set_xlabel("Seconds per video (16 frames, 12 questions)")
    ax.set_ylabel("Module energy per question (J)")
    ax.grid(True)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="Power mode", loc="upper left", fontsize=7, title_fontsize=7)
    fig.tight_layout()
    return save(fig, "power_modes")


def fig_question_only(store: dict, plt) -> str:
    """What the video adds: question-only against with-video scores per task,
    for each TAR-clean model, beside the floors."""
    import numpy as np

    models = store["models"]
    floors = store["dataset"]["floors"]
    tasks = ["bcq", "bcq_openended", "mcq", "mcq_openended", "temporal_localization"]
    ms = [m for m, r in models.items() if r["tar_exposure"] != "confirmed" and "question_only" in r]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    spread = np.linspace(-0.28, 0.28, len(ms))
    for ti, t in enumerate(tasks):
        for k, m in enumerate(ms):
            r = models[m]
            q, v = score_of(r["question_only"], t), score_of(r["anchor"], t)
            y = ti + spread[k]
            c = pick_color(store, m) or GRAY
            ax.plot([q, v], [y, y], "-", color=c, lw=1.0, alpha=0.8, zorder=2)
            ax.scatter(q, y, s=16, facecolors=SURFACE, edgecolors=c, linewidths=1.0, zorder=3)
            ax.scatter(v, y, s=18, facecolors=c, edgecolors=SURFACE, linewidths=0.8, zorder=3)
        f = floors["prior"][t]
        ax.plot([f, f], [ti - 0.4, ti + 0.4], "-", color=INK2, lw=1.2, zorder=1)
    ax.scatter([], [], s=16, facecolors=SURFACE, edgecolors=MUTED, label="question only (no video)")
    ax.scatter([], [], s=18, facecolors=MUTED, edgecolors=SURFACE, label="with the video (BF16 anchor)")
    ax.plot([], [], "-", color=INK2, lw=1.2, label="prior floor (majority answer; whole clip for TL)")
    for k, m in enumerate(store["picks"]):
        ax.plot([], [], "-", color=SLOTS[k], lw=2, label=models[m]["label"])
    ax.plot([], [], "-", color=GRAY, lw=2, label="other TAR-clean models")
    ax.set_yticks(range(len(tasks)), [TASK_LABEL[t] for t in tasks])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%; temporal localization: mean IoU)")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5)
    fig.tight_layout()
    return save(fig, "question_only")


def save(fig, name: str) -> str:
    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    return name


def write_figures(store: dict) -> None:
    plt = setup()
    for f in (fig_pareto, fig_sensitivity, fig_power, fig_question_only):
        name = f(store, plt)
        plt.close("all")
        print(f"wrote outputs/figures/{name}.pdf")



# --- tables ----------------------------------------------------------------------

ORDER = ["lfm2.5-vl-3b", "north-micro", "cosmos3-edge", "qwen3-vl-8b", "qwen3.5-9b", "molmo2-8b",
         "glm-4.6v-flash", "cosmos3-nano", "nemotron-3-nano-omni", "qwen3.8-27b"]
#: The size classes for "best in class" (bold): small (up to 4B), 8-10B, large (27B dense, 31B MoE).
SIZE = {"small": ORDER[:3], "mid": ORDER[3:8], "large": ORDER[8:]}


def n(v, d=1, sign=False) -> str:
    """A number for a table: fixed decimals, a proper minus, thousands separated."""
    if v is None:
        return "---"
    s = f"{abs(v):,.{d}f}".replace(",", "{,}")
    if v < 0 and float(s.replace("{,}", "")) != 0:
        return "$-$" + s
    return ("+" if sign else "") + s


def best(store: dict, values: dict, higher: bool = True) -> set:
    """The models with the best value in their size class, among those not
    trained on TAR (bold in the tables)."""
    out = set()
    for ms in SIZE.values():
        vals = {m: values[m] for m in ms if values.get(m) is not None
                and store["models"][m]["tar_exposure"] != "confirmed"}
        if len(vals) >= 2:
            top = max(vals.values()) if higher else min(vals.values())
            out |= {m for m, v in vals.items() if v == top}
    return out


def cell(text: str, bold: bool) -> str:
    return f"\\textbf{{{text}}}" if bold and text != "---" else text


def label(store: dict, m: str) -> str:
    r = store["models"][m]
    t = tex_escape(r["label"])
    if r["tar_exposure"] == "confirmed":
        t += "\\textsuperscript{a}"
    if r["explains_first"]:
        t += "\\textsuperscript{b}"
    return t


def int4_build(r: dict) -> str:
    i = r["int4"]
    if i["format"].startswith("GGUF"):
        return "GGUF Q4\\_K\\_M"
    method = "RTN" if i["own_build"] else ("AWQ" if "awq" in (i["repo"] or "").lower() else "W4A16")
    return f"{method}, g{i['group']}" + (" (ours)" if i["own_build"] else "")


def plain(cell: str) -> str:
    """A table cell's LaTeX as plain text."""
    t = cell
    t = re.sub(r"\\textbf\{([^{}]*)\}", r"\1*", t)
    t = re.sub(r"\\textsuperscript\{([^{}]*)\}", r"^\1", t)
    t = re.sub(r"\\multicolumn\{\d+\}\{[^{}]*\}\{([^{}]*)\}", r"\1", t)
    for a, b in (("\\ensuremath{-}", "-"), ("$-$", "-"), ("$\\Delta$", "Δ"), ("$\\times$", "×"),
                 ("\\%", "%"), ("{,}", ","), ("\\_", "_"), ("---", "—"), ("--", "–"), ("~", " ")):
        t = t.replace(a, b)
    return t.replace("$", "").strip()


def tab(cols: str, head: list[str], rows: list[list[str]], groups: list[int] = (), wide: bool = False) -> str:
    """A table as CSV (cols, groups and wide are the LaTeX layout's, unused here)."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([plain(h) for h in head])
    for r in rows:
        w.writerow([plain(c) for c in r])
    return buf.getvalue()


def table_models(store: dict) -> str:
    rows = []
    for m in ORDER:
        r = store["models"][m]
        size = n(r["params_b"]) + (f" ({n(r['active_params_b'])} active)" if r["active_params_b"] else "")
        rows.append([label(store, m), size, tex_escape(r["language_model"]), r["released"], int4_build(r)])
    return tab("llLll", ["Model", "Params (B)", "Language model", "Released", "INT4 weights"], rows,
               groups=[3, 8], wide=True)


def table_baselines(store: dict) -> str:
    ms = store["models"]
    tl = lambda rec: (rec["per_task"]["temporal_localization"].get("final")
                      or rec["per_task"]["temporal_localization"]["score"])
    cols = {
        "bf16": {m: ms[m]["anchor"]["closed_form"]["value"] for m in ORDER},
        "int4": {m: (ms[m].get("pivot") or {}).get("closed_form", {}).get("value") for m in ORDER},
        "d": {m: ms[m].get("int4_cost") for m in ORDER},
        "tl": {m: tl(ms[m]["anchor"]) for m in ORDER},
        "dtl": {m: (ms[m].get("pivot_vs_anchor") or {}).get("temporal_localization", {}).get("delta") for m in ORDER},
    }
    b = {k: best(store, v) for k, v in cols.items()}
    rows = []
    for m in ORDER:
        r = ms[m]
        a, q = r["anchor"]["closed_form"], (r.get("question_only") or {}).get("closed_form")
        rows.append([label(store, m),
                     cell(f"{n(a['value'])}", m in b["bf16"]), f"{n(a['ci95'][0])}--{n(a['ci95'][1])}",
                     cell(n(cols["int4"][m]), m in b["int4"]), cell(n(cols["d"][m], sign=True), m in b["d"]),
                     cell(n(cols["tl"][m]), m in b["tl"]), cell(n(cols["dtl"][m], sign=True), m in b["dtl"]),
                     n(q["value"]) if q else "---", n(r.get("video_adds"))])
    return tab("lcCCCCCCC", ["Model", "BF16", "95\\% CI", "INT4", "$\\Delta$ INT4", "TL (BF16)",
                                    "$\\Delta$ TL INT4", "Question only", "Video adds"],
               rows, groups=[3, 8], wide=True)


def table_board(store: dict) -> str:
    ms = store["models"]
    o = {m: ms[m]["orin"]["maxn"] for m in ORDER}
    spec = [("video_s", 1, False), ("prefill_tok_s", 0, True), ("decode_tok_s", 1, True),
            ("throughput_tok_s", 1, True), ("j_per_question", 1, False), ("mean_w", 1, None), ("peak_gib", 1, None)]
    b = {k: best(store, {m: o[m][k] for m in ORDER}, hi) if hi is not None else set() for k, _, hi in spec}
    rows, groups = [], []
    for ms_ in SIZE.values():
        groups.append(len(rows))
        for m in sorted(ms_, key=lambda m: o[m]["video_s"]):
            rows.append([label(store, m), "vLLM" if o[m]["runtime"] == "vllm" else "llama.cpp"]
                        + [cell(n(o[m][k], d), m in b[k]) for k, d, _ in spec])
    return tab("llCCCCCCC", ["Model", "Runtime", "Video (s)", "Prefill (tok/s)", "Decode (tok/s)",
                                     "Batched (tok/s)", "Energy (J/q)", "Power (W)", "Peak (GiB)"], rows,
               groups=groups[1:], wide=True)


SETTING_ROWS = ["pivot", "f8", "f32", "cap896", "w8", "w16", "llama", "kvq8", "q8mmproj", "constrained",
                "vendor", "video", "think_pivot"]


def table_settings(store: dict) -> str:
    sweep, picks = store["sweep"], store["picks"]
    rows = []
    for k in SETTING_ROWS:
        if not any(k in sweep[m] for m in picks):
            continue
        row = [tex_escape(KIND_LABEL[k])]
        for m in picks:
            rec = sweep[m].get(k)
            if not rec:
                row += ["", "", ""]
                continue
            vs = (rec.get("orin") or {}).get("vs_pivot")
            part = " \\textsuperscript{c}" if (rec.get("orin") or {}).get("partial") else ""

            def times(x):
                return n(x, 2 if x < 10 else 0) + "$\\times$"
            if vs:
                energy = (times(vs["j_per_question"]) if vs["j_per_question"] >= 5
                          else n(100 * (vs["j_per_question"] - 1), 0, sign=True) + "\\%")
                row += [n(rec.get("closed_form_delta"), sign=True), times(vs["video_s"]) + part, energy]
            else:
                row += [n(rec.get("closed_form_delta"), sign=True), "1.00$\\times$" if k == "pivot" else "---",
                        "0\\%" if k == "pivot" else "---"]
        rows.append(row)
    head = ["Setting"] + [f"{store['models'][m]['label']} {x}" for m in picks
                          for x in ("ΔQ (points)", "time (× INT4)", "energy (vs INT4)")]
    return tab("l" + "rrr" * len(picks), head, rows)


def table_runtimes(store: dict) -> str:
    rt, sweep, ms = store["runtimes"], store["sweep"], store["models"]
    eq = rt["edgellm_quality"]["vs_anchor"]["closed_form"]["delta"]
    blocks = [("qwen3-vl-8b", [("vLLM", ms["qwen3-vl-8b"]["orin"]["maxn"], ms["qwen3-vl-8b"].get("int4_cost")),
                               ("TensorRT Edge-LLM", rt["edgellm.qwen3-vl-8b"], eq),
                               ("Edge-LLM, batched", rt["edgellm_b12.qwen3-vl-8b"], None),
                               ("llama.cpp", rt["llama.qwen3-vl-8b"], sweep["qwen3-vl-8b"]["llama"].get("closed_form_delta"))]),
              ("lfm2.5-vl-3b", [("vLLM", ms["lfm2.5-vl-3b"]["orin"]["maxn"], ms["lfm2.5-vl-3b"].get("int4_cost")),
                                ("llama.cpp", rt["llama.lfm2.5-vl-3b"], sweep["lfm2.5-vl-3b"]["llama"].get("closed_form_delta"))]),
              ("nemotron-3-nano-omni", [("vLLM", ms["nemotron-3-nano-omni"]["orin"]["maxn"],
                                         ms["nemotron-3-nano-omni"].get("int4_cost")),
                                        ("llama.cpp", rt["llama.nemotron-3-nano-omni"],
                                         sweep["nemotron-3-nano-omni"]["llama"].get("closed_form_delta"))])]
    spec = [("video_s", False), ("decode_tok_s", True), ("throughput_tok_s", True), ("j_per_question", False)]
    rows = []
    for m, runs in blocks:
        bests = {}
        for k, hi in spec:
            vals = [o[k] for _, o, _ in runs]
            bests[k] = max(vals) if hi else min(vals)
        dqs = [dq for _, _, dq in runs if dq is not None]
        for i, (name, o, dq) in enumerate(runs):
            rows.append([tex_escape(ms[m]["label"]) if i == 0 else "", name]
                        + [cell(n(o[k]), o[k] == bests[k]) for k, _ in spec]
                        + [cell(n(dq, sign=True), dq is not None and dq == max(dqs))])
    return tab("llCCCCC", ["Model", "Runtime", "Video (s)", "Decode (tok/s)", "Batched (tok/s)",
                                   "Energy (J/q)", "$\\Delta$Q"], rows, groups=[4, 6], wide=True)


def table_tarbench(store: dict) -> str:
    """TAR-Bench: the server's 9-task mean (the leaderboard's) and 10-task mean
    (the TAR paper's definition), the published zero-shot mean, the yes/no and
    multiple-choice accuracies, our evaluation subset's 9-task mean and the gap between the
    two (the contamination probe)."""
    runs = store["tarbench"]["runs"]
    srv = {m: (runs.get(f"tarbench.{m}") or {}).get("server") or {} for m in ORDER}
    b = {k: best(store, {m: srv[m].get(k) for m in ORDER}) for k in ("mean", "mean_10", "bcq", "mcq")}
    rows = []
    for m in ORDER:
        for cid in (f"tarbench.{m}", f"tarbench_think.{m}"):
            r = runs.get(cid)
            if not r:
                continue
            s = r.get("server") or {}
            think = r["thinking"]
            anchor = store["models"][m]["anchor"]
            rows.append([label(store, m) + (" (thinking)" if think else "")]
                        + [cell(n(s.get("mean_10")), not think and m in b["mean_10"]), n(r["published_mean"])]
                        + [cell(n(s.get("mean")), not think and m in b["mean"]),
                           n(anchor.get("official_mean_9")) if not think else "---", n(r.get("eval_minus_tarbench"))]
                        + [cell(n(s.get(k)), not think and m in b[k]) for k in ("bcq", "mcq")])
    return tab("lCCCCCCC", ["Model", "Ten-task mean (server)", "Ten-task mean (published)",
                            "Nine-task mean (server)", "Nine-task mean (our subset)", "Gap",
                            "Yes/no accuracy (server)", "Multiple-choice accuracy (server)"],
               rows, groups=[3, 8], wide=True)


#: TAR's sources, as its annotation files name them, with what our camera
#: survey (001c) found in them; in table order, used sources first.
SOURCES = {"so-tad": ("so-tad", "road CCTV"), "TAD-benchmark": ("TAD-benchmark", "road CCTV"),
           "TAD": ("TAD", "road CCTV"), "Accident-Bench": ("Accident-Bench", "road CCTV, accidents"),
           "HTV": ("HTV", "one highway camera"),
           "barbados_traffic_challenge": ("barbados", "one roundabout camera"),
           "Vad-R1": ("Vad-R1", "mostly campus walkways"),
           "UCF_Crimes": ("UCF-Crime", "surveillance, incl. crime")}


def table_sources(store: dict) -> str:
    src = store["dataset"]["sources"]
    keys = ["annotated", "pool_anomalous", "eval_anomalous", "eval_normal", "dev"]
    rows = []
    for k, (name, what) in SOURCES.items():
        r = src[k]
        note = {"Vad-R1": " (dropped)", "UCF_Crimes": " (skipped)"}.get(k, "")
        rows.append([name + note, what] + [f"{r[c]:,}" if r["pool"] or c == "annotated" else "---" for c in keys])
    tot = {c: sum(r[c] for r in src.values()) for c in keys}
    rows.append(["Total", ""] + [f"{tot[c]:,}" for c in keys])
    return tab("llCCCCC", ["Source", "Content", "Training videos", "Training anomalous", "Eval anomalous",
                           "Eval normal", "Dev videos"], rows, groups=[6, 8], wide=True)


TABLES = {"sources": table_sources, "models": table_models, "baselines": table_baselines, "board": table_board,
          "settings": table_settings, "runtimes": table_runtimes, "tarbench": table_tarbench}


def write_tables(store: dict) -> None:
    out = OUT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    for name, f in TABLES.items():
        (out / f"{name}.csv").write_text(f(store))
    print(f"wrote {len(TABLES)} tables to outputs/tables/")

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", default="all", choices=["tables", "figures", "all"])
    args = ap.parse_args()
    store = json.loads(STORE.read_text())
    if args.what in ("tables", "all"):
        write_tables(store)
    if args.what in ("figures", "all"):
        write_figures(store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
