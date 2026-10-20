"""Measure a tar-run configuration on a Jetson: latency split, memory, power
and energy.

    uv run tar-bench [--interval-ms 100] [--idle-s 20] [--warmup 3] -- \\
        --model qwen3-vl-8b --backend vllm --task-dir bench --limit-videos 10 ...

Everything after `--` is a tar-run command line. The run directory must be
new: a resumed run would measure only its remainder. tar-run gets
`--text-probe` and `--decode-probe` (both after a video's answers, so no
reply changes), and the sampler runs in this process for the whole
session. Writes, beside the run's predictions.jsonl and videos.jsonl:

  power.jsonl    t, each INA3221 rail's V, A, W, and `module_w`: the sum of
                 VDD_GPU_SOC, VDD_CPU_CV and VIN_SYS_5V0 (the module's input
                 rails; the board input is not measured)
  thermal.jsonl  t, every thermal zone (°C), GPU/EMC/CPU clocks (MHz), used
                 memory (MemTotal - MemAvailable, MB)
  bench.json     the session (power mode, pinned clocks, idle baseline) and,
                 per video: prefill probe, text-only probe, image cost (probe
                 minus text probe), per-question increment; the three token
                 rates: prefill (probe prompt tokens over its time, and
                 the text-only probe's), single-stream decode (the decode
                 probe: 256 tokens, EOS ignored, against 1, the prefix
                 cached), throughput (output tokens over the questions' time,
                 batched on vLLM); end-to-end latency and energy up to the
                 video's last answer (the probes excluded), peak memory over
                 the baseline; then the medians over the measured videos, and
                 the throttling flag

Warm-up: the first `--warmup` videos are reported but excluded from the
medians. Throttling: a sample is throttled if the GPU clock is below its
pinned value (the devfreq minimum, when jetson_clocks has set min = max) or
tj is above the lowest trip point that throttles the GPU (99 °C); the run
is flagged if more than 1% of its measured samples are.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HWMON_INA = "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"
GPU_DEVFREQ = Path("/sys/devices/platform/17000000.gpu/devfreq_dev")
EMC_DEVFREQ = Path("/sys/class/devfreq/bwmgr")
#: The module's input rails, summed as the headline power.
MODULE_RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
THROTTLE_LIMIT = 0.01


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _int(path: Path) -> int | None:
    v = _read(path)
    return int(v) if v not in (None, "") and v.lstrip("-").isdigit() else None


def ina_rails() -> dict[str, tuple[Path, Path]]:
    """{label: (voltage file, current file)}, found by driver and label:
    hwmon numbers change across boots."""
    import glob
    rails = {}
    for d in sorted(glob.glob(HWMON_INA)):
        d = Path(d)
        for label in d.glob("in*_label"):
            n = label.name[2:-len("_label")]
            name = _read(label)
            if name and not name.startswith("sum") and (d / f"curr{n}_input").exists():
                rails[name] = (d / f"in{n}_input", d / f"curr{n}_input")
    return rails


def thermal_zones() -> dict[str, Path]:
    return {(_read(z / "type") or z.name).removesuffix("-thermal"): z / "temp"
            for z in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))}


def throttle_trip_c() -> float | None:
    """The lowest trip point, in °C, that throttles the GPU: a trip bound to
    its devfreq cooling device (99 °C passive on JetPack 7.2.1). The lower
    ones throttle nothing: tj's 75 and 95 °C drive the fan, and the 70 °C
    passive trips raise a hot-surface alert."""
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*"):
        for c in z.glob("cdev*_trip_point"):
            cdev = z / c.name.removesuffix("_trip_point")
            if _read(cdev / "type") == "devfreq-17000000.gpu":
                t = _int(z / f"trip_point_{_read(c)}_temp")
                if t:
                    temps.append(t / 1000)
    return min(temps) if temps else None


def used_mb() -> float:
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines()[:5]:
        k, v = line.split(":")
        info[k] = int(v.split()[0])
    return round((info["MemTotal"] - info["MemAvailable"]) / 1024, 1)


def power_mode() -> str | None:
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if "Power Mode" in line:
            return line.split(":", 1)[1].strip()
    return None


def clocks_state() -> dict:
    """The devfreq and cpufreq bounds: jetson_clocks sets min = max."""
    cpu = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    return {
        "gpu_min_mhz": (_int(GPU_DEVFREQ / "min_freq") or 0) / 1e6,
        "gpu_max_mhz": (_int(GPU_DEVFREQ / "max_freq") or 0) / 1e6,
        "emc_min_mhz": (_int(EMC_DEVFREQ / "min_freq") or 0) / 1e6,
        "emc_max_mhz": (_int(EMC_DEVFREQ / "max_freq") or 0) / 1e6,
        "cpu0_min_mhz": (_int(cpu / "scaling_min_freq") or 0) / 1e3,
        "cpu0_max_mhz": (_int(cpu / "scaling_max_freq") or 0) / 1e3,
        "cpu0_governor": _read(cpu / "scaling_governor"),
        "fan_pwm": _int(next(iter(Path("/sys/class/hwmon").glob("hwmon*/pwm1")), Path("/nonexistent"))),
    }


class Sampler(threading.Thread):
    """Reads the sensors every `interval_s` until stopped, appending one line
    per sample to power.jsonl and thermal.jsonl. Samples stay in memory only
    while `keep` is set (the idle baseline): at 10 Hz they grew tar-bench by
    ~0.1 GB/h, which a multi-hour run on the board can't spare. The
    summary reads them back with `load()` once tar-run has exited."""

    def __init__(self, out_dir: Path, interval_s: float):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.rails = ina_rails()
        self.zones = thermal_zones()
        self.cpus = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
        self.power_f = (out_dir / "power.jsonl").open("a")
        self.thermal_f = (out_dir / "thermal.jsonl").open("a")
        self.power: list[dict] = []
        self.thermal: list[dict] = []
        self.keep = True
        self.peak_used_mb = 0.0
        self._halt = threading.Event()

    def sample(self) -> None:
        t = round(time.time(), 3)
        p = {"t": t}
        module = 0.0
        for name, (vf, cf) in self.rails.items():
            v, a = _int(vf), _int(cf)
            if v is None or a is None:
                continue
            w = v * a / 1e6
            p[name] = {"v": v / 1000, "a": a / 1000, "w": round(w, 3)}
            if name in MODULE_RAILS:
                module += w
        p["module_w"] = round(module, 3)
        th = {"t": t, "temp_c": {k: (_int(f) or 0) / 1000 for k, f in self.zones.items()
                                 if _int(f) is not None},
              "gpu_mhz": (_int(GPU_DEVFREQ / "cur_freq") or 0) / 1e6,
              "emc_mhz": (_int(EMC_DEVFREQ / "cur_freq") or 0) / 1e6,
              "cpu_mhz": [(_int(c) or 0) / 1e3 for c in self.cpus],
              "used_mb": used_mb()}
        self.peak_used_mb = max(self.peak_used_mb, th["used_mb"] or 0.0)
        if self.keep:
            self.power.append(p)
            self.thermal.append(th)
        self.power_f.write(json.dumps(p) + "\n")
        self.thermal_f.write(json.dumps(th) + "\n")

    def run(self) -> None:
        nxt = time.perf_counter()
        while not self._halt.is_set():
            self.sample()
            nxt += self.interval_s
            self._halt.wait(max(0.0, nxt - time.perf_counter()))

    def stop(self) -> None:
        self._halt.set()
        self.join()
        self.power_f.close()
        self.thermal_f.close()

    def load(self) -> None:
        """Every sample, from the files written."""
        self.power = [json.loads(l) for l in Path(self.power_f.name).open() if l.strip()]
        self.thermal = [json.loads(l) for l in Path(self.thermal_f.name).open() if l.strip()]


def energy_j(samples: list[dict], t0: float, t1: float) -> tuple[float, float]:
    """(energy in J, mean module power in W) over [t0, t1], trapezoidal."""
    pts = [(s["t"], s["module_w"]) for s in samples if t0 <= s["t"] <= t1]
    if len(pts) < 2:
        return 0.0, pts[0][1] if pts else 0.0
    e = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pts, pts[1:]))
    return e, e / (pts[-1][0] - pts[0][0])


def summarize(out_dir: Path, sampler: Sampler, session: dict, warmup: int) -> dict:
    videos = [json.loads(l) for l in (out_dir / "videos.jsonl").open() if l.strip()]
    base_mem = session["idle"]["used_mb"]
    pinned_gpu = session["clocks"]["gpu_min_mhz"] if session["clocks"]["gpu_min_mhz"] == \
        session["clocks"]["gpu_max_mhz"] else None
    trip = session["throttle_trip_c"]
    rows, measured = [], []
    for i, v in enumerate(videos):
        # The video's own time ends with its answers (t_answered); the probes
        # after them (decode, text-only) are not part of it.
        t0, t1 = v["t_start"], v.get("t_answered", v["t_end"])
        e, w = energy_j(sampler.power, t0, t1)
        th = [s for s in sampler.thermal if t0 <= s["t"] <= t1]
        n_q = v["n_questions"]
        row = {
            "video_id": v["video_id"], "warmup": i < warmup, "n_questions": n_q,
            "prefill_probe_s": v.get("prefill_probe_s"),
            "text_probe_s": v.get("text_probe_s"),
            "image_cost_s": round(v["prefill_probe_s"] - v["text_probe_s"], 4)
            if v.get("text_probe_s") is not None else None,
            "per_question_s": round(v["questions_s"] / n_q, 4),
            # The three token rates:
            # prefill: the probe's prompt tokens over its time, images included;
            "prefill_tok_s": round(v["prompt_tokens_first"] / v["prefill_probe_s"], 1)
            if v.get("prefill_probe_s") else None,
            "text_prefill_tok_s": round(v["text_probe_tokens"] / v["text_probe_s"], 1)
            if v.get("text_probe_tokens") and v.get("text_probe_s") else None,
            # single-stream decode: the decode probe (backends.decode_rate);
            "decode_tok_s": (v.get("decode_probe") or {}).get("tok_s"),
            # throughput: all output tokens of the questions over their wall
            # time (batched for vLLM, sequential for llama.cpp and Edge-LLM).
            "throughput_tok_s": round(v["output_tokens"] / v["questions_s"], 2) if v["questions_s"] else None,
            "e2e_s": round(t1 - t0, 3),
            "energy_j": round(e, 2), "energy_per_item_j": round(e / n_q, 2),
            "mean_module_w": round(w, 2),
            "peak_mem_mb": round(max((s["used_mb"] for s in th), default=base_mem) - base_mem, 1),
            "gpu_mhz_min": min((s["gpu_mhz"] for s in th), default=None),
            "tj_max_c": max((s["temp_c"].get("tj", 0) for s in th), default=None),
        }
        if "server" in v:
            row["server"] = v["server"]
        if "decode_probe" in v:
            row["decode_probe"] = v["decode_probe"]
        rows.append(row)
        if i >= warmup:
            measured.extend(th)
    thr = [s for s in measured
           if (pinned_gpu is not None and s["gpu_mhz"] < pinned_gpu)
           or (trip is not None and s["temp_c"].get("tj", 0) > trip)]
    frac = len(thr) / len(measured) if measured else 0.0
    keep = [r for r in rows if not r["warmup"]]

    def med(key):
        vals = [r[key] for r in keep if r.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    return {
        "session": session,
        "videos": rows,
        "medians": {k: med(k) for k in ("prefill_probe_s", "text_probe_s", "image_cost_s",
                                         "per_question_s", "prefill_tok_s", "text_prefill_tok_s",
                                         "decode_tok_s", "throughput_tok_s", "e2e_s", "energy_j",
                                         "energy_per_item_j", "mean_module_w", "peak_mem_mb")},
        "n_measured_videos": len(keep),
        "throttle": {"pinned_gpu_mhz": pinned_gpu, "throttle_trip_c": trip,
                     "throttled_samples": len(thr), "measured_samples": len(measured),
                     "fraction": round(frac, 4), "throttled": frac > THROTTLE_LIMIT,
                     "clocks_pinned": pinned_gpu is not None},
    }


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: tar-bench [options] -- <tar-run arguments>")
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--idle-s", type=float, default=20.0,
                    help="idle baseline before the run starts (model not loaded)")
    ap.add_argument("--warmup", type=int, default=3, help="videos excluded from the medians")
    args = ap.parse_args(argv[:split])
    run_args = argv[split + 1:]
    if "--text-probe" not in run_args:
        run_args.append("--text-probe")
    if "--decode-probe" not in run_args:
        run_args.append("--decode-probe")

    run = [sys.executable, "-m", "tar_runner.run", *run_args]
    out_dir = Path(subprocess.run([*run, "--print-run-dir"], capture_output=True, text=True,
                                  check=True).stdout.strip().splitlines()[-1])
    if (out_dir / "predictions.jsonl").exists():
        raise SystemExit(f"{out_dir} already has predictions: a measured run starts fresh "
                         "(use another --task-dir)")
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = Sampler(out_dir, args.interval_ms / 1000)
    session = {"power_mode": power_mode(), "clocks": clocks_state(), "throttle_trip_c": throttle_trip_c(),
               "rails": sorted(sampler.rails), "module_rails": list(MODULE_RAILS),
               "interval_ms": args.interval_ms, "warmup_videos": args.warmup,
               "tar_run": run_args}
    sampler.start()
    t_idle = time.time()
    time.sleep(args.idle_s)
    e, w = energy_j(sampler.power, t_idle, time.time())
    session["idle"] = {"s": args.idle_s, "mean_module_w": round(w, 3),
                       "used_mb": statistics.median(s["used_mb"] for s in sampler.thermal)}
    sampler.keep = False
    sampler.power, sampler.thermal = [], []
    session["t_run_start"] = round(time.time(), 3)
    rc = subprocess.run(run).returncode
    session["t_run_end"] = round(time.time(), 3)
    session["tar_run_rc"] = rc
    session["peak_mem_run_mb"] = round(sampler.peak_used_mb - session["idle"]["used_mb"], 1)
    sampler.stop()
    sampler.load()
    if rc != 0 or not (out_dir / "videos.jsonl").exists():
        (out_dir / "bench.json").write_text(json.dumps({"session": session}, indent=2) + "\n")
        print(f"tar-run failed ({rc}); session in {out_dir / 'bench.json'}")
        return rc or 1
    summary = summarize(out_dir, sampler, session, args.warmup)
    (out_dir / "bench.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "bench.done").touch()
    print(json.dumps({"run": out_dir.name, **summary["medians"],
                      "throttled": summary["throttle"]["throttled"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
