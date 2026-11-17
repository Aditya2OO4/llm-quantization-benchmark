"""
STEP 4: Speed + RAM benchmark across every quantization level
==============================================================
For each model file found in ./models this launches a fresh child process
(bench_worker.py) that loads the model and measures:

  - decode speed      tokens/sec while generating the answer
  - prompt speed      tokens/sec while reading a long prompt
  - time to 1st token latency before the answer starts
  - peak RAM          highest resident memory of that process
  - repeatability     mean ± std over several runs

Results are merged into results/speed.json (re-running one model keeps the rest).

Usage:
    python step4_benchmark.py                       # all models found, default threads
    python step4_benchmark.py --labels F16,Q4_K_M   # only some models
    python step4_benchmark.py --thread-sweep 1,2,4,8  # also: speed vs CPU threads
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import psutil

from bench_utils import aggregate_runs
from bench_worker import RESULT_PREFIX
from config import (
    ALL_LABELS,
    available_models,
    default_threads,
    file_size_mb,
    model_path,
    results_dir,
)

WORKER_TIMEOUT_S = 3600


def cpu_name() -> str:
    """Marketing name of the CPU (platform.processor() is cryptic on Windows)."""
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        return platform.processor() or platform.machine()


def hardware_info() -> dict:
    return {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "os": f"{platform.system()} {platform.release()}",
    }


def run_worker(path: Path, threads: int, repeats: int, max_tokens: int) -> dict:
    cmd = [
        sys.executable, "bench_worker.py",
        "--model", str(path), "--threads", str(threads),
        "--repeats", str(repeats), "--max-tokens", str(max_tokens),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    tail = (proc.stderr or proc.stdout)[-1500:]
    raise RuntimeError(f"worker failed for {path.name} (exit {proc.returncode}):\n{tail}")


def benchmark_one(kind: str, label: str, threads: int, repeats: int, max_tokens: int) -> dict:
    path = model_path(label)
    raw = run_worker(path, threads, repeats, max_tokens)
    return {
        "kind": kind,  # "main" or "thread_sweep"
        "label": label,
        "threads": threads,
        "file_mb": file_size_mb(path),
        "load_s": raw["load_s"],
        "peak_rss_mb": raw["peak_rss_mb"],
        "rss_after_load_mb": raw["rss_after_load_mb"],
        "baseline_rss_mb": raw["baseline_rss_mb"],
        "summary": aggregate_runs(raw["runs"]),
        "runs": raw["runs"],
    }


def merge_and_save(new_entries: list[dict], settings: dict) -> Path:
    out = results_dir() / "speed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"configs": []}
    key = lambda c: (c["kind"], c["label"], c["threads"])  # noqa: E731
    merged = {key(c): c for c in old["configs"]}
    merged.update({key(c): c for c in new_entries})
    data = {"hardware": hardware_info(), "settings": settings, "configs": list(merged.values())}
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def show(entry: dict) -> None:
    s = entry["summary"]["decode_tps"]
    mean = f"{s['mean']:.1f} ± {s['std']:.1f}" if s["mean"] is not None else "n/a"
    print(f"{mean} tok/s | peak RAM {entry['peak_rss_mb']:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--labels", help="comma list, e.g. F16,Q4_K_M (default: all found)")
    ap.add_argument("--threads", type=int, default=default_threads())
    ap.add_argument("--repeats", type=int, default=3, help="runs per prompt")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--thread-sweep", help="comma list of thread counts, e.g. 1,2,4,8")
    ap.add_argument("--sweep-label", default="Q4_K_M", help="model used for the thread sweep")
    args = ap.parse_args()

    wanted = args.labels.split(",") if args.labels else ALL_LABELS
    models = available_models(wanted)
    if not models:
        sys.exit("No models found in ./models. Run steps 2 and 3 first.")

    print(f"Hardware: {hardware_info()}")
    print(f"Threads={args.threads}  repeats={args.repeats}  max_tokens={args.max_tokens}\n")

    entries: list[dict] = []
    for label, _ in models:
        print(f"[{label:>6}] benchmarking ...", end=" ", flush=True)
        try:
            entries.append(benchmark_one("main", label, args.threads, args.repeats, args.max_tokens))
            show(entries[-1])
        except Exception as exc:  # keep going so one bad model doesn't lose the others
            print(f"FAILED\n{exc}")

    if args.thread_sweep:
        if not model_path(args.sweep_label).exists():
            sys.exit(f"--sweep-label model {args.sweep_label} not found")
        for t in (int(x) for x in args.thread_sweep.split(",")):
            print(f"[{args.sweep_label}] threads={t} ...", end=" ", flush=True)
            entries.append(benchmark_one("thread_sweep", args.sweep_label, t, args.repeats, args.max_tokens))
            show(entries[-1])

    settings = {"threads": args.threads, "repeats": args.repeats, "max_tokens": args.max_tokens}
    out = merge_and_save(entries, settings)
    print(f"\nSaved {out}\nNext: python step4b_perplexity.py   then   python step6_report.py")


if __name__ == "__main__":
    main()
