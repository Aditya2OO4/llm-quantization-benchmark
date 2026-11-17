"""Small, dependency-light helpers: statistics, RAM sampling, output parsing."""

from __future__ import annotations

import re
import statistics
import threading

import psutil

from config import LONG_PROMPT_MIN_TOKENS


def summarize(values: list[float | None]) -> dict:
    """Mean / standard deviation / count, ignoring missing values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": statistics.fmean(vals), "std": std, "n": len(vals)}


def aggregate_runs(runs: list[dict]) -> dict:
    """Collapse per-run measurements into mean ± std summaries."""
    long_runs = [r for r in runs if r["prompt_tokens"] >= LONG_PROMPT_MIN_TOKENS]
    return {
        "decode_tps": summarize([r["decode_tps"] for r in runs]),
        "ttft_s": summarize([r["ttft_s"] for r in runs]),
        "prompt_tps": summarize([r["prompt_tps"] for r in long_runs]),
    }


_PPL_RE = re.compile(r"Final estimate: PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")


def parse_perplexity(text: str) -> tuple[float, float] | None:
    """Pull (ppl, stderr) out of llama-perplexity's output; None if absent."""
    match = _PPL_RE.search(text)
    return (float(match.group(1)), float(match.group(2))) if match else None


def rss_mb(proc: psutil.Process | None = None) -> float:
    return (proc or psutil.Process()).memory_info().rss / (1024**2)


class PeakRSSSampler:
    """Tracks the highest resident memory (RSS) of this process.

    A background thread samples every few ms. On Windows we also read the OS's
    own peak counter, so short spikes between samples are not missed.
    """

    def __init__(self, interval: float = 0.02) -> None:
        self._proc = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_mb = rss_mb(self._proc)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb(self._proc))
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakRSSSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def peak(self) -> float:
        os_peak = getattr(self._proc.memory_info(), "peak_wset", None)  # Windows only
        if os_peak:
            return max(self.peak_mb, os_peak / (1024**2))
        return self.peak_mb
