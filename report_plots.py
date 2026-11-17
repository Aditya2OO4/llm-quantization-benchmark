"""Matplotlib charts for step6_report.py (kept separate so plotting stays optional)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402


def _scatter(rows: list[dict], x: str, y: str, xlabel: str, ylabel: str, title: str, path: Path) -> Path | None:
    pts = [r for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(pts) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs, ys = [r[x] for r in pts], [r[y] for r in pts]
    ax.plot(xs, ys, "-", color="#9ca3af", zorder=1)
    ax.scatter(xs, ys, s=60, color="#2563eb", zorder=2)
    for r in pts:
        ax.annotate(r["label"], (r[x], r[y]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def thread_plot(speed: dict, path: Path) -> Path | None:
    sweep = sorted((c for c in speed["configs"] if c["kind"] == "thread_sweep"), key=lambda c: c["threads"])
    pts = [(c["threads"], c["summary"]["decode_tps"]["mean"]) for c in sweep]
    pts = [(t, v) for t, v in pts if v is not None]
    if len(pts) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(*zip(*pts), "o-", color="#2563eb")
    ax.set_xlabel("CPU threads")
    ax.set_ylabel("Decode speed (tokens/sec)")
    ax.set_title(f"Thread scaling ({sweep[0]['label']})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def make_all(rows: list[dict], speed: dict, out_dir: Path) -> list[Path]:
    made = [
        _scatter(rows, "size_mb", "ppl", "Model file size (MB)", "Perplexity (lower = better)",
                 "Smaller files cost quality", out_dir / "size_vs_quality.png"),
        _scatter(rows, "decode", "ppl", "Decode speed (tokens/sec)", "Perplexity (lower = better)",
                 "Faster models cost quality", out_dir / "speed_vs_quality.png"),
        thread_plot(speed, out_dir / "thread_scaling.png"),
    ]
    return [p for p in made if p]
