"""Markdown for results/imatrix.json (used by step6_report.py)."""

from __future__ import annotations


def _f(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def report_lines(data: dict | None) -> list[str]:
    if not data or not data.get("runs"):
        return []
    f16 = data["f16"]["ppl"]
    out = ["", "## Importance matrix (imatrix) experiment", "",
           f"Calibration: {data['calibration']['source']}; evaluation: WikiText-2 test, "
           f"{data['eval']['chunks']} x {data['eval']['ctx']} tokens (same as the main sweep). "
           f"F16 perplexity {f16:.3f}. \"Gap closed\" = share of the quantization loss the imatrix wins back.", "",
           "| Level | Calibration chunks | Perplexity, no imatrix | Perplexity, with imatrix | vs F16 (plain → imatrix) | Gap closed |",
           "|---|---|---|---|---|---|"]
    for level, entry in data["runs"].items():
        plain = entry.get("plain")
        for n, r in sorted(entry["imatrix"].items(), key=lambda kv: int(kv[0])):
            if plain:
                delta = f"{(plain['ppl'] / f16 - 1) * 100:+.1f}% → {(r['ppl'] / f16 - 1) * 100:+.1f}%"
            else:
                delta = "-"
            closed = "-" if r.get("gap_closed_pct") is None else f"{r['gap_closed_pct']:.0f}%"
            out.append(f"| {level} | {n} | {_f(plain and plain['ppl'])} | {r['ppl']:.3f} ± {r['ppl_err']:.3f} | {delta} | {closed} |")
    return out
