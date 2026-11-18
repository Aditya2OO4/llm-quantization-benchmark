"""
STEP 6: Build the final report from the measured results
=========================================================
Reads results/speed.json (+ perplexity.json, load_test.json if present) and writes:

  results/RESULTS.md                 tables, headline numbers, caveats
  results/size_vs_quality.png        file size vs perplexity
  results/speed_vs_quality.png       decode speed vs perplexity
  results/thread_scaling.png         speed vs CPU threads (if you ran --thread-sweep)

Usage:
    python step6_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import custom_quant_report
from config import ALL_LABELS, REFERENCE_LABEL, results_dir

# "Good enough" quality budget for the recommendation. 5% is an arbitrary choice,
# not a standard: override with --budget or $MAX_PPL_INCREASE_PCT, and see
# budget_table() for what changes at 2% / 5% / 10%.
MAX_PPL_INCREASE_PCT = float(os.environ.get("MAX_PPL_INCREASE_PCT", 5.0))
BUDGETS_SHOWN = (2.0, 5.0, 10.0)


def load_json(name: str):
    path = results_dir() / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_rows(speed: dict, ppl: dict | None) -> list[dict]:
    """One row per quantization level, with comparisons against F16."""
    ppl = ppl or {}
    rows = []
    for c in speed["configs"]:
        if c["kind"] != "main":
            continue
        s = c["summary"]
        rows.append({
            "label": c["label"], "size_mb": c["file_mb"], "peak_rss_mb": c["peak_rss_mb"],
            "load_s": c["load_s"],
            "decode": s["decode_tps"]["mean"], "decode_std": s["decode_tps"]["std"],
            "prompt": s["prompt_tps"]["mean"], "ttft": s["ttft_s"]["mean"],
            "ppl": ppl.get(c["label"], {}).get("ppl"), "ppl_err": ppl.get(c["label"], {}).get("ppl_err"),
        })
    rows.sort(key=lambda r: ALL_LABELS.index(r["label"]) if r["label"] in ALL_LABELS else 99)

    ref = next((r for r in rows if r["label"] == REFERENCE_LABEL), None)
    for r in rows:
        r["size_pct"] = r["size_mb"] / ref["size_mb"] * 100 if ref else None
        r["speedup"] = r["decode"] / ref["decode"] if ref and ref["decode"] and r["decode"] else None
        has_ppl = ref and ref["ppl"] and r["ppl"]
        r["ppl_delta_pct"] = (r["ppl"] / ref["ppl"] - 1) * 100 if has_ppl else None
    return rows


def recommend(rows: list[dict], budget_pct: float | None = None) -> dict | None:
    """Smallest model whose perplexity is within the quality budget of F16."""
    budget = MAX_PPL_INCREASE_PCT if budget_pct is None else budget_pct
    ok = [r for r in rows if r["label"] != REFERENCE_LABEL and r["ppl_delta_pct"] is not None
          and r["ppl_delta_pct"] <= budget]
    return min(ok, key=lambda r: r["size_mb"]) if ok else None


def budget_table(rows: list[dict], budgets: tuple[float, ...] = BUDGETS_SHOWN) -> list[str]:
    """Which level is recommended at each quality budget (shows how arbitrary one budget is)."""
    if not any(r["ppl_delta_pct"] is not None for r in rows):
        return []
    lines = ["", "## Recommendation vs. quality budget", "",
             "The budget is a judgement call, not a standard. The pick changes with it:", "",
             "| Max perplexity increase | Picked level | Size MB | vs F16 |", "|---|---|---|---|"]
    for b in budgets:
        pick = recommend(rows, b)
        lines.append(f"| {b:g}% | {pick['label']} | {pick['size_mb']:.0f} | {pick['ppl_delta_pct']:+.1f}% |"
                     if pick else f"| {b:g}% | none | - | - |")
    return lines


def headline(row: dict) -> str:
    parts = [f"{row['label']}: {100 - row['size_pct']:.0f}% smaller than F16"]
    if row["speedup"]:
        parts.append(f"{row['speedup']:.1f}x faster decoding")
    if row["ppl_delta_pct"] is not None:
        parts.append(f"{row['ppl_delta_pct']:+.1f}% perplexity")
    return ", ".join(parts)


def fmt(value, spec: str = ".1f", dash: str = "-") -> str:
    return dash if value is None else format(value, spec)


def table(rows: list[dict]) -> list[str]:
    head = ("| Level | Size MB | % of F16 | Peak RAM MB | Decode tok/s (±std) | Prompt tok/s "
            "| Speedup | Perplexity | vs F16 |")
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['size_mb']:.0f} | {fmt(r['size_pct'], '.0f')}% | {r['peak_rss_mb']:.0f} "
            f"| {fmt(r['decode'])} ± {fmt(r['decode_std'])} | {fmt(r['prompt'], '.0f')} "
            f"| {fmt(r['speedup'], '.2f')}x | {fmt(r['ppl'], '.3f')} ± {fmt(r['ppl_err'], '.3f')} "
            f"| {fmt(r['ppl_delta_pct'], '+.1f')}% |"
        )
    return lines


def samples(speed: dict) -> list[str]:
    """Same prompt, different quant levels - shows the quality loss in plain English."""
    lines = ["| Level | Answer to: \"Explain what a neural network is in simple terms.\" (first 220 chars) |",
             "|---|---|"]
    for c in speed["configs"]:
        if c["kind"] != "main":
            continue
        run = next((r for r in c["runs"] if r["prompt_id"] == "short"), None)
        if run:
            text = run["text"].replace("\n", " ").replace("|", "/")[:220]
            lines.append(f"| {c['label']} | {text} |")
    return lines


def thread_table(speed: dict) -> list[str]:
    sweep = sorted((c for c in speed["configs"] if c["kind"] == "thread_sweep"), key=lambda c: c["threads"])
    if not sweep:
        return []
    lines = ["", f"## Thread scaling ({sweep[0]['label']})", "", "| Threads | Decode tok/s |", "|---|---|"]
    lines += [f"| {c['threads']} | {fmt(c['summary']['decode_tps']['mean'])} |" for c in sweep]
    return lines


def load_table(load: dict | None) -> list[str]:
    if not load:
        return []
    lines = ["", "## API load test", "",
             "| Concurrent clients | Req/s | Tokens/s | p50 latency s | p95 latency s |", "|---|---|---|---|---|"]
    lines += [f"| {x['concurrency']} | {x['req_per_s']:.2f} | {x['tokens_per_s']:.1f} "
              f"| {x['p50_s']:.2f} | {x['p95_s']:.2f} |" for x in load["levels"]]
    return lines


def build_markdown(speed: dict, rows: list[dict], load: dict | None, budget_pct: float | None = None,
                   custom: dict | None = None, ppl: dict | None = None) -> str:
    budget = MAX_PPL_INCREASE_PCT if budget_pct is None else budget_pct
    hw, st = speed["hardware"], speed["settings"]
    out = ["# Quantization results: TinyLlama 1.1B", "",
           f"**Hardware:** {hw['cpu']} | {hw['physical_cores']} cores / {hw['logical_cores']} threads "
           f"| {hw['ram_gb']} GB RAM | {hw['os']}",
           f"**Settings:** {st['threads']} threads, {st['repeats']} runs per prompt, "
           f"{st['max_tokens']} max new tokens, greedy decoding", ""]
    q4 = next((r for r in rows if r["label"] == "Q4_K_M"), None)
    if q4:
        out += [f"**Headline:** {headline(q4)}", ""]
    best = recommend(rows, budget)
    if best:
        out += [f"**Recommendation:** {best['label']} is the smallest level within "
                f"{budget:g}% of F16 perplexity ({headline(best)}). The {budget:g}% budget is an arbitrary choice; "
                "see the budget table below.", ""]
    out += ["## Size, speed and quality", "", *table(rows), "", "![size vs quality](size_vs_quality.png)",
            "![speed vs quality](speed_vs_quality.png)", "", "## Same prompt, every level", "", *samples(speed)]
    out += budget_table(rows) + thread_table(speed) + load_table(load)
    out += custom_quant_report.report_lines(custom, ppl)
    out += ["", "## How to read this / caveats", "",
            "- **Perplexity** = how surprised the model is by WikiText-2 text (lower = better). It is a proxy for quality, not a task score.",
            "- **Peak RAM** = highest resident memory of the process. llama.cpp memory-maps the model file, so this includes the model's pages, plus a roughly constant overhead for Python, libraries and the 1024-token KV cache. Compare levels by the differences, not the absolute floor.",
            "- **Repacking effect:** on CPUs with AVX2, llama.cpp repacks `Q4_K` tensors into an interleaved layout (`q4_K_8x8`) with faster kernels, at the cost of an extra in-memory copy. Verified with `verbose=True` loading: Q4_K_M +455 MiB, Q3_K_M +185 MiB, Q2_K +6 MiB, all other levels none. That is why Q4_K_M's RAM is higher than Q5_K_M/Q6_K and why its speed is relatively high.",
            "- **Decode tok/s** excludes prompt processing; **Prompt tok/s** is measured on a ~100+ token prompt.",
            "- Each level ran in a fresh process; std is across repeated runs. Numbers are specific to this CPU."]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=MAX_PPL_INCREASE_PCT,
                    help="max perplexity increase vs F16, in %% (default 5, or $MAX_PPL_INCREASE_PCT)")
    args = ap.parse_args()
    speed = load_json("speed.json")
    if not speed:
        sys.exit("results/speed.json not found. Run step4_benchmark.py first.")
    ppl = load_json("perplexity.json")
    if not ppl:
        print("Note: no perplexity.json yet - quality columns will be empty (run step4b_perplexity.py).")
    rows = build_rows(speed, ppl)

    out = results_dir() / "RESULTS.md"
    out.write_text(build_markdown(speed, rows, load_json("load_test.json"), args.budget,
                                        load_json("custom_quant.json"), ppl), encoding="utf-8")
    print(f"Wrote {out}")

    try:
        import report_plots

        for path in report_plots.make_all(rows, speed, results_dir()):
            print(f"Wrote {path}")
    except ImportError:
        print("matplotlib not installed - skipped plots (pip install matplotlib)")


if __name__ == "__main__":
    main()
