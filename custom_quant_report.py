"""Plots and markdown tables for results/custom_quant.json (kept out of step9 and step6)."""

from __future__ import annotations

import re
from pathlib import Path

from config import REFERENCE_LABEL

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.weight")
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]


def classify(name: str) -> tuple[int | None, str | None]:
    """HF tensor name -> (layer index or None, tensor kind or None for non-linear tensors)."""
    m = _LAYER_RE.fullmatch(name)
    if m:
        return int(m.group(1)), m.group(2)
    if name == "model.embed_tokens.weight":
        return None, "embed_tokens"
    if name == "lm_head.weight":
        return None, "lm_head"
    return None, None


def mean_sqnr_by_kind(tensors: dict) -> dict[str, float]:
    """Average SQNR (dB) per tensor kind across layers."""
    groups: dict[str, list[float]] = {}
    for t in tensors.values():
        if t.get("sqnr_db") is not None:
            groups.setdefault(t["kind"], []).append(t["sqnr_db"])
    return {k: sum(v) / len(v) for k, v in groups.items()}


def plot_layers(weights: dict, path: Path) -> Path | None:
    """SQNR by layer index (one line per tensor type) for int8 and int4, plus a per-type bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not weights.get("int8") or not weights.get("int4"):
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    layer_kinds = KINDS[:7]
    for ax, key in zip(axes[:2], ("int8", "int4")):
        for kind in layer_kinds:
            pts = sorted((t["layer"], t["sqnr_db"]) for t in weights[key].values() if t["kind"] == kind)
            ax.plot(*zip(*pts), marker="o", ms=3, label=kind)
        for kind, style in (("embed_tokens", "--"), ("lm_head", ":")):
            ys = [t["sqnr_db"] for t in weights[key].values() if t["kind"] == kind]
            if ys:
                ax.axhline(ys[0], color="gray", ls=style, lw=1, label=kind)
        ax.set_title(f"{key}: SQNR by layer")
        ax.set_xlabel("layer index")
        ax.set_ylabel("SQNR (dB, higher = less error)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    width = 0.4
    means = {k: mean_sqnr_by_kind(weights[k]) for k in ("int8", "int4")}
    kinds = [k for k in KINDS if k in means["int8"]]
    xs = range(len(kinds))
    axes[2].bar([x - width / 2 for x in xs], [means["int8"][k] for k in kinds], width, label="int8")
    axes[2].bar([x + width / 2 for x in xs], [means["int4"][k] for k in kinds], width, label="int4")
    axes[2].set_xticks(list(xs), kinds, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("mean SQNR by tensor type")
    axes[2].set_ylabel("SQNR (dB)")
    axes[2].set_ylim(0, max(means["int8"].values()) * 1.2)  # headroom so the legend covers no bar
    axes[2].legend(loc="upper right", ncol=2)
    axes[2].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _fmt(v, spec=".2f"):
    return "-" if v is None else format(v, spec)


def ppl_table(custom: dict, llama_ppl: dict | None) -> list[str]:
    """Each tool's perplexity and its delta vs ITS OWN F16. Absolute values are not comparable across tools."""
    from custom_quant_eval import paired_delta

    mine = custom.get("ppl", {})
    lines = ["| Tool | Level | Bits/weight (incl. scales) | Perplexity (own evaluator) | Δ vs own F16 | ± paired SE |",
             "|---|---|---|---|---|---|"]
    ref = mine.get("fp16")
    bpw = {"fp16": "16", "int8": "8.5", "int4": "4.5", "int4_head8": "~4.5 (lm_head 8.5)"}
    for name in ("fp16", "int8", "int4", "int4_head8"):
        r = mine.get(name)
        if not r:
            continue
        if ref and name != "fp16":
            d = paired_delta(r["window_mean_nll"], ref["window_mean_nll"])
            delta, se = f"{d['delta_pct']:+.2f}%", f"{d['se_paired_pct']:.2f}%"
        else:
            delta, se = "-", "-"
        lines.append(f"| mine (HF transformers) | {name} | {bpw[name]} | {r['ppl']:.3f} | {delta} | {se} |")
    if llama_ppl and REFERENCE_LABEL in llama_ppl:
        base = llama_ppl[REFERENCE_LABEL]["ppl"]
        for label, bits in (("F16", "16"), ("Q8_0", "8.5"), ("Q4_0", "4.5 (output.weight Q6_K)")):
            r = llama_ppl.get(label)
            if r:
                delta = "-" if label == REFERENCE_LABEL else f"{(r['ppl'] / base - 1) * 100:+.2f}%"
                lines.append(f"| llama.cpp | {label} | {bits} | {r['ppl']:.3f} | {delta} | n/a (stored ± is unpaired) |")
    return lines


def weight_error_table(custom: dict) -> list[str]:
    cmp = custom.get("gguf_weight_compare")
    if not cmp:
        return []
    lines = ["| Format | Tensors compared | llama.cpp SQNR (dB) | mine SQNR (dB) |", "|---|---|---|---|"]
    for key in ("int8", "int4"):
        c = cmp.get(key)
        if c:
            s = c["summary"]
            lines.append(f"| {c['type']} vs my {key} | {len(c['tensors'])} | {_fmt(s['llama_cpp']['sqnr_db'])} "
                         f"| {_fmt(s['mine']['sqnr_db'])} |")
    return lines


def report_lines(custom: dict | None, llama_ppl: dict | None) -> list[str]:
    if not custom:
        return []
    out = ["", "## My quantizer vs llama.cpp (Q8_0 / Q4_0)", ""]
    if custom.get("ppl"):
        out += ["Each tool is compared with **its own** F16 baseline. Absolute perplexities differ between the two "
                "tools (different tokenizer and eval code), so only the deltas are comparable.", "", *ppl_table(custom, llama_ppl)]
    errs = weight_error_table(custom)
    if errs:
        out += ["", "Weight error on identical tensors (reference = the F16 GGUF):", "", *errs]
    if custom.get("weights"):
        out += ["", "![SQNR by layer and tensor type](custom_quant_layers.png)"]
    return out
