"""
STEP 4b: Quality benchmark - perplexity on WikiText-2
======================================================
Perplexity measures how "surprised" the model is by real text. Lower is better.
A quantized model that is badly damaged shows a much HIGHER perplexity than F16.

We use llama.cpp's own `llama-perplexity` tool on the WikiText-2 test set, the
standard dataset for this comparison. Results go to results/perplexity.json.

Usage:
    python step4b_perplexity.py                    # all models found
    python step4b_perplexity.py --chunks 20        # faster, noisier
    python step4b_perplexity.py --labels F16,Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from bench_utils import parse_perplexity
from config import ALL_LABELS, available_models, data_dir, default_threads, find_llama_binary, results_dir

CTX = 512  # tokens per evaluation window


def ensure_dataset() -> Path:
    """Download WikiText-2 (test split) once and save it as plain text."""
    out = data_dir() / "wiki.test.raw"
    if out.exists():
        return out
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print("Downloading WikiText-2 test set ...")
    parquet = hf_hub_download(
        repo_id="Salesforce/wikitext", repo_type="dataset",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
    )
    text = "".join(pq.read_table(parquet, columns=["text"]).column("text").to_pylist())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def run_perplexity(model: Path, text: Path, chunks: int, threads: int) -> tuple[float, float]:
    cmd = [
        str(find_llama_binary("llama-perplexity")),
        "-m", str(model), "-f", str(text),
        "-c", str(CTX), "--chunks", str(chunks), "-t", str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    parsed = parse_perplexity(proc.stdout + proc.stderr)
    if parsed is None:
        raise RuntimeError(f"no PPL found for {model.name}:\n{(proc.stderr or proc.stdout)[-1200:]}")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="comma list (default: all found)")
    ap.add_argument("--chunks", type=int, default=40, help="windows of 512 tokens to score")
    ap.add_argument("--threads", type=int, default=default_threads())
    args = ap.parse_args()

    models = available_models(args.labels.split(",") if args.labels else ALL_LABELS)
    if not models:
        sys.exit("No models found in ./models. Run steps 2 and 3 first.")
    text = ensure_dataset()

    out = results_dir() / "perplexity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    for label, path in models:
        print(f"[{label:>6}] perplexity ({args.chunks} chunks) ...", end=" ", flush=True)
        ppl, err = run_perplexity(path, text, args.chunks, args.threads)
        data[label] = {"ppl": ppl, "ppl_err": err, "chunks": args.chunks, "ctx": CTX}
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")  # save as we go
        print(f"PPL = {ppl:.3f} ± {err:.3f}")

    print(f"\nSaved {out}\nNext: python step6_report.py")


if __name__ == "__main__":
    main()
