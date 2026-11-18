"""
STEP 10: Importance-matrix experiment
======================================
Hypothesis: llama.cpp's importance matrix (imatrix) recovers a large share of the quality
lost at low bit widths (Q2_K, Q3_K_M, maybe Q4_K_M).

How it is tested:
  1. Calibration text comes from the WikiText-2 TRAIN split (never the test split). A
     leakage check refuses to run if any test paragraph appears in the calibration text.
  2. llama-imatrix runs the F16 model over N chunks of 512 tokens and records, for every
     weight column, how large its activations are.
  3. llama-quantize --imatrix quantizes with those weights; without it all columns count equally.
  4. Perplexity of both files is measured on the TEST windows with exactly step4b's settings
     (40 x 512 tokens). The no-imatrix numbers are the ones already in results/perplexity.json.
  5. gap closed = (ppl_plain - ppl_imatrix) / (ppl_plain - ppl_F16)

Results go to results/imatrix.json. Re-running skips finished entries (use --force to redo).

Usage:
    python step10_imatrix.py                              # Q2_K, Q3_K_M, Q4_K_M with 50 chunks
    python step10_imatrix.py --levels Q2_K --chunks-list 10,50,200   # calibration-size ablation
    python step10_imatrix.py --force
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from config import (
    REFERENCE_LABEL,
    data_dir,
    default_threads,
    file_size_mb,
    find_llama_binary,
    imatrix_dir,
    imatrix_model_path,
    model_path,
    results_dir,
)

OUT_NAME = "imatrix.json"
CALIB_CHARS = 1_000_000  # ~220k tokens, more than the largest ablation (200 x 512)
DEFAULT_LEVELS = "Q2_K,Q3_K_M,Q4_K_M"
MIN_PARAGRAPH = 100  # shorter lines (headings, blanks) are not a meaningful leakage signal


def gap_closed_pct(ppl_f16: float, ppl_plain: float, ppl_imatrix: float) -> float | None:
    """Share of the perplexity lost to quantization that the imatrix wins back (100 = back to F16)."""
    gap = ppl_plain - ppl_f16
    return None if gap <= 0 else (ppl_plain - ppl_imatrix) / gap * 100


def leaked_paragraphs(calib: str, test: str, min_len: int = MIN_PARAGRAPH) -> list[str]:
    """Test-set paragraphs that also occur verbatim in the calibration text (should be none)."""
    return [p for p in (line.strip() for line in test.splitlines()) if len(p) >= min_len and p in calib]


def imatrix_cmd(model: Path, text: Path, out: Path, chunks: int, threads: int) -> list[str]:
    return [str(find_llama_binary("llama-imatrix")), "-m", str(model), "-f", str(text), "-o", str(out),
            "-c", "512", "--chunks", str(chunks), "-t", str(threads), "--no-ppl"]


def quantize_cmd(f16: Path, out: Path, level: str, imatrix: Path) -> list[str]:
    return [str(find_llama_binary("llama-quantize")), "--imatrix", str(imatrix), str(f16), str(out), level]


def ensure_calibration_text() -> Path:
    """First ~1 MB of the WikiText-2 TRAIN split, saved as plain text."""
    out = data_dir() / "wiki.train.calib.raw"
    if out.exists():
        return out
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    print("Downloading WikiText-2 train split ...")
    parquet = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                              filename="wikitext-2-raw-v1/train-00000-of-00001.parquet")
    text = "".join(pq.read_table(parquet, columns=["text"]).column("text").to_pylist())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text[:CALIB_CHARS], encoding="utf-8")
    return out


def load_results() -> dict:
    path = results_dir() / OUT_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_results(data: dict) -> None:
    path = results_dir() / OUT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def build_imatrix(chunks: int, calib: Path, threads: int, force: bool) -> tuple[Path, float | None]:
    out = imatrix_dir() / f"imatrix-{chunks}.gguf"
    if out.exists() and not force:
        return out, None
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[imatrix] {chunks} chunks of calibration text ...", flush=True)
    t0 = time.perf_counter()
    subprocess.run(imatrix_cmd(model_path(REFERENCE_LABEL), calib, out, chunks, threads),
                   check=True, capture_output=True)
    secs = time.perf_counter() - t0
    print(f"[imatrix] done in {secs:.0f}s")
    return out, secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=DEFAULT_LEVELS)
    ap.add_argument("--chunks-list", default="50", help="calibration sizes in 512-token chunks, e.g. 10,50,200")
    ap.add_argument("--eval-chunks", type=int, default=40, help="test windows (same as step4b)")
    ap.add_argument("--threads", type=int, default=default_threads())
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-models", action="store_true", help="keep the imatrix-quantized .gguf files")
    args = ap.parse_args()

    import step4b_perplexity as ppl_tool  # same evaluator as the main sweep

    f16 = model_path(REFERENCE_LABEL)
    base_ppl = json.loads((results_dir() / "perplexity.json").read_text(encoding="utf-8"))
    if REFERENCE_LABEL not in base_ppl:
        sys.exit("results/perplexity.json has no F16 entry. Run step4b_perplexity.py first.")
    test_text = ppl_tool.ensure_dataset()
    calib = ensure_calibration_text()
    leaks = leaked_paragraphs(calib.read_text(encoding="utf-8"), test_text.read_text(encoding="utf-8"))
    if leaks:
        sys.exit(f"Calibration text shares {len(leaks)} paragraph(s) with the test set. Refusing to run.")

    data = load_results()
    data["calibration"] = {"source": "WikiText-2 train split (first 1,000,000 chars)", "file": calib.name,
                           "test_paragraphs_found_in_calibration": 0, "ctx": 512}
    data["eval"] = {"text": test_text.name, "chunks": args.eval_chunks, "ctx": ppl_tool.CTX}
    data["f16"] = base_ppl[REFERENCE_LABEL]
    runs = data.setdefault("runs", {})

    for n in (int(c) for c in args.chunks_list.split(",")):
        imat, imat_secs = build_imatrix(n, calib, args.threads, args.force)
        for level in (lv.strip() for lv in args.levels.split(",")):
            entry = runs.setdefault(level, {"plain": base_ppl.get(level), "imatrix": {}})
            if str(n) in entry["imatrix"] and not args.force:
                print(f"[{level}] {n} chunks: already done, skipping")
                continue
            out = imatrix_model_path(level, n)
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            subprocess.run(quantize_cmd(f16, out, level, imat), check=True, capture_output=True)
            quant_s = time.perf_counter() - t0
            print(f"[{level}] {n} chunks: perplexity ...", end=" ", flush=True)
            ppl, err = ppl_tool.run_perplexity(out, test_text, args.eval_chunks, args.threads)
            plain = entry["plain"]
            closed = gap_closed_pct(data["f16"]["ppl"], plain["ppl"], ppl) if plain else None
            entry["imatrix"][str(n)] = {"ppl": ppl, "ppl_err": err, "size_mb": file_size_mb(out),
                                        "quantize_s": round(quant_s, 1), "imatrix_s": imat_secs,
                                        "gap_closed_pct": closed}
            save_results(data)
            print(f"PPL = {ppl:.3f} ± {err:.3f}" + (f"  (closes {closed:.0f}% of the gap to F16)" if closed is not None else ""))
            if not args.keep_models:
                out.unlink()  # only needed for the measurement

    print(f"\nSaved {results_dir() / OUT_NAME}\nNext: python step6_report.py")


if __name__ == "__main__":
    main()
