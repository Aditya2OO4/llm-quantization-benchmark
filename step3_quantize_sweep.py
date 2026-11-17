"""
STEP 3: Convert to GGUF and quantize at every level
====================================================
  1. raw safetensors  -> F16 GGUF         (llama.cpp's convert_hf_to_gguf.py)
  2. F16 GGUF         -> Q8_0 ... Q2_K    (llama.cpp's llama-quantize)

Already-built files are skipped, so it is safe to re-run.

Usage:
    python step3_quantize_sweep.py                  # all levels
    python step3_quantize_sweep.py --levels Q4_K_M  # just one
    python step3_quantize_sweep.py --force          # rebuild everything
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from config import (
    QUANT_LEVELS,
    file_size_mb,
    find_llama_binary,
    llama_cpp_dir,
    model_path,
    raw_model_dir,
)


def convert_to_f16(force: bool) -> None:
    out = model_path("F16")
    if out.exists() and not force:
        print(f"[F16] already exists ({file_size_mb(out):.0f} MB), skipping")
        return
    script = llama_cpp_dir() / "convert_hf_to_gguf.py"
    if not script.exists():
        sys.exit(f"{script} not found. Clone llama.cpp first (see step3_quantize.md).")
    if not raw_model_dir().exists():
        sys.exit(f"{raw_model_dir()} not found. Run step2_download_model.py first.")
    print("[F16] converting safetensors -> GGUF ...")
    cmd = [sys.executable, str(script), str(raw_model_dir()), "--outtype", "f16", "--outfile", str(out)]
    subprocess.run(cmd, check=True)


def quantize(level: str, force: bool) -> None:
    out = model_path(level)
    if out.exists() and not force:
        print(f"[{level}] already exists ({file_size_mb(out):.0f} MB), skipping")
        return
    print(f"[{level}] quantizing ...", end=" ", flush=True)
    t0 = time.perf_counter()
    subprocess.run(
        [str(find_llama_binary("llama-quantize")), str(model_path("F16")), str(out), level],
        check=True, capture_output=True,
    )
    print(f"done in {time.perf_counter() - t0:.0f}s -> {file_size_mb(out):.0f} MB")


def print_size_table() -> None:
    base = file_size_mb(model_path("F16"))
    print(f"\n{'Level':<8}{'Size MB':>10}{'vs F16':>10}")
    for label in ["F16", *QUANT_LEVELS]:
        if model_path(label).exists():
            size = file_size_mb(model_path(label))
            print(f"{label:<8}{size:>10.0f}{size / base:>9.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default=",".join(QUANT_LEVELS), help="comma list of quant levels")
    ap.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = ap.parse_args()

    model_path("F16").parent.mkdir(parents=True, exist_ok=True)
    convert_to_f16(args.force)
    for level in args.levels.split(","):
        quantize(level.strip(), args.force)
    print_size_table()
    print("\nNext: python step4_benchmark.py")


if __name__ == "__main__":
    main()
