"""
STEP 9: My own quantizer - weight error and perplexity
=======================================================
custom_quant.py implements block-wise symmetric absmax quantization (block 32, one fp16
scale per block, int8 and int4), similar in spirit to llama.cpp's Q8_0 / Q4_0 but written
from scratch. This script measures it:

  weights  per-tensor MSE and SQNR (dB) for every linear weight + embeddings + lm_head,
           and a plot by layer and tensor type
  gguf     the same weight error for llama.cpp's Q8_0 / Q4_0 tensors on identical inputs
  ppl      perplexity of the HF model with my fake-quantized weights, WikiText-2 test,
           same 40 x 512-token windows as step4b, F16 scored by the SAME evaluator

Results go to results/custom_quant.json (new file) and results/custom_quant_layers.png.

Usage:
    python step9_custom_quant.py weights
    python step9_custom_quant.py gguf
    python step9_custom_quant.py ppl                      # ~10-20 min per variant on 4 cores
    python step9_custom_quant.py ppl --variants fp16,int8 --chunks 10
    python step9_custom_quant.py all
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time

from bench_utils import PeakRSSSampler
from config import data_dir, default_threads, model_path, raw_model_dir, results_dir

OUT_NAME = "custom_quant.json"
CTX = 512


def load_results() -> dict:
    path = results_dir() / OUT_NAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _clean(v):
    """JSON has no infinity; 'no error at all' is stored as null."""
    if isinstance(v, float) and math.isinf(v):
        return None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_clean(x) for x in v]
    return v


def save_results(data: dict) -> None:
    path = results_dir() / OUT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(data), indent=1), encoding="utf-8")


def stage_weights(data: dict) -> None:
    import torch
    from safetensors import safe_open

    import custom_quant as cq
    from custom_quant_report import classify, mean_sqnr_by_kind

    src = raw_model_dir() / "model.safetensors"
    if not src.exists():
        sys.exit(f"{src} not found. Run step2_download_model.py first.")
    out: dict = {"int8": {}, "int4": {}}
    t0 = time.perf_counter()
    with safe_open(str(src), framework="pt") as f:
        for name in sorted(f.keys()):
            layer, kind = classify(name)
            if kind is None:
                continue  # norm weights are not quantized
            # start from fp16 weights, like llama.cpp does (the F16 GGUF)
            x = f.get_tensor(name).to(torch.float16).float().numpy()
            for key, bits in (("int8", 8), ("int4", 4)):
                out[key][name] = {**cq.quantization_error(x, bits), "layer": layer, "kind": kind}
            del x
            gc.collect()
    data["weights"] = out
    data["block"] = cq.BLOCK
    print(f"weights: {len(out['int8'])} tensors in {time.perf_counter() - t0:.0f}s")
    for key in ("int8", "int4"):
        means = mean_sqnr_by_kind(out[key])
        print(f"  {key} mean SQNR dB: " + ", ".join(f"{k} {v:.1f}" for k, v in means.items()))


def stage_gguf(data: dict) -> None:
    import gguf_compare

    f16 = model_path("F16")
    data["gguf_weight_compare"] = {}
    for key, bits, label in (("int8", 8, "Q8_0"), ("int4", 4, "Q4_0")):
        quant = model_path(label)
        if not (f16.exists() and quant.exists()):
            print(f"gguf: skipping {label} ({quant.name} missing)")
            continue
        res = gguf_compare.compare(f16, quant, bits)
        res["summary"] = gguf_compare.summarize(res)
        data["gguf_weight_compare"][key] = res
        s = res["summary"]
        print(f"gguf {label}: {len(res['tensors'])} tensors | llama.cpp SQNR {s['llama_cpp']['sqnr_db']:.2f} dB "
              f"| mine {s['mine']['sqnr_db']:.2f} dB | skipped (other types): {res['skipped_other_types']}")


def stage_ppl(data: dict, variants: list[str], chunks: int) -> None:
    import torch
    from transformers import AutoTokenizer

    import custom_quant_eval as ev

    torch.set_num_threads(default_threads())
    text_path = data_dir() / "wiki.test.raw"
    if not text_path.exists():
        import step4b_perplexity

        text_path = step4b_perplexity.ensure_dataset()
    tok = AutoTokenizer.from_pretrained(str(raw_model_dir()))
    ids = tok(text_path.read_text(encoding="utf-8"))["input_ids"]  # starts with BOS
    windows = ev.build_windows(ids, CTX, chunks, tok.bos_token_id)
    print(f"{len(ids)} tokens in file, scoring {len(windows)} windows x {CTX}, {default_threads()} threads")

    data.setdefault("ppl", {})
    for name in variants:
        print(f"[{name}] loading ...", flush=True)
        t0 = time.perf_counter()
        with PeakRSSSampler() as ram:
            model = ev.load_model(raw_model_dir(), name)
            load_s = time.perf_counter() - t0
            nll = ev.window_nll(model, windows)
        res = ev.summarize_nll(nll)
        res.update(evaluator="hf-transformers", n_ctx=CTX, chunks=len(windows), load_s=round(load_s, 1),
                   eval_s=round(time.perf_counter() - t0 - load_s, 1), peak_rss_mb=round(ram.peak(), 0))
        data["ppl"][name] = res
        save_results(data)  # save as we go
        print(f"[{name}] PPL = {res['ppl']:.4f} ± {res['ppl_err']:.4f}  "
              f"({res['eval_s']:.0f}s, peak {res['peak_rss_mb']:.0f} MB)")
        del model
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["weights", "gguf", "ppl", "plot", "all"])
    ap.add_argument("--variants", default="fp16,int8,int4,int4_head8", help="for the ppl stage")
    ap.add_argument("--chunks", type=int, default=40, help="512-token windows (same as step4b)")
    args = ap.parse_args()

    data = load_results()
    if args.stage in ("weights", "all"):
        stage_weights(data)
        save_results(data)
    if args.stage in ("gguf", "all"):
        stage_gguf(data)
        save_results(data)
    if args.stage in ("ppl", "all"):
        stage_ppl(data, args.variants.split(","), args.chunks)
    if data.get("weights"):
        import custom_quant_report

        path = custom_quant_report.plot_layers(data["weights"], results_dir() / "custom_quant_layers.png")
        print(f"Wrote {path}")
    save_results(data)
    print(f"\nSaved {results_dir() / OUT_NAME}\nNext: python step6_report.py")


if __name__ == "__main__":
    main()
