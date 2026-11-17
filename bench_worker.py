"""Child process for the speed benchmark: loads ONE model and measures it.

step4_benchmark.py launches this once per (model, thread-count) so every model
starts from a clean process, so a previous model can't leave memory behind
and skew the RAM numbers.

Prints one line:  RESULT_JSON:{...}
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from bench_utils import PeakRSSSampler, rss_mb
from config import BENCH_PROMPTS, MIN_DECODE_TOKENS, format_prompt

RESULT_PREFIX = "RESULT_JSON:"


def _clear_cache(llm) -> None:
    """Forget the previous prompt, otherwise repeats would skip prompt processing."""
    llm.reset()
    try:
        llm._ctx.kv_cache_clear()
    except Exception:  # private API; not present in every version
        pass


def measure_once(llm, prompt_id: str, user_text: str, max_tokens: int) -> dict:
    prompt = format_prompt(user_text)
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    _clear_cache(llm)

    start = time.perf_counter()
    first = None
    pieces: list[str] = []
    for chunk in llm(prompt, max_tokens=max_tokens, temperature=0.0, stop=["</s>"], stream=True):
        choice = chunk["choices"][0]
        if choice["text"] == "" and choice.get("finish_reason"):
            continue  # trailing "finished" marker, not a generated token
        if first is None:
            first = time.perf_counter()  # prompt is now fully processed
        pieces.append(choice["text"])
    end = time.perf_counter()

    n = len(pieces)
    ttft = (first - start) if first else None
    decode_ok = first is not None and n >= MIN_DECODE_TOKENS and end > first
    return {
        "prompt_id": prompt_id,
        "prompt_tokens": prompt_tokens,
        "tokens_generated": n,
        "ttft_s": ttft,
        "prompt_tps": prompt_tokens / ttft if ttft else None,
        "decode_tps": (n - 1) / (end - first) if decode_ok else None,
        "text": "".join(pieces).strip(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--ctx", type=int, default=1024)
    args = ap.parse_args()

    from llama_cpp import Llama  # imported here so tests don't need it installed

    baseline_mb = rss_mb()
    with PeakRSSSampler() as sampler:
        t0 = time.perf_counter()
        llm = Llama(model_path=args.model, n_threads=args.threads, n_ctx=args.ctx, seed=1, verbose=False)
        load_s = time.perf_counter() - t0
        # Warm-up (not recorded): a realistic prompt, so weights are paged in and the CPU is at speed.
        for _ in range(2):
            measure_once(llm, "warmup", BENCH_PROMPTS["code"], 16)
        after_load_mb = rss_mb()

        runs = []
        for prompt_id, text in BENCH_PROMPTS.items():
            for _ in range(args.repeats):
                runs.append(measure_once(llm, prompt_id, text, args.max_tokens))
        peak_mb = sampler.peak()

    result = {
        "load_s": load_s,
        "baseline_rss_mb": baseline_mb,  # python + libraries, before the model is loaded
        "rss_after_load_mb": after_load_mb,
        "peak_rss_mb": peak_mb,
        "runs": runs,
    }
    print(RESULT_PREFIX + json.dumps(result))


if __name__ == "__main__":
    sys.exit(main())
