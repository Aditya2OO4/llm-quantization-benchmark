"""
STEP 7: Load-test the API (how does it behave with several users at once?)
===========================================================================
Start the server first (python step5_api.py), then in another terminal:

    python step7_load_test.py
    python step7_load_test.py --levels 1,2,4 --requests 6

Because one CPU model can only answer one request at a time, you should see
throughput stay flat while latency grows as more clients arrive. This is the
queueing behaviour that real serving systems (batching, vLLM) are built to fix.

To test a different server, e.g. llama.cpp's own llama-server with parallel slots and
continuous batching (step11_serving_compare.py automates this):

    python step7_load_test.py --format llamaserver --url http://127.0.0.1:8080 \
           --levels 1,2,4,8 --requests 24 --out load_test_llamaserver.json

Uses only the standard library. Results go to results/load_test.json by default.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from config import format_prompt, results_dir


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    rank = max(1, round(pct / 100 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


# How to talk to each kind of server: endpoint, request body, and where the token count is.
FORMATS = {
    "fastapi": {  # step5_api.py (default, the original behaviour)
        "endpoint": "/generate",
        "body": lambda prompt, n: {"prompt": prompt, "max_tokens": n, "temperature": 0.0},
        "tokens_key": "tokens_generated",
    },
    "llamaserver": {  # llama.cpp's llama-server. cache_prompt is off so no request gets a free prompt reuse.
        "endpoint": "/completion",
        "body": lambda prompt, n: {"prompt": prompt, "n_predict": n, "temperature": 0.0, "cache_prompt": False},
        "tokens_key": "tokens_predicted",
    },
}


def build_payload(fmt: str, prompt: str, max_tokens: int) -> dict:
    return FORMATS[fmt]["body"](prompt, max_tokens)


def one_request(url: str, payload: dict, timeout: float, fmt: str = "fastapi") -> dict:
    spec = FORMATS[fmt]
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{url}{spec['endpoint']}", body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return {"latency_s": time.perf_counter() - start, "tokens": data[spec["tokens_key"]]}


def run_level(url: str, concurrency: int, total: int, payload: dict, timeout: float, fmt: str = "fastapi") -> dict:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: one_request(url, payload, timeout, fmt), range(total)))
    wall = time.perf_counter() - start
    lat = sorted(r["latency_s"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": total,
        "req_per_s": total / wall,
        "tokens_per_s": sum(r["tokens"] for r in results) / wall,
        "p50_s": percentile(lat, 50),
        "p95_s": percentile(lat, 95),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--levels", default="1,2,4,8", help="concurrent client counts")
    ap.add_argument("--requests", type=int, default=8, help="requests per level")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--format", choices=sorted(FORMATS), default="fastapi", help="which kind of server")
    ap.add_argument("--out", default="load_test.json", help="file name inside the results folder")
    args = ap.parse_args()

    # chat model: raw text can yield 0 tokens, so apply the chat template ourselves
    payload = build_payload(args.format, format_prompt("Explain what a CPU is."), args.max_tokens)
    print(f"{'Clients':>8}{'Req/s':>9}{'Tok/s':>9}{'p50 s':>9}{'p95 s':>9}")
    levels = []
    for c in (int(x) for x in args.levels.split(",")):
        row = run_level(args.url, c, max(args.requests, c), payload, args.timeout, args.format)
        if row["tokens_per_s"] == 0:
            raise SystemExit("The server returned 0 generated tokens, so these timings are meaningless. Check the prompt.")
        levels.append(row)
        print(f"{c:>8}{row['req_per_s']:>9.2f}{row['tokens_per_s']:>9.1f}{row['p50_s']:>9.2f}{row['p95_s']:>9.2f}")

    out = results_dir() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"server": args.format, "max_tokens": args.max_tokens, "requests_per_level": args.requests}
    out.write_text(json.dumps({**meta, "levels": levels}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}   (re-run python step6_report.py to add it to RESULTS.md)")


if __name__ == "__main__":
    main()
