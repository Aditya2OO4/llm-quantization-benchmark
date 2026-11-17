"""
STEP 7: Load-test the API (how does it behave with several users at once?)
===========================================================================
Start the server first (python step5_api.py), then in another terminal:

    python step7_load_test.py
    python step7_load_test.py --levels 1,2,4 --requests 6

Because one CPU model can only answer one request at a time, you should see
throughput stay flat while latency grows as more clients arrive. This is the
queueing behaviour that real serving systems (batching, vLLM) are built to fix.

Uses only the standard library. Results go to results/load_test.json.
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


def one_request(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{url}/generate", body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return {"latency_s": time.perf_counter() - start, "tokens": data["tokens_generated"]}


def run_level(url: str, concurrency: int, total: int, payload: dict, timeout: float) -> dict:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: one_request(url, payload, timeout), range(total)))
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
    args = ap.parse_args()

    payload = {
        "prompt": format_prompt("Explain what a CPU is."),  # chat model: raw text can yield 0 tokens
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    print(f"{'Clients':>8}{'Req/s':>9}{'Tok/s':>9}{'p50 s':>9}{'p95 s':>9}")
    levels = []
    for c in (int(x) for x in args.levels.split(",")):
        row = run_level(args.url, c, max(args.requests, c), payload, args.timeout)
        if row["tokens_per_s"] == 0:
            raise SystemExit("The server returned 0 generated tokens, so these timings are meaningless. Check the prompt.")
        levels.append(row)
        print(f"{c:>8}{row['req_per_s']:>9.2f}{row['tokens_per_s']:>9.1f}{row['p50_s']:>9.2f}{row['p95_s']:>9.2f}")

    out = results_dir() / "load_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"levels": levels}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}   (re-run python step6_report.py to add it to RESULTS.md)")


if __name__ == "__main__":
    main()
