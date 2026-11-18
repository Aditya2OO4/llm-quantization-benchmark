"""
STEP 11: Does batching fix the serving bottleneck?
===================================================
The README says the FastAPI server answers one request at a time (a lock around one
llama.cpp context), so latency grows linearly with the number of clients, and that
request batching would fix it. This tests that claim by load-testing, one server at a time:

  fastapi       step5_api.py                (the existing server, one request at a time)
  llamaserver   llama.cpp's llama-server    (parallel slots + continuous batching)

Same model (Q4_K_M), same threads, same prompt, same load generator (step7_load_test.py),
24 requests per concurrency level so p95 means something.

Usage:
    python step11_serving_compare.py                        # both servers, 1/2/4/8 clients
    python step11_serving_compare.py --server llamaserver --slots 8
    python step11_serving_compare.py --levels 1,2,4 --requests 12    # quicker
    # control: same llama-server but ONE slot (no parallel decoding), to separate "batching" from "different server"
    python step11_serving_compare.py --server llamaserver --slots 1 --out load_test_llamaserver_np1.json

Results: results/load_test_fastapi_n24.json and results/load_test_llamaserver.json
(load_test.json from the original 8-request run is left untouched).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from config import ROOT, default_threads, find_llama_binary, format_prompt, model_path, results_dir
from step7_load_test import build_payload, run_level

OUT_FILES = {"fastapi": "load_test_fastapi_n24.json", "llamaserver": "load_test_llamaserver.json"}
CTX_PER_SLOT = 512  # the prompt is ~30 tokens and we generate 32, so this is plenty


def llamaserver_cmd(model: Path, port: int, threads: int, slots: int) -> list[str]:
    """Parallel slots + continuous batching. -c is the TOTAL context, shared by all slots."""
    return [str(find_llama_binary("llama-server")), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "-t", str(threads), "-np", str(slots), "-c", str(CTX_PER_SLOT * slots), "-cb", "--no-webui"]


def fastapi_cmd() -> list[str]:
    return [sys.executable, str(ROOT / "step5_api.py")]


def wait_until_ready(url: str, path: str, proc: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url + path, timeout=2) as resp:
                if resp.status == 200 and b"loading" not in resp.read().lower():
                    return
        except Exception:  # not up yet (connection refused, or llama-server answering 503 while loading)
            pass
        time.sleep(1)
    raise TimeoutError(f"{url}{path} not ready after {timeout:.0f}s")


@contextmanager
def running(cmd: list[str], env: dict | None = None):
    proc = subprocess.Popen(cmd, env={**os.environ, **(env or {})}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def load_test(kind: str, url: str, levels: list[int], requests: int, max_tokens: int, extra: dict) -> dict:
    payload = build_payload(kind, format_prompt("Explain what a CPU is."), max_tokens)
    rows = []
    print(f"{'Clients':>8}{'Req/s':>9}{'Tok/s':>9}{'p50 s':>9}{'p95 s':>9}")
    for c in levels:
        r = run_level(url, c, max(requests, c), payload, 600, kind)
        if r["tokens_per_s"] == 0:
            raise SystemExit("server returned 0 tokens; timings would be meaningless")
        rows.append(r)
        print(f"{c:>8}{r['req_per_s']:>9.2f}{r['tokens_per_s']:>9.1f}{r['p50_s']:>9.2f}{r['p95_s']:>9.2f}")
    return {"server": kind, "max_tokens": max_tokens, "requests_per_level": requests, **extra, "levels": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", choices=["fastapi", "llamaserver", "both"], default="both")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--slots", type=int, default=8, help="llama-server parallel slots (-np)")
    ap.add_argument("--threads", type=int, default=default_threads())
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--out", help="result file name (only with a single --server); default depends on the server")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    model = model_path("Q4_K_M")
    if not model.exists():
        sys.exit(f"{model} not found. Run step3_quantize_sweep.py first.")
    url = f"http://127.0.0.1:{args.port}"
    kinds = ["fastapi", "llamaserver"] if args.server == "both" else [args.server]
    if args.out and len(kinds) > 1:
        sys.exit("--out needs a single --server")

    for kind in kinds:  # one server at a time: RAM, and so they do not compete for the same 4 cores
        print(f"\n== {kind} ==")
        if kind == "fastapi":
            cmd, env, ready = fastapi_cmd(), {"PORT": str(args.port), "N_THREADS": str(args.threads),
                                              "MODEL_PATH": str(model)}, "/health"
            extra = {"threads": args.threads, "model": model.name}
        else:
            cmd, env, ready = llamaserver_cmd(model, args.port, args.threads, args.slots), None, "/health"
            extra = {"threads": args.threads, "model": model.name, "slots": args.slots, "continuous_batching": True}
        with running(cmd, env) as proc:
            wait_until_ready(url, ready, proc)
            result = load_test(kind, url, levels, args.requests, args.max_tokens, extra)
        out = results_dir() / (args.out or OUT_FILES[kind])
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved {out}")
    print("\nNext: python step6_report.py")


if __name__ == "__main__":
    main()
