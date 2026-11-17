"""
STEP 8: Live side-by-side comparison of every quantization level
=================================================================
Type one question; each model answers in its own column, token by token, with
live timing. Models run ONE AT A TIME (they don't all fit in 8 GB of RAM, and
running them in parallel would make the speed comparison unfair).

Usage:
    python step8_compare.py            # opens http://127.0.0.1:8001
    python step8_compare.py --no-browser

Stop step5_api.py first if it is running (both load a model).
Settings (env vars): COMPARE_HOST, COMPARE_PORT, N_THREADS.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import threading
import time
import webbrowser
from collections.abc import Callable, Iterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from compare_checks import check_answer, public_presets
from config import ALL_LABELS, BITS, ROOT, available_models, default_threads, file_size_mb, format_prompt

HOST = os.environ.get("COMPARE_HOST", "127.0.0.1")
PORT = int(os.environ.get("COMPARE_PORT", 8001))
N_CTX = 2048  # TinyLlama's maximum context

_run_lock = threading.Lock()  # one comparison at a time (RAM + fair timing)
app = FastAPI(title="Quantization side-by-side")


class CompareRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    labels: list[str] | None = Field(default=None, description="Which levels to run (default: all found)")
    max_tokens: int = Field(default=200, ge=1, le=512)
    preset_id: str | None = Field(default=None, description="Enables the automatic correctness check")


def load_model(path: Path, threads: int):
    from llama_cpp import Llama  # imported here so tests don't need it

    return Llama(model_path=str(path), n_threads=threads, n_ctx=N_CTX, seed=1, verbose=False)


def _unload(llm) -> None:
    close = getattr(llm, "close", None)
    if close:
        close()
    gc.collect()


def stream_model(label: str, path: Path, req: CompareRequest, loader: Callable) -> Iterator[dict]:
    """Load one model, stream its answer as events, then free its memory."""
    yield {"type": "model_start", "label": label}
    t0 = time.perf_counter()
    try:
        llm = loader(path, default_threads())
    except Exception as exc:
        yield {"type": "error", "label": label, "message": f"could not load model: {exc}"}
        return
    try:
        yield {"type": "loaded", "label": label, "load_s": time.perf_counter() - t0}
        start = time.perf_counter()
        first = last = None
        pieces: list[str] = []
        stream = llm(format_prompt(req.question), max_tokens=req.max_tokens, temperature=0.0,
                     stop=["</s>"], stream=True)
        for chunk in stream:
            choice = chunk["choices"][0]
            if choice["text"] == "" and choice.get("finish_reason"):
                continue  # trailing "finished" marker, not a token
            last = time.perf_counter()
            first = first or last
            pieces.append(choice["text"])
            yield {"type": "token", "label": label, "text": choice["text"]}
        n = len(pieces)
        yield {
            "type": "model_done", "label": label, "tokens": n,
            "ttft_s": first - start if first else None,
            "total_s": time.perf_counter() - start,
            "decode_tps": (n - 1) / (last - first) if n >= 2 and last > first else None,
            "check": check_answer(req.preset_id, "".join(pieces).strip()),
        }
    except Exception as exc:  # one bad model must not kill the whole run
        yield {"type": "error", "label": label, "message": str(exc)}
    finally:
        _unload(llm)  # also runs if the browser disconnects mid-answer


def run_compare(req: CompareRequest, loader: Callable) -> Iterator[dict]:
    if not _run_lock.acquire(blocking=False):
        yield {"type": "error", "label": None, "message": "Another comparison is already running."}
        return
    try:
        models = available_models(req.labels or ALL_LABELS)
        if not models:
            yield {"type": "error", "label": None, "message": "No model files found. Run step 3 first."}
            return
        info = [{"label": lb, "bits": BITS.get(lb), "size_mb": file_size_mb(p)} for lb, p in models]
        yield {"type": "run_start", "models": info}
        for label, path in models:
            yield from stream_model(label, path, req, loader)
        yield {"type": "run_done"}
    finally:
        _run_lock.release()


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(ROOT / "compare_ui.html")


@app.get("/api/models")
def models() -> list[dict]:
    return [{"label": lb, "bits": BITS.get(lb), "size_mb": file_size_mb(p)} for lb, p in available_models()]


@app.get("/api/presets")
def presets() -> list[dict]:
    return public_presets()


@app.post("/api/compare")
def compare(req: CompareRequest) -> StreamingResponse:
    """Streams newline-delimited JSON events (see the 'type' field)."""
    events = (json.dumps(ev) + "\n" for ev in run_compare(req, lambda p, t: load_model(p, t)))
    return StreamingResponse(events, media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://{HOST}:{PORT}"
    print(f"\nSide-by-side demo at {url}   (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run("step8_compare:app", host=HOST, port=PORT, reload=False)
