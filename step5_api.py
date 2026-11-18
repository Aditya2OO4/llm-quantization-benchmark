"""
STEP 5: FastAPI Inference Server
=================================
Serves a quantized TinyLlama model as a REST API.

Usage:
    python step5_api.py
    set MODEL_PATH=models\\tinyllama-q8_0.gguf   # (optional) serve another level

Then open: http://localhost:8000/docs  (interactive Swagger UI)

All settings come from environment variables: MODEL_PATH, N_THREADS, N_CTX,
HOST, PORT.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import default_threads, format_prompt, model_path

MODEL_PATH = Path(os.environ.get("MODEL_PATH", model_path("Q4_K_M")))
N_THREADS = default_threads()
N_CTX = int(os.environ.get("N_CTX", 1024))
HOST = os.environ.get("HOST", "127.0.0.1")  # set HOST=0.0.0.0 to expose on your network
PORT = int(os.environ.get("PORT", 8000))

llm = None  # the loaded Llama model (set at startup)
# A llama.cpp context is NOT thread-safe: requests take turns. The lock makes
# that explicit, and we report how long each request waited for its turn.
_infer_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release it at shutdown."""
    global llm
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Complete steps 2 and 3 first.")
    from llama_cpp import Llama  # imported here so tests can run without it

    print(f"Loading {MODEL_PATH} ...")
    llm = Llama(model_path=str(MODEL_PATH), n_threads=N_THREADS, n_ctx=N_CTX, verbose=False)
    print("Model ready.")
    yield
    llm = None


app = FastAPI(
    title="TinyLlama Quantized Inference API",
    description="CPU inference for a quantized TinyLlama 1.1B model via llama.cpp.",
    version="2.0.0",
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ..., min_length=1, max_length=4000,
        examples=["Explain what quantization does to a language model."],
        description="The text prompt to complete.",
    )
    max_tokens: int = Field(default=200, ge=1, le=512, description="Maximum tokens to generate.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="0 = deterministic.")
    stop: list[str] = Field(default=["</s>"], description="Stop generating when one of these appears.")
    chat: bool = Field(
        default=False,
        description="Wrap the prompt in TinyLlama-Chat's template (format_prompt). "
                    "Use this for questions; leave off for raw text completion.",
    )


class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    tokens_per_second: float = Field(description="Generated tokens / total inference time (includes prompt processing).")
    inference_ms: float = Field(description="Time spent running the model.")
    queue_wait_ms: float = Field(description="Time spent waiting for earlier requests to finish.")
    process_rss_mb: float = Field(description="Server process memory right after this request.")
    model: str


@app.post("/generate", response_model=GenerateResponse, summary="Generate text", tags=["Inference"])
def generate(req: GenerateRequest) -> GenerateResponse:
    # A plain `def` endpoint runs in FastAPI's thread pool, so a long generation
    # does not freeze the event loop (an `async def` here would).
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    arrived = time.perf_counter()
    with _infer_lock:
        started = time.perf_counter()
        try:
            prompt = format_prompt(req.prompt) if req.chat else req.prompt
            output = llm(prompt, max_tokens=req.max_tokens, temperature=req.temperature,
                         stop=req.stop, echo=False)
        except ValueError as exc:  # llama-cpp-python: prompt longer than the context window
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finished = time.perf_counter()

    n = output["usage"]["completion_tokens"]
    inference_s = finished - started
    return GenerateResponse(
        response=output["choices"][0]["text"].strip(),
        tokens_generated=n,
        tokens_per_second=round(n / inference_s, 2) if inference_s > 0 else 0.0,
        inference_ms=round(inference_s * 1000, 1),
        queue_wait_ms=round((started - arrived) * 1000, 1),
        process_rss_mb=round(psutil.Process().memory_info().rss / 1024**2, 1),
        model=MODEL_PATH.stem,
    )


@app.get("/health", summary="Health check", tags=["System"])
def health() -> dict:
    return {
        "status": "ok" if llm is not None else "model_not_loaded",
        "model_loaded": llm is not None,
        "model": MODEL_PATH.stem,
        "process_rss_mb": round(psutil.Process().memory_info().rss / 1024**2, 1),
        "threads": N_THREADS,
    }


@app.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse({"message": "Quantized TinyLlama API", "docs": "/docs", "health": "/health"})


if __name__ == "__main__":
    import uvicorn

    print(f"\nServing {MODEL_PATH.name} at http://{HOST}:{PORT}/docs\n")
    uvicorn.run("step5_api:app", host=HOST, port=PORT, reload=False)
