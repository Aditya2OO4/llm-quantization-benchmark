"""Shared paths, settings and small helpers used by every step.

Everything can be overridden with environment variables, so nothing is
hard-coded to one machine.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent

# The quantization levels we compare, from biggest/best to smallest/worst.
QUANT_LEVELS = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]
ALL_LABELS = ["F16", *QUANT_LEVELS]
REFERENCE_LABEL = "F16"  # everything is compared against this
# Nominal bits per weight (K-quants mix several widths, so this is the headline figure).
BITS = {"F16": 16, "Q8_0": 8, "Q6_K": 6, "Q5_K_M": 5, "Q4_K_M": 4, "Q3_K_M": 3, "Q2_K": 2}

HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Prompts used for the speed benchmark. "long" exists to stress prompt processing.
_PARAGRAPH = (
    "Quantization reduces the number of bits used to store each weight of a "
    "neural network. Fewer bits means a smaller file and less memory traffic, "
    "which usually makes inference faster on CPUs, but it also adds rounding "
    "error that can lower the quality of the model's answers. "
)
BENCH_PROMPTS = {
    "short": "Explain what a neural network is in simple terms.",
    "code": "Write a Python function that reverses a string.",
    "long": "Summarize the following text in two sentences.\n\n" + _PARAGRAPH * 6,
}
LONG_PROMPT_MIN_TOKENS = 100  # runs with at least this many prompt tokens count as "long"
MIN_DECODE_TOKENS = 8  # shorter answers are too noisy to compute tokens/sec from


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default))


def models_dir() -> Path:
    return _path_env("MODELS_DIR", ROOT / "models")


def results_dir() -> Path:
    return _path_env("RESULTS_DIR", ROOT / "results")


def data_dir() -> Path:
    return _path_env("DATA_DIR", ROOT / "data")


def llama_cpp_dir() -> Path:
    return _path_env("LLAMA_CPP_DIR", ROOT / "llama.cpp")


def raw_model_dir() -> Path:
    return models_dir() / "tinyllama-raw"


def model_path(label: str) -> Path:
    """'Q4_K_M' -> models/tinyllama-q4km.gguf  (matches the original naming)."""
    return models_dir() / f"tinyllama-{label.lower().replace('_', '')}.gguf"


def available_models(labels: list[str] | None = None) -> list[tuple[str, Path]]:
    """(label, path) for every requested model that exists on disk."""
    return [(lb, model_path(lb)) for lb in (labels or ALL_LABELS) if model_path(lb).exists()]


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024**2)


def default_threads() -> int:
    """N_THREADS env var, else the number of physical CPU cores."""
    if os.environ.get("N_THREADS"):
        return int(os.environ["N_THREADS"])
    return psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 2) // 2)


def find_llama_binary(name: str) -> Path:
    """Locate a llama.cpp tool such as 'llama-quantize' or 'llama-perplexity'.

    Search order: $LLAMA_BIN_DIR, the cmake build folders inside llama.cpp, PATH.
    """
    folders: list[Path] = []
    if os.environ.get("LLAMA_BIN_DIR"):
        folders.append(Path(os.environ["LLAMA_BIN_DIR"]))
    base = llama_cpp_dir()
    folders += [base / "build" / "bin" / "Release", base / "build" / "bin", base]
    for folder in folders:
        for candidate in (name, f"{name}.exe"):
            if (folder / candidate).is_file():
                return folder / candidate
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    raise FileNotFoundError(
        f"Could not find '{name}'. Build llama.cpp (see step3_quantize.md) or "
        "download a prebuilt release and set LLAMA_BIN_DIR to its folder."
    )


def format_prompt(user_text: str) -> str:
    """TinyLlama-Chat expects this chat layout; raw text gives much worse answers."""
    return f"<|user|>\n{user_text}</s>\n<|assistant|>\n"
