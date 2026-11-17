import time

import pytest

import config
from bench_utils import PeakRSSSampler, aggregate_runs, parse_perplexity, summarize

PPL_OUTPUT = """
perplexity: calculating perplexity over 40 chunks, batch_size=512
[1]7.1234,[2]7.5,[40]7.9012,
Final estimate: PPL = 7.9012 +/- 0.31245
llama_perf_context_print: load time = 400.00 ms
"""


def test_summarize_mean_std_and_missing_values():
    s = summarize([2.0, 4.0, None, 6.0])
    assert s["n"] == 3 and s["mean"] == pytest.approx(4.0) and s["std"] == pytest.approx(2.0)
    assert summarize([None]) == {"mean": None, "std": None, "n": 0}
    assert summarize([5.0])["std"] == 0.0


def test_parse_perplexity():
    assert parse_perplexity(PPL_OUTPUT) == (7.9012, 0.31245)
    assert parse_perplexity("nothing useful here") is None


def test_aggregate_runs_only_counts_long_prompts_for_prompt_speed():
    runs = [
        {"prompt_tokens": 20, "decode_tps": 10.0, "ttft_s": 0.2, "prompt_tps": 100.0},
        {"prompt_tokens": 400, "decode_tps": 12.0, "ttft_s": 1.0, "prompt_tps": 400.0},
    ]
    agg = aggregate_runs(runs)
    assert agg["decode_tps"]["mean"] == pytest.approx(11.0)
    assert agg["prompt_tps"]["mean"] == pytest.approx(400.0) and agg["prompt_tps"]["n"] == 1


def test_peak_rss_sampler_sees_allocation():
    with PeakRSSSampler(interval=0.005) as sampler:
        start = sampler.peak_mb
        blob = bytearray(80 * 1024 * 1024)
        blob[::4096] = b"x" * len(blob[::4096])  # touch pages so they become resident
        time.sleep(0.05)
        peak = sampler.peak()
    assert peak >= start + 50
    del blob


def test_model_path_naming_matches_original_files(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    assert config.model_path("Q4_K_M") == tmp_path / "tinyllama-q4km.gguf"
    assert config.model_path("F16") == tmp_path / "tinyllama-f16.gguf"
    assert config.model_path("Q8_0") == tmp_path / "tinyllama-q80.gguf"


def test_available_models_only_lists_existing_files(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    (tmp_path / "tinyllama-q4km.gguf").write_bytes(b"x")
    (tmp_path / "tinyllama-f16.gguf").write_bytes(b"x")
    assert [lb for lb, _ in config.available_models()] == ["F16", "Q4_K_M"]


def test_find_llama_binary_uses_env_dir_and_errors_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "nope"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("LLAMA_BIN_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="LLAMA_BIN_DIR"):
        config.find_llama_binary("llama-quantize")

    (tmp_path / "llama-quantize.exe").write_bytes(b"")
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    assert config.find_llama_binary("llama-quantize") == tmp_path / "llama-quantize.exe"


def test_format_prompt_uses_tinyllama_chat_layout():
    assert config.format_prompt("hi") == "<|user|>\nhi</s>\n<|assistant|>\n"
