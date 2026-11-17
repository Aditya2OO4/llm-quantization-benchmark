import json

import pytest

import step4_benchmark
import step6_report
from step7_load_test import percentile


def _config(label, file_mb, decode, kind="main", threads=4):
    runs = [{"prompt_id": "short", "text": f"{label} answer | with pipe\nnewline"}]
    stat = lambda v: {"mean": v, "std": 0.5, "n": 3}  # noqa: E731
    return {
        "kind": kind, "label": label, "threads": threads, "file_mb": file_mb, "load_s": 1.0,
        "peak_rss_mb": file_mb * 1.2, "rss_after_load_mb": file_mb,
        "summary": {"decode_tps": stat(decode), "ttft_s": stat(0.3), "prompt_tps": stat(200.0)},
        "runs": runs,
    }


@pytest.fixture()
def speed():
    return {
        "hardware": {"cpu": "TestCPU", "physical_cores": 4, "logical_cores": 8, "ram_gb": 8, "os": "TestOS"},
        "settings": {"threads": 4, "repeats": 3, "max_tokens": 128},
        "configs": [
            _config("Q4_K_M", 640, 12.0),
            _config("F16", 2100, 4.0),
            _config("Q2_K", 480, 15.0),
            _config("Q4_K_M", 640, 6.0, kind="thread_sweep", threads=2),
            _config("Q4_K_M", 640, 11.0, kind="thread_sweep", threads=4),
        ],
    }


PPL = {"F16": {"ppl": 7.0, "ppl_err": 0.2}, "Q4_K_M": {"ppl": 7.14, "ppl_err": 0.2}, "Q2_K": {"ppl": 11.0, "ppl_err": 0.3}}


def test_rows_are_sorted_and_compared_to_f16(speed):
    rows = step6_report.build_rows(speed, PPL)
    assert [r["label"] for r in rows] == ["F16", "Q4_K_M", "Q2_K"]  # thread-sweep rows excluded
    q4 = rows[1]
    assert q4["size_pct"] == pytest.approx(640 / 2100 * 100)
    assert q4["speedup"] == pytest.approx(3.0)
    assert q4["ppl_delta_pct"] == pytest.approx(2.0)


def test_recommend_picks_smallest_within_quality_budget(speed):
    rows = step6_report.build_rows(speed, PPL)
    assert step6_report.recommend(rows)["label"] == "Q4_K_M"  # Q2_K is smaller but +57% ppl


def test_report_works_without_perplexity(speed):
    rows = step6_report.build_rows(speed, None)
    md = step6_report.build_markdown(speed, rows, None)
    assert "| Q4_K_M |" in md and "Recommendation" not in md


def test_markdown_contains_headline_samples_and_thread_scaling(speed):
    rows = step6_report.build_rows(speed, PPL)
    md = step6_report.build_markdown(speed, rows, {"levels": [
        {"concurrency": 2, "req_per_s": 0.5, "tokens_per_s": 16.0, "p50_s": 2.0, "p95_s": 3.1}]})
    assert "70% smaller than F16, 3.0x faster decoding, +2.0% perplexity" in md
    assert "Q4_K_M answer / with pipe newline" in md  # pipe and newline escaped for the table
    assert "Thread scaling" in md and "API load test" in md


def test_plots_are_written(speed, tmp_path):
    pytest.importorskip("matplotlib")
    import report_plots

    rows = step6_report.build_rows(speed, PPL)
    names = {p.name for p in report_plots.make_all(rows, speed, tmp_path)}
    assert names == {"size_vs_quality.png", "speed_vs_quality.png", "thread_scaling.png"}


def test_merge_keeps_old_entries_and_replaces_matching(monkeypatch, tmp_path, speed):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    step4_benchmark.merge_and_save(speed["configs"], speed["settings"])
    new = _config("Q4_K_M", 640, 99.0)
    out = step4_benchmark.merge_and_save([new], speed["settings"])
    configs = json.loads(out.read_text())["configs"]
    assert len(configs) == 5
    main_q4 = next(c for c in configs if c["kind"] == "main" and c["label"] == "Q4_K_M")
    assert main_q4["summary"]["decode_tps"]["mean"] == 99.0


def test_percentile_nearest_rank():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert percentile(data, 50) == 5.0
    assert percentile(data, 95) == 10.0
    assert percentile([], 50) == 0.0
