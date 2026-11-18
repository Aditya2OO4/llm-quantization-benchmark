import json
from pathlib import Path

import pytest

import config
import imatrix_report
import step6_report
import step10_imatrix as st


def test_gap_closed_pct():
    assert st.gap_closed_pct(10.0, 14.0, 12.0) == pytest.approx(50.0)
    assert st.gap_closed_pct(10.0, 14.0, 10.0) == pytest.approx(100.0)
    assert st.gap_closed_pct(10.0, 14.0, 15.0) == pytest.approx(-25.0)  # imatrix made it worse: reported, not hidden
    assert st.gap_closed_pct(10.0, 10.0, 10.0) is None  # no gap to close


def test_leak_check_finds_shared_paragraphs_and_ignores_short_lines():
    shared = "A" * 150
    calib = f"intro\n{shared}\nother text"
    test = f" = Heading = \n{shared}\n{'B' * 150}\nshort"
    assert st.leaked_paragraphs(calib, test) == [shared]
    assert st.leaked_paragraphs(calib, "short\n= x =\n" + "C" * 150) == []
    assert st.leaked_paragraphs("anything", "tiny") == []  # lines under 100 chars are ignored


def test_commands(monkeypatch, tmp_path):
    for name in ("llama-imatrix", "llama-quantize"):
        (tmp_path / f"{name}.exe").write_text("")
    monkeypatch.setenv("LLAMA_BIN_DIR", str(tmp_path))
    cmd = st.imatrix_cmd(Path("f16.gguf"), Path("calib.txt"), Path("out.gguf"), 50, 4)
    assert cmd[0].endswith("llama-imatrix.exe") and "--no-ppl" in cmd
    assert cmd[cmd.index("--chunks") + 1] == "50" and cmd[cmd.index("-f") + 1] == "calib.txt"
    q = st.quantize_cmd(Path("f16.gguf"), Path("q.gguf"), "Q2_K", Path("im.gguf"))
    assert q[1:] == ["--imatrix", "im.gguf", "f16.gguf", "q.gguf", "Q2_K"]


def test_imatrix_paths_do_not_collide_with_main_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    p = config.imatrix_model_path("Q2_K", 50)
    assert p == tmp_path / "imatrix" / "tinyllama-q2k-imat50.gguf"
    assert p != config.model_path("Q2_K")


DATA = {
    "calibration": {"source": "WikiText-2 train split"}, "eval": {"chunks": 40, "ctx": 512}, "f16": {"ppl": 10.0},
    "runs": {"Q2_K": {"plain": {"ppl": 14.0, "ppl_err": 0.3},
                      "imatrix": {"200": {"ppl": 11.0, "ppl_err": 0.2, "gap_closed_pct": 75.0},
                                  "50": {"ppl": 12.0, "ppl_err": 0.2, "gap_closed_pct": 50.0}}}},
}


def test_report_lists_each_calibration_size_in_order():
    md = "\n".join(imatrix_report.report_lines(DATA))
    assert md.index("| Q2_K | 50 |") < md.index("| Q2_K | 200 |")
    assert "+40.0% → +20.0%" in md and "50%" in md and "75%" in md
    assert imatrix_report.report_lines(None) == [] and imatrix_report.report_lines({"runs": {}}) == []


def test_results_roundtrip_and_step6_includes_section(monkeypatch, tmp_path):
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path))
    st.save_results(DATA)
    assert st.load_results() == json.loads(json.dumps(DATA))
    speed = {"hardware": {"cpu": "T", "physical_cores": 4, "logical_cores": 8, "ram_gb": 8, "os": "T"},
             "settings": {"threads": 4, "repeats": 3, "max_tokens": 128}, "configs": []}
    md = step6_report.build_markdown(speed, [], None, None, None, None, DATA)
    assert "Importance matrix" in md
