import pytest

import custom_quant_report as rep
import gguf_compare
import step6_report


def _w(sqnr, kind, layer):
    return {"sqnr_db": sqnr, "mse": 1e-6, "kind": kind, "layer": layer, "numel": 100}


WEIGHTS = {
    "int8": {"a": _w(40.0, "q_proj", 0), "b": _w(42.0, "q_proj", 1), "c": _w(41.0, "lm_head", None)},
    "int4": {"a": _w(20.0, "q_proj", 0), "b": _w(22.0, "q_proj", 1), "c": _w(21.0, "lm_head", None)},
}


def test_classify_tensor_names():
    assert rep.classify("model.layers.7.self_attn.k_proj.weight") == (7, "k_proj")
    assert rep.classify("model.layers.21.mlp.down_proj.weight") == (21, "down_proj")
    assert rep.classify("model.embed_tokens.weight") == (None, "embed_tokens")
    assert rep.classify("lm_head.weight") == (None, "lm_head")
    assert rep.classify("model.layers.3.input_layernorm.weight") == (None, None)
    assert rep.classify("model.norm.weight") == (None, None)


def test_mean_sqnr_by_kind():
    assert rep.mean_sqnr_by_kind(WEIGHTS["int8"]) == {"q_proj": 41.0, "lm_head": 41.0}


def test_layer_plot_is_written(tmp_path):
    pytest.importorskip("matplotlib")
    assert rep.plot_layers(WEIGHTS, tmp_path / "p.png").exists()
    assert rep.plot_layers({"int8": {}, "int4": {}}, tmp_path / "q.png") is None


def _ppl(ppl, nll):
    return {"ppl": ppl, "ppl_err": 0.2, "window_mean_nll": nll}


def test_ppl_table_uses_each_tools_own_baseline():
    custom = {"ppl": {"fp16": _ppl(10.0, [2.0, 2.4, 2.2, 2.3]), "int8": _ppl(10.1, [2.01, 2.41, 2.21, 2.31]),
                      "int4": _ppl(11.0, [2.1, 2.5, 2.3, 2.4])}}
    llama = {"F16": {"ppl": 8.0}, "Q8_0": {"ppl": 8.04}, "Q4_0": {"ppl": 8.4}}
    md = "\n".join(rep.ppl_table(custom, llama))
    assert "| mine (HF transformers) | int8 |" in md and "+1.01%" in md  # exp(0.01)-1 on every window
    assert "| llama.cpp | Q4_0 |" in md and "+5.00%" in md              # 8.4 / 8.0
    assert "| llama.cpp | Q8_0 |" in md and "+0.50%" in md


def test_report_lines_empty_without_data_and_included_in_markdown():
    assert rep.report_lines(None, None) == []
    custom = {"ppl": {"fp16": _ppl(10.0, [2.0, 2.2]), "int8": _ppl(10.1, [2.01, 2.21])}, "weights": WEIGHTS}
    md = "\n".join(rep.report_lines(custom, {"F16": {"ppl": 8.0}}))
    assert "My quantizer vs llama.cpp" in md and "custom_quant_layers.png" in md


def test_gguf_summary_pools_noise_energy():
    # tensor 1: signal energy 100*1e-2*1e3=1000 ... use simple numbers: noise=mse*numel
    res = {"tensors": {
        "t1": {"llama_cpp": {"mse": 0.01, "numel": 100, "sqnr_db": 20.0}, "mine": {"mse": 0.02, "numel": 100, "sqnr_db": 17.0}},
        "t2": {"llama_cpp": {"mse": 0.01, "numel": 100, "sqnr_db": 20.0}, "mine": {"mse": 0.02, "numel": 100, "sqnr_db": 17.0}},
    }}
    s = gguf_compare.summarize(res)
    assert s["llama_cpp"]["sqnr_db"] == pytest.approx(20.0)
    assert s["mine"]["sqnr_db"] == pytest.approx(17.0)
    assert s["llama_cpp"]["numel"] == 200


def test_step6_markdown_includes_custom_section():
    speed = {"hardware": {"cpu": "T", "physical_cores": 4, "logical_cores": 8, "ram_gb": 8, "os": "T"},
             "settings": {"threads": 4, "repeats": 3, "max_tokens": 128}, "configs": []}
    custom = {"ppl": {"fp16": _ppl(10.0, [2.0, 2.2]), "int4": _ppl(11.0, [2.1, 2.3])}}
    md = step6_report.build_markdown(speed, [], None, None, custom, {"F16": {"ppl": 8.0}})
    assert "My quantizer vs llama.cpp" in md


def test_pooled_sqnr_combines_tensors_by_energy():
    # tensor a: noise 1, signal 100 (20 dB); tensor b: noise 1, signal 10000 (40 dB) -> pooled 10*log10(10100/2)
    tensors = {"a": {"mse": 0.01, "numel": 100, "sqnr_db": 20.0}, "b": {"mse": 0.01, "numel": 100, "sqnr_db": 40.0}}
    assert rep.pooled_sqnr(tensors) == pytest.approx(10 * __import__("math").log10(10100 / 2))
    assert rep.pooled_sqnr({}) is None


def _ppl_entry(ppl, nll):
    return {"ppl": ppl, "ppl_err": 0.2, "window_mean_nll": nll}


def test_ladder_table_has_one_row_per_rung_and_the_llama_reference():
    base = [2.0, 2.4, 2.2, 2.3]
    custom = {
        "ppl": {"fp16": _ppl_entry(8.0, base),
                "int4_head8": _ppl_entry(8.5, [v + 0.06 for v in base]),
                "int4s_head8": _ppl_entry(8.4, [v + 0.05 for v in base]),
                "int4opt_head8": _ppl_entry(8.3, [v + 0.04 for v in base]),
                "int4imp_head8": _ppl_entry(8.2, [v + 0.03 for v in base])},
        "weights": {"int4": {"t": {"mse": 0.01, "numel": 100, "sqnr_db": 20.0}}},
        "scheme_study": {"signed": {"t": {"mse": 0.008, "numel": 100, "sqnr_db": 21.0}},
                         "opt": {"t": {"mse": 0.007, "numel": 100, "sqnr_db": 21.5}}},
        "gguf_weight_compare": {"int4": {"summary": {"llama_cpp": {"sqnr_db": 21.13}}}},
    }
    md = "\n".join(rep.ladder_table(custom, {"F16": {"ppl": 8.0}, "Q4_0": {"ppl": 8.3}}))
    assert md.count("\n| ") >= 5  # 4 rungs + reference (plus the header separator lines)
    assert "| 0 | baseline" in md and "| 3 | + my importance weighting" in md
    assert "+6.18%" in md and "-1.00%" in md  # rung 0 vs fp16; rung 1 improves on rung 0 by 0.01 in ln(ppl)
    assert "llama.cpp Q4_0" in md and "21.13" in md
    assert rep.ladder_table({"ppl": {"fp16": _ppl_entry(8.0, base)}}, None) == []  # nothing to show yet


def test_export_lines():
    custom = {"export": {"int4imp_head8": {"path": "/x/tinyllama-custom-int4imp_head8.safetensors", "mb": 640.2,
                                           "avg_bits_per_param_on_disk": 4.66, "bit_identical_after_reload": True,
                                           "windows_checked": 4}}}
    md = "\n".join(rep.export_lines(custom))
    assert "tinyllama-custom-int4imp_head8.safetensors" in md and "640 MB" in md and "yes (4 windows)" in md
    assert rep.export_lines({}) == []
