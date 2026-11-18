import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import custom_quant as cq  # noqa: E402
import custom_quant_eval as ev  # noqa: E402
import custom_quant_export as ex  # noqa: E402


def _tiny(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=False)
    model = transformers.LlamaForCausalLM(cfg).eval()
    for p in model.parameters():  # fp16 weights like from_pretrained(torch_dtype=fp16); .half() would also round RoPE's inv_freq
        p.data = p.data.half()
    return model


IDS = torch.randint(1, 96, (2, 24), generator=torch.Generator().manual_seed(3))


def _logits(model):
    with torch.inference_mode():
        return model(input_ids=IDS).logits.float()


@pytest.mark.parametrize("variant", ["fp16", "int8", "int4_head8", "int4opt_head8"])
def test_export_then_load_reproduces_the_model_exactly(tmp_path, variant):
    base = _tiny()
    base.save_pretrained(tmp_path / "hf")  # supplies config.json only
    model = ev.quantize_model(copy.deepcopy(base), *ev.VARIANTS[variant])
    info = ex.export_model(model, tmp_path / "m.safetensors", ev.VARIANTS[variant][2])
    reloaded = ex.load_exported(tmp_path / "m.safetensors", tmp_path / "hf")
    assert torch.equal(_logits(model), _logits(reloaded))  # bit-for-bit, not just close
    assert info["bytes"] == (tmp_path / "m.safetensors").stat().st_size


def test_int4_file_is_about_a_quarter_of_fp16_and_matches_the_storage_formula(tmp_path):
    base = _tiny()
    fp16_bytes = sum(p.numel() * 2 for p in base.parameters())
    model = ev.quantize_model(copy.deepcopy(base), 4, {}, "opt")
    info = ex.export_model(model, tmp_path / "m.safetensors", "opt")
    # int4 weights: 4 bits + 16 bits per 32-weight block = 4.5 bits per weight (all dims here are multiples of 32)
    expected = info["quantized_weights"] * cq.bits_per_weight(4) / 8
    with safe_open(str(tmp_path / "m.safetensors"), framework="pt") as f:
        stored = sum(f.get_tensor(k).numel() * f.get_tensor(k).element_size()
                     for k in f.keys() if k.endswith((".q", ".scales")))
    assert stored == pytest.approx(expected, rel=1e-9)
    assert info["bytes"] < fp16_bytes * 0.35  # norms and the file header add a little, still far below fp16


def test_metadata_and_wrong_format_rejected(tmp_path):
    base = _tiny()
    base.save_pretrained(tmp_path / "hf")
    model = ev.quantize_model(copy.deepcopy(base), 8, {}, "absmax")
    ex.export_model(model, tmp_path / "m.safetensors", "absmax")
    with safe_open(str(tmp_path / "m.safetensors"), framework="pt") as f:
        meta = f.metadata()
    assert meta["format"] == ex.FORMAT and meta["block"] == "32" and meta["scheme"] == "absmax"
    save_file({"x": torch.zeros(1)}, str(tmp_path / "other.safetensors"), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="not a"):
        ex.load_exported(tmp_path / "other.safetensors", tmp_path / "hf")
