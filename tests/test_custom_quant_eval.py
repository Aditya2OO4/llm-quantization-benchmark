import copy
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import custom_quant as cq  # noqa: E402
import custom_quant_eval as ev  # noqa: E402


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = transformers.LlamaForCausalLM(cfg).half().eval()  # fp16 weights, like the real loader
    return model


def _logits(model, ids):
    with torch.inference_mode():
        return model(input_ids=ids).logits.float()


IDS = torch.randint(1, 96, (2, 24), generator=torch.Generator().manual_seed(1))


def test_qlinear_matches_numpy_fake_quantize():
    w = torch.randn(70, 96, generator=torch.Generator().manual_seed(2)).half()  # 96 = 3 blocks
    x = torch.randn(5, 96, generator=torch.Generator().manual_seed(3))
    for bits in (8, 4):
        expected = x.numpy() @ cq.fake_quantize(w.float().numpy(), bits).T
        got = ev.QLinear(w, bits)(x).numpy()
        assert np.allclose(got, expected, atol=1e-5)


def test_qlinear_chunking_does_not_change_the_result(monkeypatch):
    w = torch.randn(40, 64, generator=torch.Generator().manual_seed(4)).half()
    x = torch.randn(3, 64, generator=torch.Generator().manual_seed(5))
    whole = ev.QLinear(w, 4)(x)
    monkeypatch.setattr(ev, "_CHUNK_ELEMS", 64 * 7)  # 7 rows at a time
    assert torch.allclose(ev.QLinear(w, 4)(x), whole, atol=1e-6)


def test_embedding_lookup_matches_dequantized_table():
    w = torch.randn(30, 64, generator=torch.Generator().manual_seed(6)).half()
    ids = torch.tensor([[3, 7, 3], [29, 0, 7]])
    table = torch.from_numpy(cq.fake_quantize(w.float().numpy(), 4))
    assert torch.allclose(ev.QEmbedding(w, 4)(ids), table[ids], atol=1e-6)


def test_fp16_mode_reproduces_the_original_model_and_quantization_adds_error():
    base = _tiny_model()
    reference = _logits(copy.deepcopy(base).float(), IDS)
    errs = {}
    for mode in ("fp16", "int8", "int4"):
        quantized = ev.quantize_model(copy.deepcopy(base), *ev.VARIANTS[mode])
        errs[mode] = float((_logits(quantized, IDS) - reference).abs().mean())
    assert errs["fp16"] < 1e-5          # baseline path is exact
    assert errs["fp16"] < errs["int8"] < errs["int4"]


def test_overrides_set_a_different_width_for_chosen_tensors():
    model = ev.quantize_model(_tiny_model(), *ev.VARIANTS["int4_head8"])
    assert model.lm_head.store.bits == 8
    assert model.model.layers[0].mlp.up_proj.store.bits == 4
    assert model.model.embed_tokens.store.bits == 4
    assert ev.quantize_model(_tiny_model(), 4, {"lm_head": None}).lm_head.store.bits is None


def test_build_windows_replaces_first_token_with_bos():
    w = ev.build_windows(list(range(100, 100 + 2100)), n_ctx=512, n_chunks=40, bos_id=1)
    assert w.shape == (4, 512)  # only 4 full windows exist
    assert np.all(w[:, 0] == 1) and w[1, 1] == 100 + 512 + 1
    assert ev.build_windows(list(range(5000)), 512, 2, 1).shape == (2, 512)
    with pytest.raises(ValueError):
        ev.build_windows([1, 2, 3], 512, 2, 1)


def test_window_nll_scores_second_half_only():
    model = ev.quantize_model(_tiny_model(), None)
    windows = np.random.default_rng(0).integers(2, 96, size=(3, 32)).astype(np.int64)
    windows[:, 0] = 1
    nll = ev.window_nll(model, windows, batch=2)
    assert nll.shape == (3, 32 // 2 - 1)  # 15 scored tokens per window, like llama-perplexity's 255 of 512
    # cross-check one window against a direct computation
    logits = _logits(model, torch.from_numpy(windows[:1]))[0]
    logp = torch.log_softmax(logits, -1)
    manual = -logp[torch.arange(16, 31), torch.from_numpy(windows[0, 17:32])]
    assert np.allclose(nll[0], manual.numpy(), atol=1e-4)


def test_summarize_nll_matches_definition():
    nll = np.array([[1.0, 2.0], [3.0, 2.0]])
    s = ev.summarize_nll(nll)
    assert s["ppl"] == pytest.approx(math.exp(2.0))
    var = np.mean(nll**2) - 4.0
    assert s["ppl_err"] == pytest.approx(math.exp(2.0) * math.sqrt(var / 3))
    assert s["window_mean_nll"] == [1.5, 2.5] and s["tokens"] == 4


def test_paired_delta_is_tighter_than_unpaired_when_windows_differ_a_lot():
    rng = np.random.default_rng(0)
    ref = rng.normal(2.0, 0.5, size=(40, 255)) + rng.normal(0, 0.5, size=(40, 1))  # windows differ in difficulty
    worse = ref + 0.03 + rng.normal(0, 0.01, size=ref.shape)  # quantized model: same windows, small extra loss
    d = ev.paired_delta(worse, ref)
    assert d["delta_pct"] == pytest.approx(3.0, abs=0.3)
    assert d["se_paired_pct"] < d["se_unpaired_pct"] / 5


# ---- schemes and importance in the model path -----------------------------------------------------

def test_every_variant_is_a_valid_triple():
    for name, (bits, overrides, scheme) in ev.VARIANTS.items():
        assert bits in (None, 4, 8) and scheme in cq.SCHEMES and isinstance(overrides, dict), name
    assert ev.NEEDS_IMPORTANCE <= set(ev.VARIANTS)


def test_better_schemes_give_lower_logit_error_on_a_tiny_model():
    base = _tiny_model(seed=7)
    reference = _logits(copy.deepcopy(base).float(), IDS)
    errs = {}
    for scheme in ("absmax", "signed", "opt"):
        q = ev.quantize_model(copy.deepcopy(base), 4, {}, scheme)
        errs[scheme] = float((_logits(q, IDS) - reference).abs().mean())
    assert errs["opt"] < errs["absmax"]  # signed sits in between on average; only the ends are asserted (tiny model, noisy)


def test_collect_importance_matches_an_independent_hook():
    model = ev.quantize_model(_tiny_model(), None)
    windows = np.random.default_rng(1).integers(1, 96, size=(3, 24)).astype(np.int64)
    seen = []
    h = model.model.layers[0].input_layernorm.register_forward_hook(lambda m, i, o: seen.append(o.detach().double()))
    imp = ev.collect_importance(model, windows, batch=2)
    h.remove()
    expected = torch.cat([s.reshape(-1, 64) for s in seen]).pow(2).mean(0).numpy()
    assert np.allclose(imp["model.layers.0.self_attn.q_proj"], expected, rtol=1e-4)
    assert imp["model.layers.0.self_attn.k_proj"].shape == (64,) and imp["model.layers.0.mlp.down_proj"].shape == (128,)
    assert "lm_head" in imp and all(np.all(v >= 0) for v in imp.values())
    assert len(imp) == 2 * 7 + 1  # 7 linears in each of 2 layers + lm_head; the embedding has no input activations


def test_importance_is_passed_to_the_layer_that_owns_it():
    base = _tiny_model()
    windows = np.random.default_rng(2).integers(1, 96, size=(2, 24)).astype(np.int64)
    imp = ev.collect_importance(ev.quantize_model(copy.deepcopy(base), None), windows)
    plain = ev.quantize_model(copy.deepcopy(base), 4, {}, "opt")
    weighted = ev.quantize_model(copy.deepcopy(base), 4, {}, "opt", importance=imp)
    a, b = plain.model.layers[0].mlp.up_proj.store, weighted.model.layers[0].mlp.up_proj.store
    assert not torch.equal(a.scales, b.scales) or not torch.equal(a.q, b.q)  # importance changed the encoding
    assert a.q.shape == b.q.shape and a.scales.shape == b.scales.shape       # ...but never the format


def test_load_model_refuses_importance_variants_without_statistics(tmp_path):
    with pytest.raises(ValueError, match="importance"):
        ev.load_model(tmp_path, "int4imp_head8")
