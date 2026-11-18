import numpy as np
import pytest

import custom_quant as cq


def _weights(shape, seed=0, scale=0.05):
    return (np.random.default_rng(seed).standard_normal(shape) * scale).astype(np.float32)


@pytest.mark.parametrize("bits", [8, 4])
def test_roundtrip_error_is_bounded_by_half_a_step(bits):
    x = _weights((64, 256))
    qt = cq.quantize(x, bits)
    err = np.abs(cq.dequantize(qt) - x).reshape(64, -1, 32)
    step = qt.scales.astype(np.float32)[..., None]
    # half a quantization step, plus the tiny clip caused by rounding the scale to fp16
    assert np.all(err <= 0.5 * step + cq.QMAX[bits] * step * 2.0**-10 + 1e-9)


def test_int8_is_much_more_accurate_than_int4():
    x = _weights((128, 512))
    sqnr8 = cq.error_stats(x, cq.fake_quantize(x, 8))["sqnr_db"]
    sqnr4 = cq.error_stats(x, cq.fake_quantize(x, 4))["sqnr_db"]
    assert sqnr8 > sqnr4 + 20  # ~48 dB more levels-worth: 127 vs 7 levels per side


@pytest.mark.parametrize("shape", [(64, 128), (5, 96), (7,), (2, 3, 64)])
def test_shapes_and_dtypes(shape):
    x = _weights(shape)
    qt = cq.quantize(x, 8)
    out = cq.dequantize(qt)
    assert out.shape == x.shape and out.dtype == np.float32
    assert qt.q.dtype == np.int8 and qt.scales.dtype == np.float16
    assert qt.scales.shape == qt.q.shape[:2]
    assert np.abs(qt.q).max() <= 127


@pytest.mark.parametrize("last_dim", [1, 31, 33, 50, 100])
def test_sizes_not_divisible_by_block(last_dim):
    x = _weights((3, last_dim))
    qt = cq.quantize(x, 4)
    assert qt.q.shape == (3, -(-last_dim // 32), 32)
    out = cq.dequantize(qt)
    assert out.shape == x.shape
    assert cq.error_stats(x, out)["sqnr_db"] > 10  # padding must not corrupt real values


def test_all_zero_blocks_stay_zero_without_nan():
    x = _weights((2, 96))
    x[0, :32] = 0.0  # first block of row 0 is all zero
    x[1, :] = 0.0    # whole row is zero
    for bits in (8, 4):
        qt = cq.quantize(x, bits)
        out = cq.dequantize(qt)
        assert np.all(np.isfinite(out))
        assert np.all(out[0, :32] == 0) and np.all(out[1] == 0)
        assert qt.scales[0, 0] == 0 and np.all(qt.scales[1] == 0)


def test_tiny_values_that_underflow_fp16_scale_become_zero_not_nan():
    x = np.full((1, 32), 1e-12, dtype=np.float32)
    out = cq.fake_quantize(x, 8)
    assert np.all(np.isfinite(out)) and np.all(out == 0)


def test_absmax_element_is_reproduced_almost_exactly():
    x = _weights((4, 64))
    out = cq.fake_quantize(x, 8)
    idx = np.abs(x).reshape(4, 2, 32).argmax(-1)
    rows, blocks = np.indices(idx.shape)
    got = out.reshape(4, 2, 32)[rows, blocks, idx]
    want = x.reshape(4, 2, 32)[rows, blocks, idx]
    assert np.allclose(got, want, rtol=2.0**-10)  # only fp16 scale rounding remains


def test_deterministic():
    x = _weights((16, 128), seed=3)
    a, b = cq.quantize(x, 4), cq.quantize(x.copy(), 4)
    assert np.array_equal(a.q, b.q) and np.array_equal(a.scales, b.scales)


def test_symmetric_for_negated_input():
    x = _weights((8, 64), seed=5)
    assert np.array_equal(cq.fake_quantize(-x, 4), -cq.fake_quantize(x, 4))


def test_int4_pack_roundtrip_and_storage_cost():
    x = _weights((4, 64))
    qt = cq.quantize(x, 4)
    packed = cq.pack_int4(qt.q)
    assert packed.shape == (4, 2, 16) and packed.dtype == np.uint8
    assert np.array_equal(cq.unpack_int4(packed), qt.q)
    assert qt.stored_bytes == 4 * 64 // 2 + 4 * 2 * 2
    assert cq.bits_per_weight(4) == 4.5 and cq.bits_per_weight(8) == 8.5


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        cq.quantize(np.array([1.0, np.nan]), 8)
    with pytest.raises(ValueError):
        cq.quantize(np.ones(4), 3)
    with pytest.raises(ValueError):
        cq.quantize(np.array([1e9], dtype=np.float32), 4)  # scale would not fit fp16


def test_error_stats_known_values():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    stats = cq.error_stats(x, x - 0.1)
    assert stats["mse"] == pytest.approx(0.01)
    assert stats["sqnr_db"] == pytest.approx(20.0)
    assert cq.error_stats(x, x)["sqnr_db"] == float("inf")
    assert cq.error_stats(np.zeros(3), np.zeros(3))["sqnr_db"] is None


def test_empty_tensor():
    qt = cq.quantize(np.zeros((3, 0), dtype=np.float32), 8)
    assert cq.dequantize(qt).shape == (3, 0)


@pytest.mark.parametrize("bits", [8, 4])
def test_chunked_error_matches_full_error(bits):
    x = _weights((50, 96), seed=9)
    full = cq.error_stats(x, cq.fake_quantize(x, bits))
    chunked = cq.quantization_error(x, bits, chunk_elems=96 * 7)  # 7 rows at a time
    assert chunked["mse"] == pytest.approx(full["mse"]) and chunked["sqnr_db"] == pytest.approx(full["sqnr_db"])
    assert chunked["max_abs_err"] == pytest.approx(full["max_abs_err"]) and chunked["numel"] == x.size
