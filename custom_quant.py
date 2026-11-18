"""My own block-wise symmetric absmax quantizer (NumPy only, no llama.cpp).

Same idea as llama.cpp's Q8_0 / Q4_0: split each row of a weight matrix into
blocks of 32 values, store ONE fp16 scale per block plus a small signed integer
per weight.

    scale = absmax(block) / qmax           (qmax = 127 for int8, 7 for int4)
    q     = round(x / scale)  in [-qmax, qmax]
    x'    = q * scale

Differences from llama.cpp, on purpose or by simplicity:
  - int4 uses the symmetric range [-7, 7] (15 levels). Q4_0 uses [-8, 7] with the
    scale taken from the signed max, which gives it a ~12% finer step. Mine is
    the textbook symmetric version, so I expect it to be slightly worse.
  - The scale is rounded to fp16 FIRST and the weights are quantized against
    that rounded scale, so dequantization is exactly consistent.
  - Rounding is round-half-to-even (np.rint); ggml uses round-half-away.

Edge cases handled: all-zero blocks (scale 0, no NaN), tensors whose last
dimension is not a multiple of the block size (zero-padded, then trimmed),
scalars / 1-D / N-D tensors (blocks run along the last dimension, like ggml).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BLOCK = 32
QMAX = {8: 127, 4: 7}
_FP16_MAX = float(np.finfo(np.float16).max)


@dataclass(frozen=True)
class QuantizedTensor:
    """What would be stored on disk: small ints + one fp16 scale per block."""

    q: np.ndarray        # int8, shape (rows, n_blocks, block), values in [-qmax, qmax]
    scales: np.ndarray   # float16, shape (rows, n_blocks)
    shape: tuple[int, ...]  # original tensor shape
    bits: int
    block: int

    @property
    def stored_bytes(self) -> int:
        """Bytes needed on disk with int4 packed two-per-byte."""
        weights = self.q.size * self.bits // 8
        return weights + self.scales.size * 2


def bits_per_weight(bits: int, block: int = BLOCK) -> float:
    """Storage cost including the fp16 scale: 8.5 for int8, 4.5 for int4 (same as Q8_0 / Q4_0)."""
    return (bits * block + 16) / block


def _as_rows(x: np.ndarray) -> np.ndarray:
    """View any tensor as (rows, last_dim); blocks run along the last dimension."""
    if x.ndim == 0:
        return x.reshape(1, 1)
    return x.reshape(math.prod(x.shape[:-1]), x.shape[-1])


def quantize(x: np.ndarray, bits: int = 8, block: int = BLOCK) -> QuantizedTensor:
    if bits not in QMAX:
        raise ValueError(f"bits must be one of {sorted(QMAX)}, got {bits}")
    if block < 2 or block % 2:
        raise ValueError("block must be an even number >= 2")
    x = np.asarray(x)
    if not np.all(np.isfinite(x)):
        raise ValueError("cannot quantize NaN/inf values")
    rows2d = _as_rows(x).astype(np.float32, copy=False)
    n_rows, n_cols = rows2d.shape
    n_blocks = -(-n_cols // block)  # ceil
    padded = np.zeros((n_rows, n_blocks * block), dtype=np.float32)
    padded[:, :n_cols] = rows2d  # zero padding never changes a block's absmax
    blocks = padded.reshape(n_rows, n_blocks, block)

    qmax = QMAX[bits]
    absmax = np.abs(blocks).max(axis=-1) if blocks.size else np.zeros((n_rows, n_blocks), np.float32)
    if absmax.size and absmax.max() / qmax > _FP16_MAX:
        raise ValueError("block scale does not fit in fp16")
    scales = (absmax / qmax).astype(np.float16)
    s32 = scales.astype(np.float32)
    nonzero = s32 > 0  # all-zero blocks (or scales that underflow fp16) stay exactly zero
    safe = np.where(nonzero, s32, 1.0)[..., None]
    q = np.clip(np.rint(blocks / safe), -qmax, qmax)
    q = np.where(nonzero[..., None], q, 0).astype(np.int8)
    return QuantizedTensor(q=q, scales=scales, shape=tuple(x.shape), bits=bits, block=block)


def dequantize(qt: QuantizedTensor) -> np.ndarray:
    """Back to float32 in the original shape."""
    full = qt.q.astype(np.float32) * qt.scales.astype(np.float32)[..., None]
    n_rows = full.shape[0]
    flat = full.reshape(n_rows, -1)
    n_cols = qt.shape[-1] if qt.shape else 1
    return flat[:, :n_cols].reshape(qt.shape)


def fake_quantize(x: np.ndarray, bits: int = 8, block: int = BLOCK) -> np.ndarray:
    """quantize -> dequantize: the weights the model would see after compression."""
    return dequantize(quantize(x, bits, block))


def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack int4 values in [-8, 7] two per byte (last axis halves). Inverse: unpack_int4."""
    u = (q.astype(np.int16) + 8).astype(np.uint8)
    return u[..., 0::2] | (u[..., 1::2] << 4)


def unpack_int4(packed: np.ndarray) -> np.ndarray:
    lo = (packed & 0x0F).astype(np.int16) - 8
    hi = (packed >> 4).astype(np.int16) - 8
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.int8)
    out[..., 0::2], out[..., 1::2] = lo, hi
    return out


def error_stats(original: np.ndarray, restored: np.ndarray) -> dict:
    """MSE, SQNR (dB) and max error. SQNR = 10 log10(signal energy / noise energy)."""
    x = np.asarray(original, dtype=np.float64)
    err = x - np.asarray(restored, dtype=np.float64)
    noise, signal = float(np.sum(err**2)), float(np.sum(x**2))
    if signal == 0:
        sqnr = None  # nothing to measure against
    elif noise == 0:
        sqnr = math.inf
    else:
        sqnr = 10 * math.log10(signal / noise)
    return {
        "mse": noise / x.size if x.size else 0.0,
        "sqnr_db": sqnr,
        "max_abs_err": float(np.abs(err).max()) if x.size else 0.0,
        "numel": int(x.size),
    }


def quantization_error(x: np.ndarray, bits: int, block: int = BLOCK, chunk_elems: int = 4_000_000) -> dict:
    """error_stats(x, fake_quantize(x)) computed in row chunks, so a 65M-weight matrix needs little RAM."""
    rows = _as_rows(np.asarray(x))
    step = max(1, chunk_elems // max(rows.shape[1], 1))
    noise = signal = max_err = 0.0
    for i in range(0, rows.shape[0], step):
        part = rows[i:i + step]
        err = part.astype(np.float64) - fake_quantize(part, bits, block)
        noise += float(np.sum(err**2))
        signal += float(np.sum(part.astype(np.float64) ** 2))
        max_err = max(max_err, float(np.abs(err).max())) if err.size else max_err
    n = rows.size
    sqnr = None if signal == 0 else (math.inf if noise == 0 else 10 * math.log10(signal / noise))
    return {"mse": noise / n if n else 0.0, "sqnr_db": sqnr, "max_abs_err": max_err, "numel": int(n)}
