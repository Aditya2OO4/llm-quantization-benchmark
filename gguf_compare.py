"""Compare weight error: my quantizer vs llama.cpp's, on the SAME tensors.

Reference = the tensors of the F16 GGUF (what llama-quantize starts from).
  - llama.cpp's error: dequantize the matching tensor of the Q8_0 / Q4_0 GGUF with the
    `gguf` package and compare it with the F16 tensor.
  - my error: quantize that same F16 tensor with custom_quant and compare.
Only tensors stored with the matching type in the llama.cpp file are compared
(e.g. output.weight is Q6_K inside the Q4_0 file, so it is skipped and reported).
Tensors are streamed one at a time to keep RAM low.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np

import custom_quant as cq

TYPES = {8: "Q8_0", 4: "Q4_0"}


def _f32(tensor) -> np.ndarray:
    from gguf import GGMLQuantizationType
    from gguf.quants import dequantize

    if tensor.tensor_type in (GGMLQuantizationType.F16, GGMLQuantizationType.F32):
        return np.asarray(tensor.data, dtype=np.float32)
    return dequantize(tensor.data, tensor.tensor_type).astype(np.float32, copy=False)


def compare(f16_path: Path, quant_path: Path, bits: int) -> dict:
    from gguf import GGUFReader

    ref_reader, q_reader = GGUFReader(str(f16_path)), GGUFReader(str(quant_path))
    ref = {t.name: t for t in ref_reader.tensors}
    wanted = TYPES[bits]
    per_tensor, skipped = {}, {}
    for qt in q_reader.tensors:
        if qt.tensor_type.name != wanted:
            if len(qt.shape) == 2:
                skipped[qt.name] = qt.tensor_type.name
            continue
        x = _f32(ref[qt.name])
        theirs = cq.error_stats(x, _f32(qt))
        mine = cq.quantization_error(x, bits)
        per_tensor[qt.name] = {"llama_cpp": theirs, "mine": mine}
        del x
        gc.collect()
    return {"type": wanted, "tensors": per_tensor, "skipped_other_types": skipped}


def summarize(result: dict) -> dict:
    """Pooled over all compared tensors: total noise energy, total signal energy, SQNR in dB."""
    out = {}
    for who in ("llama_cpp", "mine"):
        noise = signal = n = 0.0
        for t in result["tensors"].values():
            e = t[who]
            tensor_noise = e["mse"] * e["numel"]
            noise += tensor_noise
            signal += tensor_noise * 10 ** (e["sqnr_db"] / 10) if e["sqnr_db"] is not None else 0.0
            n += e["numel"]
        out[who] = {"mse": noise / n if n else None, "numel": int(n),
                    "sqnr_db": 10 * math.log10(signal / noise) if noise > 0 and signal > 0 else None}
    return out
