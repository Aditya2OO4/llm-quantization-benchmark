"""Write my quantized model to disk and read it back.

File format (safetensors, one file, no llama.cpp involved):
  <module>.weight.q       int8 (int8 variant) or uint8 (int4 packed two per byte), shape (rows, n_blocks, 32 or 16)
  <module>.weight.scales  float16, shape (rows, n_blocks)       one scale per block of 32 weights
  <module>.weight.w16     float16 weight, only for tensors left unquantized
  <norm>.weight           float16 RMSNorm weights (tiny)
  metadata: format name, block size, scheme, and per-module bits/shape

So the size on disk is the real compressed size: about 4.5 bits per quantized weight for int4
(4 bits + the fp16 scale shared by 32 weights).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import custom_quant as cq
import custom_quant_eval as ev

FORMAT = "custom-blockquant-v1"


def export_model(model: nn.Module, path: Path, scheme: str) -> dict:
    """Save every compressed layer and the remaining (norm) parameters. Returns size information."""
    tensors: dict[str, torch.Tensor] = {}
    layers: dict[str, dict] = {}
    for name, module in model.named_modules():
        if not isinstance(module, (ev.QLinear, ev.QEmbedding)):
            continue
        st = module.store
        key = f"{name}.weight"
        if st.bits is None:
            tensors[f"{key}.w16"] = st.w16.contiguous()
        else:
            tensors[f"{key}.q"] = st.q.contiguous()
            tensors[f"{key}.scales"] = st.scales.contiguous()
        layers[name] = {"bits": st.bits, "rows": st.rows, "cols": st.cols, "kind": type(module).__name__}
    for name, param in model.named_parameters():
        tensors[name] = param.detach().to(torch.float16).contiguous()  # norm weights: fp16 values, same as the source
    meta = {"format": FORMAT, "block": str(cq.BLOCK), "scheme": scheme, "layers": json.dumps(layers)}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=meta)
    n_quantized = sum(layer["rows"] * layer["cols"] for layer in layers.values() if layer["bits"] is not None)
    return {"path": path.name, "bytes": path.stat().st_size, "tensors": len(tensors),
            "quantized_weights": n_quantized, "scheme": scheme}


def load_exported(path: Path, model_dir: Path) -> nn.Module:
    """Rebuild a runnable model from an exported file (the HF config supplies the architecture only)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights

    config = AutoConfig.from_pretrained(str(model_dir))
    with no_init_weights():  # weights are about to be replaced, so skip random initialisation
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    model.eval()
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        layers = json.loads(meta["layers"])
        scheme = meta["scheme"]
        for name, info in layers.items():
            key = f"{name}.weight"
            bits = info["bits"]
            store = ev._Compressed.from_stored(
                bits, info["rows"], info["cols"], scheme if bits is not None else "fp16",
                w16=f.get_tensor(f"{key}.w16") if bits is None else None,
                q=f.get_tensor(f"{key}.q") if bits is not None else None,
                scales=f.get_tensor(f"{key}.scales") if bits is not None else None)
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            cls = ev.QEmbedding if info["kind"] == "QEmbedding" else ev.QLinear
            setattr(parent, child, cls(None, None, store=store))
        model.float()
        for name, param in model.named_parameters():
            param.data.copy_(f.get_tensor(name).float())
    return model
