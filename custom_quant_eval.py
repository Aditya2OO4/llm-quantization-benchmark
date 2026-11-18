"""Evaluate my quantizer inside a HuggingFace TinyLlama (perplexity on WikiText-2).

Memory is the constraint (7.2 GB machine), so instead of keeping an fp32 copy of
the model, every nn.Linear / nn.Embedding is swapped for a layer that STORES the
compressed weights (fp16, or int8 / packed int4 + fp16 scales) and dequantizes
to fp32 only while it is used. So the model really runs on my quantized
weights, and RAM stays near the compressed size.

The scoring protocol copies llama-perplexity (step4b) so deltas are comparable:
non-overlapping windows of 512 tokens, first token of each window replaced by
BOS, only the second half of each window is scored (255 tokens per window).
The tokenizer differs (HF vs llama.cpp), so ABSOLUTE perplexities are not
comparable between the two tools. Only each tool's delta vs its own F16 is.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import custom_quant as cq

LINEAR_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_CHUNK_ELEMS = 4_000_000  # max weight elements dequantized at once (bounds temporary RAM)
# name -> (default bits, per-tensor-kind overrides). llama.cpp keeps output.weight (lm_head) at
# Q6_K inside its Q4_0 file, so "int4_head8" is the fairer match for Q4_0.
VARIANTS: dict[str, tuple[int | None, dict[str, int | None]]] = {
    "fp16": (None, {}),
    "int8": (8, {}),
    "int4": (4, {}),
    "int4_head8": (4, {"lm_head": 8}),
}


class _Compressed:
    """Weights of one 2-D tensor in storage form. bits=None keeps plain fp16 (the baseline)."""

    def __init__(self, weight: torch.Tensor, bits: int | None) -> None:
        self.bits, self.rows, self.cols = bits, weight.shape[0], weight.shape[1]
        if bits is None:
            self.w16, self.q, self.scales = weight.to(torch.float16), None, None
            return
        self.w16 = None
        step = max(1, _CHUNK_ELEMS // self.cols)
        qs, ss = [], []
        for i in range(0, self.rows, step):
            qt = cq.quantize(weight[i:i + step].float().numpy(), bits)
            qs.append(torch.from_numpy(cq.pack_int4(qt.q) if bits == 4 else qt.q))
            ss.append(torch.from_numpy(qt.scales))
        self.q, self.scales = torch.cat(qs), torch.cat(ss)

    def rows_fp32(self, start: int, stop: int) -> torch.Tensor:
        if self.bits is None:
            return self.w16[start:stop].float()
        q = self.q[start:stop]
        if self.bits == 4:
            q = torch.stack(((q & 0x0F).to(torch.int8) - 8, (q >> 4).to(torch.int8) - 8), dim=-1).flatten(-2)
        w = q.float() * self.scales[start:stop].float().unsqueeze(-1)
        return w.reshape(stop - start, -1)[:, : self.cols]

    def row_step(self) -> int:
        return max(1, _CHUNK_ELEMS // self.cols)


class QLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bits: int | None) -> None:
        super().__init__()
        self.store = _Compressed(weight, bits)  # plain attribute: model.float() leaves it alone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.store
        step = s.row_step()
        outs = [F.linear(x, s.rows_fp32(i, min(i + step, s.rows))) for i in range(0, s.rows, step)]
        return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


class QEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, bits: int | None) -> None:
        super().__init__()
        self.store = _Compressed(weight, bits)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        s = self.store  # only the looked-up rows are dequantized (block boundaries align with rows)
        unique, inverse = torch.unique(ids, return_inverse=True)
        rows = torch.cat([s.rows_fp32(int(r), int(r) + 1) for r in unique])
        return rows[inverse]


def quantize_model(model: nn.Module, bits: int | None, overrides: dict[str, int | None] | None = None) -> nn.Module:
    """Swap every linear weight, the embedding table and lm_head for compressed layers.

    `overrides` maps a tensor kind (e.g. "lm_head", "down_proj") to its own bit width
    (None = keep fp16) instead of the default `bits`.
    """
    overrides = overrides or {}

    def b(kind: str) -> int | None:
        return overrides.get(kind, bits)

    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LINEAR_NAMES:
                if hasattr(parent, name):
                    old = getattr(parent, name)
                    setattr(parent, name, QLinear(old.weight.data, b(name)))
                    del old
        gc.collect()
    emb = model.model.embed_tokens
    model.model.embed_tokens = QEmbedding(emb.weight.data, b("embed_tokens"))
    head = model.lm_head
    model.lm_head = QLinear(head.weight.data, b("lm_head"))
    del emb, head
    gc.collect()
    return model.float()  # only the small norm weights are left as real parameters


def load_model(model_dir: Path, variant: str) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.eval()
    return quantize_model(model, *VARIANTS[variant])


def build_windows(token_ids: list[int], n_ctx: int, n_chunks: int, bos_id: int) -> np.ndarray:
    """Non-overlapping windows; first token of each replaced by BOS (as llama-perplexity does)."""
    n = min(len(token_ids) // n_ctx, n_chunks)
    if n < 1:
        raise ValueError(f"text has {len(token_ids)} tokens, need at least {n_ctx}")
    windows = np.asarray(token_ids[: n * n_ctx], dtype=np.int64).reshape(n, n_ctx).copy()
    windows[:, 0] = bos_id
    return windows


@torch.inference_mode()
def window_nll(model: nn.Module, windows: np.ndarray, batch: int = 2) -> np.ndarray:
    """Per-token negative log-likelihood of the 2nd half of every window, shape (n, n_ctx//2 - 1)."""
    n_ctx = windows.shape[1]
    first = n_ctx // 2
    out = []
    for i in range(0, len(windows), batch):
        ids = torch.from_numpy(windows[i:i + batch])
        logits = model(input_ids=ids, logits_to_keep=n_ctx - first).logits[:, :-1].float()
        targets = ids[:, first + 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
        out.append(nll.reshape(ids.shape[0], -1).double().numpy())
        print(f"    windows {min(i + batch, len(windows))}/{len(windows)}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return np.concatenate(out)


def summarize_nll(nll: np.ndarray) -> dict:
    """Perplexity and the same ± as llama-perplexity (token-level standard error)."""
    flat = nll.reshape(-1)
    mean = float(flat.mean())
    var = float((flat**2).mean() - mean**2)
    err = math.sqrt(var / (flat.size - 1)) if var > 0 and flat.size > 1 else 0.0
    ppl = math.exp(mean)
    return {"ppl": ppl, "ppl_err": ppl * err, "tokens": int(flat.size), "windows": int(nll.shape[0]),
            "window_mean_nll": [float(v) for v in nll.mean(axis=1)]}


def paired_delta(nll_a: np.ndarray, nll_ref: np.ndarray) -> dict:
    """Perplexity change of A vs the reference on the SAME windows, with paired and unpaired SEs.

    Works on per-window mean NLL (pass either (windows, tokens) or already-averaged (windows,)
    arrays): d_w = a_w - ref_w. The paired standard error uses the spread of d_w only; the
    unpaired one treats A and ref as independent samples.
    """
    a, r = (np.asarray(v, dtype=np.float64) for v in (nll_a, nll_ref))
    a, r = (v.mean(axis=1) if v.ndim == 2 else v for v in (a, r))
    d, n = a - r, len(a)
    paired = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    unpaired = float(math.sqrt(a.var(ddof=1) / n + r.var(ddof=1) / n)) if n > 1 else 0.0
    return {
        "delta_pct": (math.exp(float(d.mean())) - 1) * 100,
        "se_paired_pct": paired * 100,      # SE of the change in ln(ppl), shown in % (small-change approx)
        "se_unpaired_pct": unpaired * 100,
        "windows": n,
    }
