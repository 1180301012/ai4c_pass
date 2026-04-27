"""
Flash Attention pass for scale = 8.0  (sqrt(64)).
Fuses: matmul -> /scale -> softmax -> dropout(0) -> matmul -> permute(0,2,1,3) -> contiguous
and writes output directly in permuted-contiguous [B, Sq, H, D] layout.
"""

import torch
from pass_dir._flash_attn_kernel import flash_attention_forward, shared_flash_attn_dispatch

_SCALE = 8.0


# ── Pattern ───────────────────────────────────────────────────────────────────
def pattern(q, k, v):
    attn = torch.matmul(q, k)
    attn = attn / 8.0
    attn = torch.nn.functional.softmax(attn, dim=-1)
    out  = torch.matmul(attn, v)
    out  = out.permute(0, 2, 1, 3)
    out  = out.contiguous()
    return out


# ── Argument extractor ────────────────────────────────────────────────────────
def replacement_args(q, k, v):
    return (q, k, v, "scale8")


# ── Shared replacement (same function object across all scale passes) ─────────
def replacement_func():
    return shared_flash_attn_dispatch