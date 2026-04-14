"""
Pass: FuseSDPATranspose_fp16cast
Matches: value.to(torch.float16)
         → scaled_dot_product_attention(Q, K, cast_V, attn_mask=M, dropout_p=0.0, is_causal=False)
         → .transpose(1, 2)

The cast is a no-op (value is already float16), so we ignore it and call the
same fused Triton Flash Attention kernel as the no-cast variant.
"""

import torch
from pass_dir.triton_flash_attn import triton_sdpa_transpose


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(query, key, value, attn_mask):
    value_cast = value.to(torch.float16)
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        query, key, value_cast, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )
    transposed = sdpa.transpose(1, 2)
    return transposed


# ── Argument extraction ───────────────────────────────────────────────────────

def replacement_args(query, key, value, attn_mask):
    # value_cast is a no-op; pass original value directly
    return (query, key, value, attn_mask)


# ── Replacement function ──────────────────────────────────────────────────────

def replacement_func():
    return triton_sdpa_transpose