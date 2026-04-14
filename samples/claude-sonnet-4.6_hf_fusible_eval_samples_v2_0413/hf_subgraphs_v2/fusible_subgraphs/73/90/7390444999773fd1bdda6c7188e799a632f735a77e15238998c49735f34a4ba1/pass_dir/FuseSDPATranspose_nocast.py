"""
Pass: FuseSDPATranspose_nocast
Matches: scaled_dot_product_attention(Q, K, V, attn_mask=M, dropout_p=0.0, is_causal=False)
         followed by .transpose(1, 2)
for graphs that do NOT have a .to(dtype) cast on V before SDPA.

Replaces the pair with a fused Triton Flash Attention kernel that writes
output directly in [B, S, H, D] layout (i.e. the transposed result).
"""

import torch
from pass_dir.triton_flash_attn import triton_sdpa_transpose


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(query, key, value, attn_mask):
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )
    transposed = sdpa.transpose(1, 2)
    return transposed


# ── Argument extraction ───────────────────────────────────────────────────────

def replacement_args(query, key, value, attn_mask):
    return (query, key, value, attn_mask)


# ── Replacement function ──────────────────────────────────────────────────────

def replacement_func():
    return triton_sdpa_transpose