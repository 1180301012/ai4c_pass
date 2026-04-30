"""
Fused Scaled Dot-Product Attention for scale=8.0, head_dim=64.
Matches: matmul(Q, K^T) / 8.0 -> softmax -> dropout(no-op) -> matmul(attn, V) -> permute -> contiguous
Covers face-parsing graphs with head_dim=64.
Uses shared routing dispatcher so replacement_func is identical across all pass files.
"""

import torch
from pass_dir.shared_kernels import dispatch_fused_sdpa  # shared dispatcher


def pattern(query, key_t, value):
    matmul   = torch.matmul(query, key_t)
    scaled   = matmul / 8.0
    attn     = torch.nn.functional.softmax(scaled, dim=-1)
    attn_d   = torch.nn.functional.dropout(attn, 0.0, False, False)
    matmul_1 = torch.matmul(attn_d, value)
    perm     = matmul_1.permute(0, 2, 1, 3)
    out      = perm.contiguous()
    return out


def replacement_args(query, key_t, value):
    return (query, key_t, value, "80")


def replacement_func():
    return dispatch_fused_sdpa