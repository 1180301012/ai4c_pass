"""
AI4C pass: replace torch.cat((a, b, c), dim=2) with a Triton kernel.
Covers the three-input cat along dim=2 pattern used in all graphs.
"""

import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import _next_power_of_2


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------

def pattern(a, b, c):
    return torch.cat((a, b, c), dim=2)


# ---------------------------------------------------------------------------
# Argument extraction
# ---------------------------------------------------------------------------

def replacement_args(a, b, c):
    return (a, b, c)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_cat_dim2(a, b, c):
    """
    Concatenate three tensors along dim=2.
    All tensors have shape [B, 1, S, C] (i.e. rank-4, dim-1 is size 1).
    Returns tensor of shape [B, 1, S_a+S_b+S_c, C].
    """
    B  = a.shape[0]
    S_a = a.shape[2]
    S_b = b.shape[2]
    S_c = c.shape[2]
    C   = a.shape[3]
    S_total = S_a + S_b + S_c
    S_ab    = S_a + S_b

    out = torch.empty((B, 1, S_total, C), dtype=a.dtype, device=a.device)

    # Choose BLOCK_COLS as next-pow2 of C so we cover the full row in one tile
    BLOCK_COLS = _next_power_of_2(C)

    def grid(meta):
        return (B * triton.cdiv(S_total, meta['BLOCK_ROWS']),)

    _cat_dim2_kernel[grid](
        a, b, c, out,
        B, C,
        S_total, S_a, S_b, S_c,
        BLOCK_COLS=BLOCK_COLS,
    )
    return out


def replacement_func():
    return triton_cat_dim2