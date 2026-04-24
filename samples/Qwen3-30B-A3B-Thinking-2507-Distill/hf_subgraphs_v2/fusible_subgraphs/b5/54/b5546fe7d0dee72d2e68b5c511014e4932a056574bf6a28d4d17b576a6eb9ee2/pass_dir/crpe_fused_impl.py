"""
Shared wrapper for the fused CRPE (Cross-stage Token Reduction with Positional Embedding) kernel.

All scale-specific pass files import `crpe_fused` from here so they share the same
@torch.fx.wrap function object, satisfying the replacement_func_limit constraint.
"""

import torch
import triton
from triton import cdiv

from pass_dir.crpe_fused_kernel import _crpe_fused_kernel


@torch.fx.wrap
def crpe_fused(in2, in3, conv_out, in4, in6, scale, route):
    """
    Fused CRPE kernel.

    Args:
        in2:   [1, C2, H1, W1]  -- first branch
        in3:   [1, C3, H2, W2]  -- second branch
        conv:  [1, C4, H3, W3]  -- conv2d output (third branch)
        in4:   [1, 8, N+1, D]   -- scale tensor (includes class token)
        in6:   [1, 8, N,   D]   -- gate tensor
        scale: float              -- scalar multiplier
        route: str               -- routing tag (ignored; same kernel for all routes)

    Returns:
        out: [1, N+1, 8*D]
    """
    # ── shape extraction ─────────────────────────────────────────────────
    # in2: [1, C2, H1, W1]
    C2 = in2.shape[1]
    H1 = in2.shape[2]
    W1 = in2.shape[3]
    S0 = C2 * H1 * W1   # branch-0 valid range

    # in3: [1, C3, H2, W2]
    C3 = in3.shape[1]
    H2 = in3.shape[2]
    W2 = in3.shape[3]
    S1 = C3 * H2 * W2   # branch-1 valid range

    # conv: [1, C4, H3, W3]
    C4 = conv_out.shape[1]
    H3 = conv_out.shape[2]
    W3 = conv_out.shape[3]
    S_conv = C4 * H3 * W3
    N = S0 + S1 + S_conv  # total spatial tokens before padding
    D = in4.shape[3]       # head dimension

    # ── output tensor ────────────────────────────────────────────────────
    out = torch.empty((1, N + 1, 8 * D), dtype=in4.dtype, device=in4.device)

    # ── block size (next power of 2 >= D) ───────────────────────────────
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    # ── kernel launch ────────────────────────────────────────────────────
    grid = lambda META: (cdiv(N, META['BLOCK_N']),)

    _crpe_fused_kernel[grid](
        in2, in3, conv_out,
        in4, in6,
        out,
        N, D, S0, S1, float(scale),
        BLOCK_D=BLOCK_D,
    )

    return out