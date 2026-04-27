"""
Pass: replace permute(0,2,1,3) + contiguous with a Triton transpose kernel.

Uses a flat 1D grid with a FIXED BLOCK size (no autotune) so there is zero
autotune overhead inside the 100 timed trials.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Flat-1D transpose kernel (fixed BLOCK, no autotune)
# [B,G,H,W] -> [B,H,G,W]
# ---------------------------------------------------------------------------
@triton.jit
def permute_0213_fast_kernel(
    in_ptr, out_ptr,
    N, G, H, W,
    BLOCK: tl.constexpr,
):
    """
    Flat 1D kernel. Output flat index i maps to output[b,h,g,w].
    Reads input[b,g,h,w] which lives at a strided location.
    Writes to output[i] which is fully coalesced.
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Decode flat output index -> (b, h, g, w)
    w   = offs % W
    rem = offs // W
    g   = rem % G
    rem = rem // G
    h   = rem % H
    b   = rem // H

    # Flat input index in [B, G, H, W] layout
    in_off = ((b * G + g) * H + h) * W + w

    vals = tl.load(in_ptr + in_off, mask=mask)
    tl.store(out_ptr + offs, vals, mask=mask)


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(x):
    tmp = x.permute(0, 2, 1, 3)
    out = tmp.contiguous()
    return out


# ---------------------------------------------------------------------------
# Replacement argument extractor
# ---------------------------------------------------------------------------
def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Replacement kernel wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def triton_permute_0213_contiguous(x):
    B, G, H, W = x.shape          # input is [B, G, H, W]
    N = B * G * H * W
    out = torch.empty(B, H, G, W, dtype=x.dtype, device=x.device)
    BLOCK = 512
    grid = (triton.cdiv(N, BLOCK),)
    permute_0213_fast_kernel[grid](x, out, N, G, H, W, BLOCK=BLOCK)
    return out


def replacement_func():
    return triton_permute_0213_contiguous