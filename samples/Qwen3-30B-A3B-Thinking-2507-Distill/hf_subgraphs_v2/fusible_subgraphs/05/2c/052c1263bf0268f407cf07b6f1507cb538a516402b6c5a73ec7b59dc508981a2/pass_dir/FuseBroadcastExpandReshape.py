"""
Broadcast expand+reshape pass.

Matches:  x[..., None, :, :].expand(1,1,8,3,256).reshape(1,8,3,256)
Replaces with a single Triton broadcast kernel (one head-loop per row).

Applied AFTER FuseRoPEKeyEmbedding, so x = fused_rope output [1,1,3,256].

Input  x : [1, 1, S, D]  bfloat16   (S=3, D=256)
Output    : [1, 8, S, D]  bfloat16

Flat-index note: for flat dst-index i = h*(S*D) + s*D + d,
  src-index = s*D + d = i % (S*D)
  This is valid because x is [1,1,S,D] (S*D contiguous elements per "batch").
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  – connected subgraph (unsqueeze → expand → reshape)
# ---------------------------------------------------------------------------
def pattern(x):
    tmp_7  = x[slice(None, None, None), slice(None, None, None), None,
               slice(None, None, None), slice(None, None, None)]
    tmp_8  = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9  = tmp_8.reshape(1, 8, 3, 256)
    return tmp_9


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel: broadcast x over 8 heads
# ---------------------------------------------------------------------------
@triton.jit
def _broadcast_expand_kernel(
    src_ptr,    # [1,1,S,D] bfloat16  – x (input)
    dst_ptr,    # [1,8,S,D] bfloat16  – output
    S: tl.constexpr,
    D: tl.constexpr,
    TOTAL: tl.constexpr,   # S * D = 768
    BLOCK_D: tl.constexpr, # D = 256
):
    """
    Grid: (S,).  Program s handles one sequence position.
    Writes x[0,0,s,:] to all 8 head slices of dst.
    """
    s    = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = s * D

    vals = tl.load(src_ptr + base + offs, mask=mask, other=0.0)

    for h in range(8):
        dst_offs = h * TOTAL + base + offs
        tl.store(dst_ptr + dst_offs, vals, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_broadcast_expand(x):
    """
    x : [1, 1, 3, 256]  bfloat16

    Returns [1, 8, 3, 256]  bfloat16 – broadcast over 8 heads.
    """
    S = 3
    D = 256
    tmp_out = torch.empty((1, 8, S, D), dtype=x.dtype, device=x.device)
    _broadcast_expand_kernel[(S,)](
        x, tmp_out,
        S=S, D=D, TOTAL=S * D, BLOCK_D=D,
    )
    return tmp_out


def replacement_func():
    return fused_broadcast_expand