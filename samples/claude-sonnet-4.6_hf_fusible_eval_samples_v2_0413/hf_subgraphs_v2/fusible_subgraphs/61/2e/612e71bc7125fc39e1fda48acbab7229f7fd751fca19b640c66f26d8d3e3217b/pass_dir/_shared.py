"""
Shared kernels and dispatch wrapper used by all pass files.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# L2-normalisation kernel — true single-pass
#
# Loads each row ONCE into registers, computes the L2 norm, normalises, stores.
# BLOCK_SIZE=2048 covers all three column widths (768 / 1024 / 1152) in one
# tile.  Masked elements are hardware-predicated away.
# ---------------------------------------------------------------------------

@triton.jit(do_not_specialize=["n_cols", "stride_row"])
def _l2_norm_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx  = tl.program_id(0)
    row_base = row_idx * stride_row
    cols     = tl.arange(0, BLOCK_SIZE)
    mask     = cols < n_cols

    x      = tl.load(x_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    norm   = tl.sqrt(sum_sq)
    out    = x / norm
    tl.store(out_ptr + row_base + cols, out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# Simple contiguous copy (kept for completeness, inactive in pass list)
# ---------------------------------------------------------------------------

@triton.jit
def _copy_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _dispatch(x, route):
    if route == "l2_normalize":
        out = torch.empty_like(x)
        _l2_norm_kernel[(x.shape[0],)](
            x, out, x.shape[1], x.stride(0),
            BLOCK_SIZE=2048, num_warps=16,
        )
        return out
    # route == "transpose_to_cuda"  (inactive)
    N   = x.shape[1]
    out = torch.empty(N, 1, dtype=x.dtype, device=x.device)
    _copy_kernel[(1,)](x, out, N, BLOCK_SIZE=2048, num_warps=4)
    return out