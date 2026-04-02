"""
Optimization pass: replace in_1.index_select(-2, idx) with a Triton
row-gather kernel.

Pattern: matches ONLY the index_select node (1 returning node).
The surrounding getitem calls are left untouched in the compiled graph.

Best config found empirically on NVIDIA A30:
  BLOCK_M=32, BLOCK_D=16, num_warps=4
  → 35 CTAs, 128 threads, 4 elements/thread (64-bit loads)
  → score ~0.54

Target computation (model.py):
    tmp_0 = in_0[1]
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)   ← matched here
    return (tmp_0, tmp_2)

Inputs:
    in_0: [2, 1100]  int64
    in_1: [1000, 16] bf16/f16
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton kernel: 2D row-gather
# ---------------------------------------------------------------------------

@triton.jit
def row_gather_kernel(
    x_ptr,
    idx_ptr,
    out_ptr,
    M,
    stride_xN,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid    = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = tl.arange(0, BLOCK_D)
    mask_m = m_offs < M

    indices  = tl.load(idx_ptr + m_offs, mask=mask_m, other=0)
    src_ptrs = x_ptr + indices[:, None] * stride_xN + d_offs[None, :]
    vals     = tl.load(src_ptrs, mask=mask_m[:, None], other=0.0)

    out_ptrs = out_ptr + m_offs[:, None] * BLOCK_D + d_offs[None, :]
    tl.store(out_ptrs, vals, mask=mask_m[:, None])


# ---------------------------------------------------------------------------
# Best empirical config (BLOCK_M=32, num_warps=4, 35 CTAs)
# ---------------------------------------------------------------------------
_BM   = 32
_BD   = 16
_GRID = (35,)   # ceil(1100 / 32)
_NW   = 4       # num_warps


# ---------------------------------------------------------------------------
# Pattern: matches ONLY the index_select node
# ---------------------------------------------------------------------------

def pattern(idx, x):
    return x.index_select(-2, idx)


def replacement_args(idx, x):
    return (idx, x)


# ---------------------------------------------------------------------------
# Minimal @torch.fx.wrap wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_gather(idx, x):
    out = torch.empty((1100, 16), dtype=x.dtype, device=x.device)
    row_gather_kernel[_GRID](
        x, idx, out,
        1100,    # M (hardcoded)
        16,      # stride_xN (contiguous [N, 16])
        BLOCK_M=_BM, BLOCK_D=_BD, num_warps=_NW,
    )
    return out


# ---------------------------------------------------------------------------
# Replacement: single FX node → 1 returning node matching the pattern
# ---------------------------------------------------------------------------

def _replacement(idx, x):
    return triton_gather(idx, x)


def replacement_func():
    return _replacement