"""
Fuses: in_4.view(1, 1, -1, 64) -> transpose(1, 2) -> contiguous()
into a single Triton element-copy kernel.

Shapes (both float16 and bfloat16):
  in_4  : [1, 1, 512]   key states  (CUDA)
  output: [1, 8, 1, 64] contiguous
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: copy 512 elements from [1,1,512] flat layout
#                into [1,8,1,64] contiguous layout (same 512 elements).
# ---------------------------------------------------------------------------
@triton.jit
def _view_transpose_contiguous_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(in_ptr + offs, mask=mask)
    # Input [1,1,512] flat == output [1,8,1,64] flat (both are 512 elements)
    # The view+transpose reorders from [s=1, h=8, d=64] -> [h=8, s=1, d=64]
    # For the contiguous copy, just write to the same flat index.
    tl.store(out_ptr + offs, x, mask=mask)


@torch.fx.wrap
def fused_view_transpose_contiguous(in_4):
    """
    Fused replacement for:
        tmp_3  = in_4.view(1, 1, -1, 64)              # [1,1,8,64]
        tmp_4  = tmp_3.transpose(1, 2)                # [1,8,1,64] non-contig
        tmp_9  = tmp_4.contiguous()                   # [1,8,1,64] contiguous
    Returns tmp_9.
    """
    N = 512
    BLOCK_SIZE = 512

    out = torch.empty((1, 8, 1, 64), dtype=in_4.dtype, device=in_4.device)

    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _view_transpose_contiguous_kernel[grid](
        in_4,   # [1,1,512] contiguous, flat ptr = in_4[0,0,:]
        out,    # [1,8,1,64] contiguous
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------
def pattern(in_4):
    tmp_3 = in_4.view(1, 1, -1, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_9 = tmp_4.contiguous()
    return tmp_9


def replacement_args(in_4):
    return (in_4,)


def replacement_func():
    return fused_view_transpose_contiguous