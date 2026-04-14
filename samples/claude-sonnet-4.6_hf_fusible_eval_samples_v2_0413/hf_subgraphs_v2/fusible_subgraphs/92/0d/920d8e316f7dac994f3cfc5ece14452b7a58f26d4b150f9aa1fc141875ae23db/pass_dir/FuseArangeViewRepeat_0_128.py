import torch
import triton
import triton.language as tl
from torch import device


# ---------------------------------------------------------------------------
# Triton kernel: fills a [2, N] int64 tensor with row-wise arange values.
# Each CTA handles one full row (pid_row in {0,1}).
# BLOCK_SIZE must be >= N so the entire row fits in one CTA.
# ---------------------------------------------------------------------------
@triton.jit
def _arange_repeat_kernel_128(
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)            # shape [BLOCK_SIZE]
    mask = col_offsets < N
    vals = col_offsets.to(tl.int64)
    row_base = pid_row * N
    tl.store(out_ptr + row_base + col_offsets, vals, mask=mask)


@torch.fx.wrap
def _arange_view_repeat_fused_128():
    N = 128
    out = torch.empty([2, N], dtype=torch.int64, device='cuda')
    # 2 CTAs – one per row; BLOCK_SIZE=128 covers all columns in one pass
    _arange_repeat_kernel_128[(2,)](
        out_ptr=out,
        N=N,
        BLOCK_SIZE=128,
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern():
    tmp_0 = torch.arange(0, 128, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    return ()


def replacement_func():
    return _arange_view_repeat_fused_128