import torch
import triton
import triton.language as tl
from torch import device


def pattern():
    tmp_0 = torch.arange(0, 128, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return (tmp_2,)


def replacement_args():
    return ()


@triton.jit
def _arange_repeat_rows_kernel(out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    vals = offsets.to(tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)
    tl.store(out_ptr + n_cols + offsets, vals, mask=mask)


@torch.fx.wrap
def _arange_repeat_2x128():
    n_cols = 128
    out = torch.empty((2, n_cols), device=torch.device('cuda'), dtype=torch.int64)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_cols, BLOCK_SIZE),)
    _arange_repeat_rows_kernel[grid](out, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return (out,)


def replacement_func():
    return _arange_repeat_2x128