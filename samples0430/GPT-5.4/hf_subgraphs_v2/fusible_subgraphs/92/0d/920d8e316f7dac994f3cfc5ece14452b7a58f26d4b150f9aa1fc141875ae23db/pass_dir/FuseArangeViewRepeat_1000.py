import torch
from torch import device
import triton
import triton.language as tl


def pattern():
    tmp_0 = torch.arange(0, 1000, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return (tmp_2,)


def replacement_args():
    return ()


@triton.jit
def _arange_repeat_rows_kernel(
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    values = offsets.to(tl.int64)
    tl.store(out_ptr + row * N + offsets, values, mask=mask)


@torch.fx.wrap
def _arange_repeat_1000():
    out = torch.empty((2, 1000), device='cuda', dtype=torch.int64)
    _arange_repeat_rows_kernel[(2,)](
        out,
        N=1000,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return (out,)


def replacement_func():
    return _arange_repeat_1000