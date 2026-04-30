import torch
from torch import device
import triton
import triton.language as tl


def pattern():
    tmp_0 = torch.arange(0, 128, device = device(type='cuda'))
    return tmp_0


def replacement_args():
    return ()


@triton.jit
def _arange_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, offsets.to(tl.int64), mask=mask)


@torch.fx.wrap
def _make_arange_128():
    out = torch.empty((128,), device='cuda', dtype=torch.int64)
    _arange_kernel[(1,)](
        out,
        128,
        BLOCK_SIZE=128,
        num_warps=1,
    )
    return out


def replacement_func():
    return _make_arange_128