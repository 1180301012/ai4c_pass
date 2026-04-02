import torch
import triton
import triton.language as tl
from torch import device


def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    return tmp_0


def replacement_args():
    return ()


@triton.jit
def arange_one_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, offsets.to(tl.int64), mask=mask)


@torch.fx.wrap
def triton_arange_one():
    n_elements = 1
    out = torch.empty(n_elements, dtype=torch.int64, device='cuda')
    BLOCK_SIZE = 16
    grid = (1,)
    arange_one_kernel[grid](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return triton_arange_one