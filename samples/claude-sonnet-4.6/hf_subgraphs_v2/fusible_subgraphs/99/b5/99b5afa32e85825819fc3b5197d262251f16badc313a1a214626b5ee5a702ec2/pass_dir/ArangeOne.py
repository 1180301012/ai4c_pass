import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def arange_1_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, offsets.to(tl.int64), mask=mask)


@torch.fx.wrap
def triton_arange_1():
    n = 1
    out = torch.empty(n, dtype=torch.int64, device='cuda')
    BLOCK_SIZE = 32
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    arange_1_kernel[grid](
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    return tmp_0


def replacement_args():
    return ()


def replacement_func():
    return triton_arange_1