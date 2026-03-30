import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _arange_1024_kernel(out_ptr):
    offsets = tl.arange(0, 1024)
    tl.store(out_ptr + offsets, offsets.to(tl.int64))


@triton.jit
def _cast_bool_1024_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = (x != 0).to(tl.int8)
    tl.store(out_ptr + offsets, out, mask=mask)


def pattern(in_0):
    tmp_1 = torch.arange(0, 1024, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def fused_arange_cast_1024(in_0):
    arange_out = torch.empty(1024, dtype=torch.int64, device=in_0.device)
    _arange_1024_kernel[(1,)](arange_out)

    n_elements = in_0.numel()
    bool_out = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _cast_bool_1024_kernel[grid](in_0, bool_out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return (arange_out, bool_out)


def replacement_func():
    return fused_arange_cast_1024