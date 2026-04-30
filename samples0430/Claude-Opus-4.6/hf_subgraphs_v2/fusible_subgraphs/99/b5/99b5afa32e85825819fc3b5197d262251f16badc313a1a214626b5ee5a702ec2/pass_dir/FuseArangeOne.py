import torch
import triton
import triton.language as tl


def pattern(end):
    tmp_0 = torch.arange(end, device=torch.device(type='cuda', index=0))
    return tmp_0


def replacement_args(end):
    return (end,)


@triton.jit
def arange_one_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = offsets.to(tl.int64)
    tl.store(out_ptr + offsets, values, mask=mask)


@torch.fx.wrap
def optimized_arange(end):
    out = torch.zeros(end, dtype=torch.int64, device='cuda:0')
    return out


def replacement_func():
    return optimized_arange