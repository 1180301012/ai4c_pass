import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def cast_to_bool_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = (x != 0).to(tl.int1)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_cast_to_bool(in_0):
    bool_out = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    n_cast = in_0.numel()
    BLOCK_SIZE = 1024
    grid_cast = (triton.cdiv(n_cast, BLOCK_SIZE),)
    cast_to_bool_kernel[grid_cast](in_0, bool_out, n_cast, BLOCK_SIZE=BLOCK_SIZE)
    return bool_out


def pattern(in_0):
    tmp_2 = in_0.to(device = device(type='cuda', index=0), dtype = torch.bool)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return triton_cast_to_bool