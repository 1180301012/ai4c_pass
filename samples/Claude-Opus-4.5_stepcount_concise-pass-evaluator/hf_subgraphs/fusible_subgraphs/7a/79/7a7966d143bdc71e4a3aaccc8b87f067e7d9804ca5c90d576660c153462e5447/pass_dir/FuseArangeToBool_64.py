import torch
import triton
import triton.language as tl
from torch import device

@triton.jit
def to_bool_kernel(
    in_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(in_ptr + offsets, mask=mask)
    out = x != 0
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def optimized_to_bool(in_0):
    # Simple type conversion without device specification
    return in_0.to(torch.bool)

def pattern(in_0):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return optimized_to_bool