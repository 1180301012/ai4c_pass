import torch
import triton
import triton.language as tl
from torch import device

@triton.jit
def arange_kernel_512(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 512
    tl.store(out_ptr + offsets, offsets.to(tl.int64), mask=mask)

@triton.jit
def to_bool_kernel_512(
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
def fused_arange_to_bool_512(in_0):
    ARANGE_SIZE = 512
    BLOCK_SIZE = 512
    
    out_arange = torch.empty(ARANGE_SIZE, dtype=torch.int64, device=in_0.device)
    out_bool = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    
    grid_arange = ((ARANGE_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    arange_kernel_512[grid_arange](out_arange, BLOCK_SIZE)
    
    n_elements = in_0.numel()
    grid_bool = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    to_bool_kernel_512[grid_bool](in_0, out_bool, n_elements, BLOCK_SIZE)
    
    return (out_arange, out_bool)

def pattern(in_0):
    tmp_1 = torch.arange(0, 512, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return fused_arange_to_bool_512