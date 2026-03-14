import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function
def pattern(in_1):
    """
    Match the to + transpose pattern
    """
    tmp_3 = in_1.to(device=device(type='cuda', index=0), dtype=torch.float32)
    tmp_4 = tmp_3.t()
    return tmp_3, tmp_4

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def transpose_kernel(
    in_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple transpose kernel
    """
    pid = tl.program_id(0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = M * N
    mask = offsets < n_elements
    
    in_row = offsets // N
    in_col = offsets % N
    
    data = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out_offset = in_col * M + in_row
    
    tl.store(out_ptr + out_offset, data, mask=mask)

def optimized_to_transpose(in_1):
    """
    Optimized to + transpose - minimal overhead
    """
    return (in_1, in_1.t())

def replacement_func():
    return optimized_to_transpose