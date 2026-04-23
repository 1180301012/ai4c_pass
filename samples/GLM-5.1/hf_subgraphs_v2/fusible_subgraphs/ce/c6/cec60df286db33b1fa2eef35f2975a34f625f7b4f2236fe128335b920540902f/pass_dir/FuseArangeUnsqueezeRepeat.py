import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - must match the exact computation in model.py
def pattern():
    tmp_0 = torch.arange(0, 1, device = device(type='cuda', index=0))
    tmp_1 = tmp_0.unsqueeze(0)
    tmp_2 = tmp_1.repeat(1, 1)
    return (tmp_0, tmp_2)

# Argument extraction function - no external args needed for this pattern
def replacement_args():
    return ()

@triton.jit
def arange_unsqueeze_repeat_kernel(
    out0_ptr,  # flat arange output [1]
    out1_ptr,  # unsqueeze+repeat output [1, 1]
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Compute arange values: value = offset
    values = offsets.to(tl.int64)
    
    # Store to flat output (tmp_0)
    tl.store(out0_ptr + offsets, values, mask=mask)
    
    # Store to 2D output (tmp_2) - since repeat(1,1) with unsqueeze(0), 
    # the 2D layout is just [1, 1] which maps linearly the same as flat
    tl.store(out1_ptr + offsets, values, mask=mask)

@torch.fx.wrap
def kernel_wrapper():
    N = 1  # arange(0, 1) produces 1 element
    BLOCK_SIZE = 1
    
    # Create output tensors
    out0 = torch.empty(N, dtype=torch.int64, device='cuda')
    out1 = torch.empty(1, 1, dtype=torch.int64, device='cuda')
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    arange_unsqueeze_repeat_kernel[(num_programs,)](
        out0_ptr=out0,
        out1_ptr=out1,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out0, out1)

def replacement_func():
    return kernel_wrapper