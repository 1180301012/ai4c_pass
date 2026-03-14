import torch
import triton
import triton.language as tl
from torch import device

# Pattern matching function - matches arange followed by in-place add of 0
def pattern():
    tmp_0 = torch.arange(128, device=device(type='cuda', index=0))
    tmp_0 += 0
    return (tmp_0,)

# Argument extraction function
def replacement_args():
    return ()

# Triton kernel that generates the arange sequence directly
@triton.jit
def arange_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # For arange, the output value equals the index
    values = offsets
    
    # Store the values
    tl.store(out_ptr + offsets, values, mask=mask)

# Wrapper function decorated with @torch.fx.wrap
@torch.fx.wrap
def fused_arange_add0():
    N = 128
    BLOCK_SIZE = 128  # Process all elements in one block since N is small
    
    # Create output tensor with int64 dtype (default for arange)
    out = torch.empty(N, dtype=torch.int64, device='cuda')
    
    # Calculate grid
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    arange_kernel[(num_programs,)](
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out,)

# Replacement function - returns the wrapper function
def replacement_func():
    return fused_arange_add0