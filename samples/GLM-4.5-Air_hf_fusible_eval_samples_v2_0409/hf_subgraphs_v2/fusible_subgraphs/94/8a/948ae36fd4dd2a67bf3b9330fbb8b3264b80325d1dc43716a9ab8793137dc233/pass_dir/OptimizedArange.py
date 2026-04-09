import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function  
def pattern(end_val):
    """Matches torch.arange(end_val, device=device(type='cuda', index=0))"""
    # Create the device object outside the traced operation
    cuda_device = device(type='cuda', index=0)
    return torch.arange(0, end_val, device=cuda_device, dtype=torch.int64)

# Argument extraction function
def replacement_args(end_val):
    """Extract the end value for the arange operation"""
    return (end_val,)

# Optimized triton kernel for arange
@triton.jit
def arange_kernel(output_ptr, end_val, BLOCK_SIZE: tl.constexpr):
    """Generate a range [0, end_val) efficiently"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate offsets for this program
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < end_val
    
    # Generate range values: offset
    tl.store(output_ptr + offsets, offsets.to(output_ptr.type()), mask=mask)

@torch.fx.wrap
def optimized_arange(end_val):
    """Optimized arange implementation using Triton"""
    if isinstance(end_val, torch.Tensor):
        end_val = end_val.item()
    
    # Determine optimal block size based on end_val
    if end_val <= 1024:
        BLOCK_SIZE = end_val
        num_programs = 1
    else:
        BLOCK_SIZE = 1024
        num_programs = (end_val + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor on GPU
    output = torch.empty(end_val, dtype=torch.int64, device='cuda:0')
    
    # Launch the optimized kernel
    arange_kernel[(num_programs,)](
        output_ptr=output,
        end_val=end_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_arange