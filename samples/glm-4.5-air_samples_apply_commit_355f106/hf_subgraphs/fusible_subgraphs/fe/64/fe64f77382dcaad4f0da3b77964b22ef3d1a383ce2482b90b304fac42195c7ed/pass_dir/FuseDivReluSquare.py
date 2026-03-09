import torch
import triton
import triton.language as tl
from torch import fx

# Build pattern as FX Graph directly
class PatternModule(torch.nn.Module):
    def forward(self, in_0):
        # Division by constant
        tmp_0 = in_0 / 11.313708498984761
        # ReLU activation
        tmp_1 = torch.nn.functional.relu(tmp_0)
        # Square operation - return as tuple to match model structure
        tmp_2 = torch.square(tmp_1)
        return tmp_2

# Use the class to create the pattern
def pattern(in_0):
    # Just delegate to the PatternModule
    pm = PatternModule()
    return pm(in_0)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel that fuses div + relu + square
@triton.jit
def fused_div_relu_square_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    divisor: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Operation 1: Division by constant
    x = x / divisor
    
    # Operation 2: ReLU activation
    x = tl.maximum(x, 0.0)
    
    # Operation 3: Square
    x = x * x
    
    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_div_relu_square_wrapper(in_0):
    # Compute total number of elements
    n_elements = in_0.numel()
    
    # Choose block size - use 1024 as default, can be tuned
    BLOCK_SIZE = 1024
    
    # Calculate grid
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(in_0)
    
    # Launch kernel
    fused_div_relu_square_kernel[(num_programs,)](
        in_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
        divisor=11.313708498984761,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_div_relu_square_wrapper