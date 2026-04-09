import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function for simple bool conversion
def pattern(x):
    """Matches tensor.to(device=device(type='cuda', index=0), dtype=torch.bool)"""
    return x.to(device=device(type='cuda', index=0), dtype=torch.bool)

# Argument extraction function
def replacement_args(x):
    """Extract the input tensor"""
    return (x,)

# Simple optimized kernel for bool conversion
@triton.jit
def simple_bool_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert tensor to bool efficiently"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices < n_elements
    
    # Load values and convert to boolean
    vals = tl.load(input_ptr + indices, mask=mask, other=0)
    bool_vals = (vals != 0)
    
    # Store results (Triton automatically handles bool to int1 conversion)
    tl.store(output_ptr + indices, bool_vals, mask=mask)

@torch.fx.wrap
def simple_bool_conversion(x):
    """Optimized bool conversion with smart kernel selection"""
    n_elements = x.numel()
    
    # For small tensors, use PyTorch's built-in .to() which is highly optimized
    if n_elements <= 4096:
        return x.to(device=x.device, dtype=torch.bool)
    
    # For larger tensors, use the optimized Triton kernel
    output = torch.empty_like(x, dtype=torch.bool)
    
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_bool_kernel[(grid_size,)](
        input_ptr=x,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return simple_bool_conversion