import torch
import triton
import triton.language as tl

# Pattern matching function for division operation
def pattern(in_0):
    """Match the computation: in_0 / constant"""
    tmp_0 = in_0 / 11.313708498984761
    return (tmp_0,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel for fused division + ReLU + squaring
@triton.jit
def optimized_div_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    const_div: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that performs division by constant
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data preserving dtype precision
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Division with exact constant matching PyTorch precision
    x_div = x / const_div
    
    # Store result with same precision as input
    tl.store(output_ptr + offsets, x_div, mask=mask)

@torch.fx.wrap  
def optimized_division(input_tensor):
    """
    Wrapper function for the optimized division kernel
    Handles different data types and launches the kernel with appropriate settings
    """
    n_elements = input_tensor.numel()
    
    # Use smaller block size for better GPU occupancy and lower overhead
    BLOCK_SIZE = 512
    
    if n_elements < 1024:
        # For very small tensors, use PyTorch directly to avoid kernel launch overhead
        return input_tensor / 11.313708498984761
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as input
    output_tensor = torch.empty_like(input_tensor)
    
    # Launch the optimized kernel
    optimized_div_kernel[(num_programs,)](
        input_tensor,
        output_tensor,
        n_elements,
        const_div=11.313708498984761,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_tensor

# Replacement function (returns function reference, not actual call)
def replacement_func():
    return optimized_division