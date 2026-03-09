import torch
import triton
import triton.language as tl

# Pattern matching function for Hardtanh operation
def pattern(input_tensor, min_val, max_val):
    # Match the exact hardtanh operation pattern with fixed parameters
    return torch.nn.functional.hardtanh(input_tensor, min_val, max_val, False)

# Argument extraction function
def replacement_args(input_tensor, min_val, max_val):
    return (input_tensor, min_val, max_val)

# Optimized Hardtanh kernel using Triton
@triton.jit
def hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply hardtanh activation: hardtanh(x) = max(min(max(x, min_val), max_val)
    # This can be optimized with conditional branches
    x = tl.where(x < min_val, min_val, x)
    x = tl.where(x > max_val, max_val, x)
    
    # Store output values
    tl.store(output_ptr + offsets, x, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_hardtanh(input_tensor, min_val, max_val):
    # Create output tensor (not inplace since we're using custom kernel)
    output = torch.empty_like(input_tensor)
    
    # Get total number of elements
    n_elements = input_tensor.numel()
    
    # Block size optimization
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    hardtanh_kernel[grid](
        input_tensor,
        output,
        n_elements,
        min_val=min_val,
        max_val=max_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_hardtanh