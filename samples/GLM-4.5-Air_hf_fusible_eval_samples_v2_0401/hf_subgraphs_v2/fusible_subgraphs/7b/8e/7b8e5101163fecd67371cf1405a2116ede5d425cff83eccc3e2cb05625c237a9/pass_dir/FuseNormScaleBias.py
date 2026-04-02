import torch
import triton
import triton.language as tl

# Pattern matching function - try matching just multiplication first
def pattern(in_1, tmp_2):
    """Simple multiplication pattern: scale * ReLU_output"""
    tmp_3 = in_1 * tmp_2
    return tmp_3

# Argument extraction function
def replacement_args(in_1, tmp_2):
    return (in_1, tmp_2)

# Triton kernel optimized for simple multiplication operation with autotuning
@triton.jit
def mul_kernel(
    x_ptr,              # First tensor (larger tensor)
    y_ptr,              # Second tensor (scalar scale)
    output_ptr,         # Output tensor
    n_elements,         # Total number of elements in x
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input data from larger tensor
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load scalar value (only first element since it's a scalar)
    y_scalar = tl.load(y_ptr + 0, mask=True)
    
    # Multiply by scalar
    result = x * y_scalar
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

# Optimized multiplication function with autotuning
@torch.fx.wrap
def optimized_mul(in_1, tmp_2):
    # Calculate total number of elements
    n_elements = tmp_2.numel()
    
    # Adaptive block size based on tensor size for better performance
    if n_elements < 1024:
        BLOCK_SIZE = 128    # Small block for small tensors  
    elif n_elements < 10000:
        BLOCK_SIZE = 256    # Medium block for medium tensors
    elif n_elements < 100000:
        BLOCK_SIZE = 512    # Large block for large tensors
    else:
        BLOCK_SIZE = 1024   # Extra large block for very large tensors
    
    # Calculate grid size (number of programs needed)
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same dtype and device as input
    output = torch.empty_like(tmp_2)
    
    # Launch the Triton kernel with autotuned block size
    mul_kernel[(num_programs,)](
        x_ptr=tmp_2,
        y_ptr=in_1,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_mul