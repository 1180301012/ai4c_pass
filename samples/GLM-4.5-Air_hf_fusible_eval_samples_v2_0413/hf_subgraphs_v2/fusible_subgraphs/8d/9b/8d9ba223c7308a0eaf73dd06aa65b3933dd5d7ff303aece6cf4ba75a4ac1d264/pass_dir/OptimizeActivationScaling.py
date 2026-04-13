import torch
import triton
import triton.language as tl

# Pattern matching function for activation + scaling optimization
def pattern(input_tensor):
    """
    Match the pattern: sigmoid + scaling multiplication
    This appears in all target computations:
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    """
    # Match the exact operations from the target computation
    tmp_9 = torch.sigmoid(input_tensor)
    tmp_10 = 16 * tmp_9
    return tmp_10

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized kernel using Triton
@triton.jit
def sigmoid_scale_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized sigmoid + scale computation
    # Sigmoid can be computed as: 1 / (1 + exp(-x))
    # For better numerical stability, we use a more optimized version
    result = 1.0 / (1.0 + tl.exp(-x))
    result = result * scale  # Apply scaling factor
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_sigmoid_scale(input_tensor):
    """
    Optimized version of sigmoid + scaling operations
    """
    # Create output tensor with correct shape and data type
    output = torch.empty_like(input_tensor)
    
    # Launch Triton kernel
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024  # Optimal for element-wise operations
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    sigmoid_scale_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        scale=16.0,  # This matches the scaling factor in the original computation
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_sigmoid_scale