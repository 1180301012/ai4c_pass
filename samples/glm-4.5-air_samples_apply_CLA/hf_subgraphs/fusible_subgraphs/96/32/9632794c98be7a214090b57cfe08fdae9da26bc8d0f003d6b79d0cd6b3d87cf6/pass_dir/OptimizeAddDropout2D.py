import torch
import triton
import triton.language as tl
import math

# Pattern matching function for element-wise addition followed by dropout2d
def pattern(in_3, in_4):
    # Match the addition operation
    tmp_3 = in_4 + in_3
    # Match the dropout2d operation
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    # Return the result (must match what model returns)
    return tmp_4

# Argument extraction function
def replacement_args(in_3, in_4):
    return (in_3, in_4)

# Optimized addition + dropout2d kernel using Triton
@triton.jit
def optimized_add_dropout_kernel(
    x_3_ptr,
    x_4_ptr,
    output_ptr,
    n_elements,
    p: tl.constexpr,
    training: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input tensors
    x_3 = tl.load(x_3_ptr + offsets, mask=mask, other=0.0)
    x_4 = tl.load(x_4_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    sum_val = x_4 + x_3
    
    # Apply dropout if in training mode
    if training:
        # Generate random mask using Triton's built-in functions
        # Scale by 1/(1-p) to maintain expected value during training
        dropout_mask = tl.rand(offsets) > p
        out = sum_val * dropout_mask * (1.0 / (1.0 - p))
    else:
        # During inference, dropout is a no-op
        out = sum_val
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_add_dropout2d(x_3, x_4, p=0.1, training=False):
    # Get tensor properties
    n_elements = x_3.numel()
    
    # Choose optimal block size based on tensor characteristics
    # For large tensors, use larger blocks for better occupancy
    if n_elements >= 131072:  # 128K+ elements
        BLOCK_SIZE = 1024
    elif n_elements >= 65536:  # 64K+ elements
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 256
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(x_3)
    
    # Launch kernel
    optimized_add_dropout_kernel[(num_programs,)](
        x_3_ptr=x_3,
        x_4_ptr=x_4,
        output_ptr=output,
        n_elements=n_elements,
        p=p,
        training=training,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    # Return a function that can be called with the correct arguments
    # The pattern matches in_3 and in_4, and the args are extracted correctly
    return optimized_add_dropout2d