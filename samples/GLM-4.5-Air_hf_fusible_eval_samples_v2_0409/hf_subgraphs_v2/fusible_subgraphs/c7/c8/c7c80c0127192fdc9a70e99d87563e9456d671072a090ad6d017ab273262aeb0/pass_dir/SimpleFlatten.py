import torch
import triton
import triton.language as tl

# Pattern matching function - simple flatten
def pattern(in_0):
    """
    Simple pattern to match just flatten operation
    """
    tmp_1 = in_0.flatten(1, -1)
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0):
    """
    Extract input tensor for the replacement
    """
    return (in_0,)

# Simple flatten kernel
@triton.jit
def simple_flatten_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple flatten kernel - just copy data since flatten is memory layout only"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data and store (flatten is just layout change, no computation)
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def simple_flatten(input_tensor):
    """Wrapper function for launching the simple flatten kernel"""
    # Calculate total number of elements
    n_elements = input_tensor.numel()
    
    # Set up block size and grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with flattened shape
    original_shape = input_tensor.shape
    # Calculate flattened shape: [N, C*H*W*...]
    flattened_dim = 1
    for dim in original_shape[1:]:
        flattened_dim *= dim
    flattened_shape = (original_shape[0], flattened_dim)
    output = torch.empty(flattened_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    simple_flatten_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    """
    Returns reference to the simple flatten implementation
    """
    return simple_flatten