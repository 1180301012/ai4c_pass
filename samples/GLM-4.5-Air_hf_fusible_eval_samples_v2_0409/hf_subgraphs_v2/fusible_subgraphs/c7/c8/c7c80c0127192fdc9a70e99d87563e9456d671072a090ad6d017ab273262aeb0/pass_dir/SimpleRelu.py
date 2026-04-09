import torch
import triton
import triton.language as tl

# Pattern matching function - simple ReLU
def pattern(in_0):
    """
    Simple pattern to match just ReLU operation
    """
    tmp_0 = torch.nn.functional.relu(in_0, inplace = True)
    return (tmp_0,)

# Argument extraction function
def replacement_args(in_0):
    """
    Extract input tensor for the replacement
    """
    return (in_0,)

# Simple ReLU kernel
@triton.jit
def simple_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple ReLU kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation
    out = tl.maximum(x, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_relu(input_tensor):
    """Wrapper function for launching the simple ReLU kernel"""
    # Calculate total number of elements
    n_elements = input_tensor.numel()
    
    # Set up block size and grid
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with same shape and dtype
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    simple_relu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    """
    Returns reference to the simple ReLU implementation
    """
    return simple_relu