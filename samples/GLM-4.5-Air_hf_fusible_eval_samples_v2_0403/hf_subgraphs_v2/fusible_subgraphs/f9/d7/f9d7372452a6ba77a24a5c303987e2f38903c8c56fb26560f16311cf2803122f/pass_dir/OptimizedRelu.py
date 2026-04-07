import torch
import triton
import triton.language as tl

# Pattern matching function - matches the ReLU operation
def pattern(tmp_0):
    """Matches the ReLU operation:
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace = True)
    Returns tmp_2 which is the ReLU result
    """
    tmp_2 = torch.nn.functional.relu(tmp_0, inplace=True)
    return tmp_2

# Argument extraction function
def replacement_args(tmp_0):
    """Extracts arguments needed for the optimized ReLU kernel"""
    return (tmp_0,)

# Optimized kernel for ReLU operation
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that performs ReLU operation efficiently"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform ReLU operation
    out = tl.maximum(x, 0.0)
    
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_relu(tmp_0):
    """Wrapper function to launch the optimized ReLU kernel"""
    # Determine the number of elements
    n_elements = tmp_0.numel()
    
    # Set block size and grid size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(tmp_0)
    
    # Launch kernel
    relu_kernel[(num_programs,)](
        x_ptr=tmp_0,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_relu