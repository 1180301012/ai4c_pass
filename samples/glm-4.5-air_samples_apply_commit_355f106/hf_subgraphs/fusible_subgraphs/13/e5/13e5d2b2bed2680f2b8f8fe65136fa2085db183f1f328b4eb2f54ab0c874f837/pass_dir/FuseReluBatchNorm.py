import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple pattern for testing"""
    return x

def replacement_args(x):
    """Extract arguments"""
    return (x,)

@triton.jit
def add_kernel(
    x_ptr,
    running_mean_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Addition kernel using Triton"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    running_mean = tl.load(running_mean_ptr + 0)  # Broadcast scalar to all elements
    
    # Addition operation
    out = x + running_mean
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(input_tensor, running_mean):
    """Triton addition function wrapper"""
    # Calculate number of elements
    n_elements = input_tensor.numel()
    
    # Determine block size and grid size
    BLOCK_SIZE = 1024  # Optimal for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(input_tensor)
    
    # Launch kernel
    add_kernel[(num_programs,)](
        x_ptr=input_tensor,
        running_mean_ptr=running_mean,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out  # Return output for pattern matching

def replacement_func():
    """Return the addition function"""
    return triton_add