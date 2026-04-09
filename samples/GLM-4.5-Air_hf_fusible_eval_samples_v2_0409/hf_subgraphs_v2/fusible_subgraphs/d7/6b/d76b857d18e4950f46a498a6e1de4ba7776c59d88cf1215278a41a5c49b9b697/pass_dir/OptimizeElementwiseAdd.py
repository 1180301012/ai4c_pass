import torch
import triton
import triton.language as tl

@triton.jit
def elementwise_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized elementwise addition kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load operands
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    # Perform addition
    out = x + y

    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

def pattern(x, y):
    """Match the elementwise addition operation"""
    result = x + y
    return result

def replacement_args(x, y):
    """Extract arguments needed for the optimized addition"""
    return (x, y)

@torch.fx.wrap
def optimized_elementwise_add(x, y):
    """Optimized elementwise addition using Triton"""
    # Ensure tensors are on the same device and have same shape
    if x.shape != y.shape:
        raise ValueError(f"Tensor shapes must match: x.shape={x.shape}, y.shape={y.shape}")
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Get number of elements
    n_elements = x.numel()
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    elementwise_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized addition function"""
    return optimized_elementwise_add