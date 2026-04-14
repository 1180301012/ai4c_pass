import torch
import triton
import triton.language as tl

# Pattern matching function - matches the add followed by division pattern
def pattern(in_2, in_3):
    tmp_2 = in_2 + in_3
    tmp_3 = tmp_2 / 2
    return tmp_3

# Argument extraction function
def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Triton kernel for fused add+scale operation
@triton.jit
def fused_add_scale_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused operation: (x + y) / 2
    out = (x + y) * 0.5  # Multiplication is faster than division
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_add_scale(x, y):
    # Get the total number of elements
    n_elements = x.numel()
    
    # Set block size for optimal GPU occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_add_scale_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns the kernel wrapper)
def replacement_func():
    return fused_add_scale