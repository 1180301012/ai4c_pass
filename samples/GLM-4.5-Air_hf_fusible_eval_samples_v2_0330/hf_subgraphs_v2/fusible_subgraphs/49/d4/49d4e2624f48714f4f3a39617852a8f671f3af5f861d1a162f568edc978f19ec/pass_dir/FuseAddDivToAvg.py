import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern matching: addition followed by division by 2"""
    tmp_2 = x + y
    tmp_3 = tmp_2 / 2
    return tmp_3

def replacement_args(x, y):
    """Extract arguments needed for fusion"""
    return (x, y)

@triton.jit
def fused_avg_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance kernel for computing (x + y) / 2"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load both inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute fused operation: (x + y) / 2
    out = (x + y) * 0.5  # Multiply by 0.5 for better performance than division
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_average(x, y):
    """Wrapper function to launch the fused average kernel"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output tensor with same dtype and device as inputs
    out = torch.empty_like(x)
    
    # Launch kernel
    fused_avg_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the fused average function"""
    return fused_average