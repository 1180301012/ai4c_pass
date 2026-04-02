import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern matches: add followed by division by 2 (averaging)"""
    tmp = x + y
    result = tmp / 2
    return result

def replacement_args(x, y):
    """Extract arguments for the optimized kernel"""
    return (x, y)

@triton.jit
def optimized_adddiv_kernel(
    x_ptr,
    y_ptr, 
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized fused add+divide kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load both tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Fused: add and divide by 2 (equivalent to average)
    out = (x + y) * 0.5
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_adddiv(x, y):
    """Wrapper function for the optimized add+divide operation"""
    N = x.numel()
    
    # Choose block size that's a power of 2 for Triton compatibility
    if N <= 512:
        BLOCK_SIZE = 512  # Largest power of 2 <= N
        if N == 0:
            num_programs = 0
        else:
            num_programs = 1
    elif N <= 1024:
        BLOCK_SIZE = 1024  # Power of 2
        num_programs = 1
    else:
        BLOCK_SIZE = 1024  # Optimal for larger tensors
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Ensure output has same dtype as inputs
    out = torch.empty_like(x)
    
    # Launch kernel
    optimized_adddiv_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return optimized_adddiv