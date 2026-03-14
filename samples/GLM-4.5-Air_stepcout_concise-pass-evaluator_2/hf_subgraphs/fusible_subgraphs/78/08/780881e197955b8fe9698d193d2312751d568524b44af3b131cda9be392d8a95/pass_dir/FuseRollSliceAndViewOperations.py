import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Simple addition pattern"""
    return x+y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def triton_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance triton addition with optimized memory access"""
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs with better memory access patterns
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute using fused operations
    out = x + y
    
    # Store result with vectorized write
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add(x, y):
    """Optimized Triton addition wrapper with adaptive grid sizing"""
    x = x.contiguous()
    y = y.contiguous()
    N = x.numel()
    
    if N == 0:
        return torch.empty_like(x)
    
    # Optimized block sizes for different tensor sizes
    if N < 1000:
        BLOCK_SIZE = 256
    elif N < 10000:
        BLOCK_SIZE = 512
    elif N < 50000:
        BLOCK_SIZE = 1024
    elif N < 200000:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty_like(x)

    triton_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out

def replacement_func():
    return triton_add