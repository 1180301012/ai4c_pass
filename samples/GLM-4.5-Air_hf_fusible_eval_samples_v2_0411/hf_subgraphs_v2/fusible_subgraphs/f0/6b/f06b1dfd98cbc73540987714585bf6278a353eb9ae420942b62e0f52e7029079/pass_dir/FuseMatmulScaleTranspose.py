import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x * y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def triton_multiply_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Optimized for small to medium operations
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load with proper type handling
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute
    out = x * y
    
    # Store
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_multiply(x, y):
    # Handle different tensor configurations
    if x.dim() == 0 and y.dim() > 0:
        N = y.numel()
        scalar = True
    elif y.dim() == 0 and x.dim() > 0:
        N = x.numel()
        scalar = True
    else:
        N = max(x.numel(), y.numel())
        scalar = False
    
    # Use optimal block size based on operation size
    if N < 500:
        BLOCK_SIZE = 128  # Smaller block for very small operations
    elif N < 5000:
        BLOCK_SIZE = 256  # Medium block for small operations
    else:
        BLOCK_SIZE = 1024  # Larger block for bigger operations
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output with proper shape and type
    if scalar:
        out = torch.empty_like(y if x.dim() == 0 else x)
    else:
        out = torch.empty_like(x if x.numel() >= y.numel() else y)
    
    # Launch kernel
    triton_multiply_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_multiply