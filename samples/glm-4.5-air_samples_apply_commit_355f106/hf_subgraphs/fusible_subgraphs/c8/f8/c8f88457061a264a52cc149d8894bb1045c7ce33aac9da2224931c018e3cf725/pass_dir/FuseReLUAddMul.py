import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x * y

def replacement_args(x, y):
    return (x, y)

@triton.autotune(
    configs=[
        triton.Config(num_warps=4, num_stages=3, kwargs={}),
        triton.Config(num_warps=8, num_stages=3, kwargs={}),
        triton.Config(num_warps=8, num_stages=4, kwargs={}),
        triton.Config(num_warps=16, num_stages=4, kwargs={}),
    ],
    key=['n_elements'],
)
@triton.jit
def optimized_mul_kernel(
    x_ptr,
    scalar_val,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Multiplication with scalar
    out = x * scalar_val
    
    # Store output
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_mul(x, y):
    # We're matching the specific case where y is a scalar [1] and x is a tensor
    if y.numel() == 1 and x.numel() > 1:
        # Scalar multiplication kernel
        N = x.numel()
        
        # Choose optimal block size based on tensor size
        if N <= 65536:  # Small tensors (<= 64K elements)
            BLOCK_SIZE = 256
        elif N <= 524288:  # Medium tensors (<= 512K elements)
            BLOCK_SIZE = 512
        else:  # Large tensors (> 512K elements, e.g., [128, 16, 128, 128] = 33M elements)
            BLOCK_SIZE = 1024
        
        num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        # Load scalar value once
        scalar_val = y.item()
        
        out = torch.empty_like(x)
        
        optimized_mul_kernel[(num_programs,)](
            x_ptr=x,
            scalar_val=scalar_val,
            out_ptr=out,
            n_elements=N,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out
    else:
        # For other cases, fall back to standard PyTorch
        return x * y

def replacement_func():
    return optimized_mul