import torch
import triton
import triton.language as tl

# Pattern matching function for scalar multiplication
def pattern(in_1):
    """Pattern: in_1 * 0.3535533905932738"""
    tmp_0 = in_1 * 0.3535533905932738
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized Triton kernel for scalar multiplication
@triton.jit
def scalar_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scalar_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance kernel for element-wise scalar multiplication"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to handle edge cases
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform scalar multiplication
    out = x * scalar_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_scalar_mul(x, scalar):
    """Wrapper function to launch the optimized kernel"""
    N = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with autotuning for better performance
    scalar_mul_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scalar_val=scalar,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (must return a callable function reference)
def replacement_func():
    return lambda in_1: optimized_scalar_mul(in_1, 0.3535533905932738)