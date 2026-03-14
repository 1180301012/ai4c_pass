import torch
import triton
import triton.language as tl

# Pattern matching function for scalar multiplication with 0.125
def pattern(in_1):
    """Match scalar multiplication operation with 0.125"""
    tmp_0 = in_1 * 0.125
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# High-performance triton kernel for scalar multiplication
@triton.jit
def scalar_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for scalar multiplication"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform scalar multiplication
    out = x * scalar
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper with autotuning capabilities
@torch.fx.wrap
def optimized_scalar_mul(x, scalar_val=0.125):
    """High-performance scalar multiplication using Triton"""
    N = x.numel()
    
    # Use a reasonable block size for good GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the optimized kernel
    scalar_mul_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scalar=scalar_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_scalar_mul