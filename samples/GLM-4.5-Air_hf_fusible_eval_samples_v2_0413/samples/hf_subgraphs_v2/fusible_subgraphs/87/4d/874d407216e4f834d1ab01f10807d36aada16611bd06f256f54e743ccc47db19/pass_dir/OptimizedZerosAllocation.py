import torch
import triton
import triton.language as tl

# Pattern matching function - matches zeros creation with different shapes
def pattern(tmp_1, shape):
    """Match: tmp_1.new_zeros(shape)"""
    result = tmp_1.new_zeros(shape)
    return result

# Argument extraction function
def replacement_args(tmp_1, tmp_4_shape):
    """Extract dtype/device from tmp_1 and pass the desired shape"""
    return (tmp_1, tmp_4_shape)

# Optimized Triton kernel for efficient zero tensor initialization
@triton.jit
def fill_zeros_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for filling tensor with zeros"""
    # Each program handles a contiguous block of elements
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Fill with zeros
    zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(out_ptr + offsets, zeros, mask=mask)

# Kernel wrapper that handles different shapes and dtypes
@torch.fx.wrap
def optimized_zeros_like(source_tensor, shape):
    """Optimized zeros tensor with custom shape"""
    n_rows, n_cols = shape
    n_elements = n_rows * n_cols
    
    # Determine dtype and device from source tensor
    dtype = source_tensor.dtype
    device = source_tensor.device
    
    # Optimized block size for maximum GPU occupancy
    BLOCK_SIZE = 1024  # Good for memory coalescing
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor
    out = torch.empty(shape, dtype=dtype, device=device)
    
    # Launch Triton kernel - only if size > 0 to avoid unnecessary kernel launches
    if n_elements > 0:
        fill_zeros_kernel[(num_programs,)](
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

# Replacement function - returns the optimized kernel wrapper
def replacement_func():
    return optimized_zeros_like