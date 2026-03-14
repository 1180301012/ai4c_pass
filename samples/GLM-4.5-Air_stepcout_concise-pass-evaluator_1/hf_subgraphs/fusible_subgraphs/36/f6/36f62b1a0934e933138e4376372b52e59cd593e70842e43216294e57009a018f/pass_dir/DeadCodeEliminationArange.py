import torch
import triton
import triton.language as tl

# Pattern matching function - matches the dead code pattern
def pattern(tmp_0):
    """
    Pattern matches: arange + add 0 + copy + cleanup
    We only need to match the observable parts: arange and the copy
    """
    # Create the arange tensor (observable operation)
    tmp_0 = torch.arange(128, device=torch.device(type='cuda', index=0))
    # Add 0 (dead code)
    tmp_0 += 0
    # Copy (dead code)
    tmp_1 = tmp_0
    # Cleanup (not observable)
    # tmp_0 = None
    return (tmp_1,)

# Argument extraction function
def replacement_args(tmp_0):
    return (tmp_0,)

# Optimized kernel function - creates arange directly without redundant operations
@triton.jit
def optimized_arange_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Creates an arange sequence directly on GPU without redundant operations
    This eliminates the dead code: add 0 and copy operations
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Create arange values directly: start + offsets where start = 0
    values = offsets
    tl.store(out_ptr + offsets, values, mask=mask)

# Kernel wrapper - creates the arange tensor efficiently
@torch.fx.wrap
def optimized_arange():
    """
    Directly creates an arange tensor on GPU without intermediate operations
    This avoids the dead code pattern: tmp_0 += 0 and tmp_1 = tmp_0
    """
    n_elements = 128
    BLOCK_SIZE = 128  # Use block size that matches the arange length for efficiency
    
    # Create output tensor directly
    out = torch.empty((n_elements,), dtype=torch.int64, device=torch.device(type='cuda', index=0))
    
    # Use only one program since we're creating a simple sequence
    optimized_arange_kernel[(1,)](
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out,)

# Replacement function (no arguments, returns function reference)
def replacement_func():
    return optimized_arange