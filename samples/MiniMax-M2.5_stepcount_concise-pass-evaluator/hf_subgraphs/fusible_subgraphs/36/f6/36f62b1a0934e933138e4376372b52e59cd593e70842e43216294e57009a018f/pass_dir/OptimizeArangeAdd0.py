import torch
import triton
import triton.language as tl


# Pattern matching function - matches the exact computation in model.py
# The pattern: torch.arange(128) followed by += 0 (a no-op)
def pattern():
    """
    Match the pattern: torch.arange(128, device=...) followed by += 0
    Returns the result of the arange operation.
    """
    # First create the arange tensor
    tmp_0 = torch.arange(128, device=torch.device(type='cuda', index=0))
    # Add 0 (no-op, but appears in the graph)
    tmp_0 += 0
    # Return the result
    return tmp_0


# Argument extraction function - no arguments needed since there are no inputs
def replacement_args():
    return ()


# Optimized Triton kernel to generate arange sequence
@triton.jit
def arange_kernel(
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel to generate arange sequence [0, 1, 2, ..., n-1]"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Generate the sequence values
    values = offsets.to(tl.int64)
    
    # Store results
    tl.store(output_ptr + offsets, values, mask=mask)


# Wrapper function that uses the Triton kernel
@torch.fx.wrap
def optimized_arange():
    """
    Optimized implementation using Triton kernel.
    Generates the sequence [0, 1, 2, ..., 127] directly.
    """
    N = 128
    BLOCK_SIZE = 128  # Small tensor, use single block
    
    # Allocate output tensor
    output = torch.empty((N,), dtype=torch.int64, device=torch.device(type='cuda', index=0))
    
    # Launch kernel with grid = 1 (single program handles all elements)
    arange_kernel[(1,)](
        output_ptr=output,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# Replacement function - returns the optimized function
def replacement_func():
    return optimized_arange