import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(x):
    """
    Match the pattern: boolean mask creation followed by masked_fill
    """
    mask = x != 0
    result = x.masked_fill(mask, -1000.0)
    return result

# Argument extraction function
def replacement_args(x):
    return (x,)

@triton.jit
def optimized_masked_fill_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    replace_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel using arithmetic operations:
    - result = -1000.0 * (x != 0)
    Optimized for memory bandwidth efficiency
    """
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor once and reuse
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Use arithmetic instead of expensive conditional logic
    # For non-zero elements: -1000.0
    # For zero elements: 0.0
    is_non_zero = (x != 0).to(tl.float32)
    out = replace_val * is_non_zero
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_masked_fill(x):
    """
    Wrapper function that launches the optimized masked_fill kernel
    """
    N = x.numel()
    # Use optimal block size for this lightweight operation
    BLOCK_SIZE = 1024  # Best balance of overhead and work per computation
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the kernel with optimized configuration
    optimized_masked_fill_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        replace_val=-1000.0,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (must return function reference)
def replacement_func():
    return optimized_masked_fill