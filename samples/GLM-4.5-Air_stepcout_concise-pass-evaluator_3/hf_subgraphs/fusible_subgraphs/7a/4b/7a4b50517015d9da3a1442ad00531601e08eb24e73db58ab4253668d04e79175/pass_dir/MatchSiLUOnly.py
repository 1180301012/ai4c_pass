import torch
import triton
import triton.language as tl

# Pattern matching function - matches only silu operation
def pattern(in_0):
    """
    Match only SiLU operation:
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    
    This tests if we can match individual operations in the decomposed graph.
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=False)
    return tmp_0

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel - SiLU only
@triton.jit
def silu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized SiLU kernel: silu(x) = x * sigmoid(x)
    """
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU: silu(x) = x * sigmoid(x) = x * 1/(1 + exp(-x))
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    result = x * sigmoid_x
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_silu(x):
    """
    Optimized SiLU operation wrapper
    """
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=torch.float32)
    
    silu_kernel[(num_programs,)](
        input_ptr=x,
        output_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_silu