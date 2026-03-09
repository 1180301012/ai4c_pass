import torch
import triton
import triton.language as tl

# Pattern matching function - matches the SiLU computation (accepts all inputs)
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    return tmp_0

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0,)

# Triton optimized SILU kernel
@triton.jit
def silu_kernel__(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SILU: x * sigmoid(x)
    # Using fast sigmoid approximation for better performance
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_silu(x):
    N = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    silu_kernel_[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_silu