import torch
import triton
import triton.language as tl

# Pattern matching function - must exactly match the model computation
def pattern(in_0, in_1, in_2):
    """Match the entire forward computation including SiLU and detach operations"""
    tmp_0 = torch.nn.functional.silu(in_0)
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3, tmp_0)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the optimized kernel"""
    return (in_0, in_1, in_2)

# High-performance SiLU kernel using Triton
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU kernel: out = x * sigmoid(x) = x / (1 + exp(-x))"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU using expf for efficiency
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_forward(x, y, z):
    """
    Optimized forward implementation.
    """
    # Optimized SiLU computation for x
    silu_result = torch.empty_like(x, device=x.device)
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Optimal block size for most GPUs
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel for SiLU computation
    silu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=silu_result,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return the same structure as the original
    return (y, z, silu_result, silu_result)

# Replacement function (returns the optimized function)
def replacement_func():
    return optimized_forward