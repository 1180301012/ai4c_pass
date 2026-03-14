import torch
import triton
import triton.language as tl

def pattern(x):
    # Simple pattern: just division (working perfectly)
    return x / 11.313708498984761

def replacement_args(x):
    return (x,)

@triton.jit
def fused_div_kernel(
    x_ptr,
    out_ptr,
    divisor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Create mask for boundary conditions
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Division operation only (simple and fast)
    result = x / divisor
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_div_kernel_wrapper(x):
    # Use the constant divisor from the original pattern
    divisor = 11.313708498984761
    
    # Determine the optimal block size based on tensor size
    N = x.numel()
    
    # Optimize block sizes for different tensor sizes
    if N >= 512 * 512:  # Large tensors - use larger blocks for better occupancy
        BLOCK_SIZE = 2048
    elif N >= 128 * 128:  # Medium tensors
        BLOCK_SIZE = 1024
    else:  # Small tensors - use smaller blocks to avoid over-subscription
        BLOCK_SIZE = 512
    
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch the optimized division kernel
    fused_div_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        divisor=divisor,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_div_kernel_wrapper