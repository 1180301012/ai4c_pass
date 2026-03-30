import torch
import triton
import triton.language as tl

# Pattern matching function - start with just the scaling operation
def pattern(in_0):
    """Match the scaling operation"""
    tmp_1 = in_0 * 1000000.0
    return tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Simple Triton kernel for the scaling operation
@triton.jit
def simple_scale_kernel(
    in_0_ptr,
    out_ptr, 
    n_elements,
    scale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(in_0_ptr + offsets, mask=mask)
    # Apply scaling
    out = x.to(tl.float32) * scale_factor
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def simple_scale_torch(in_0):
    # Move to GPU if not already there
    if in_0.device.type != 'cuda':
        in_0 = in_0.cuda()
    
    # Get input shape
    n_elements = in_0.numel()
    
    # Create output tensor
    out = torch.empty_like(in_0, dtype=torch.float32)
    
    # Launch kernel
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    simple_scale_kernel[(num_programs,)](
        in_0,
        out,
        n_elements,
        1000000.0,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

# Replacement function
def replacement_func():
    return simple_scale_torch