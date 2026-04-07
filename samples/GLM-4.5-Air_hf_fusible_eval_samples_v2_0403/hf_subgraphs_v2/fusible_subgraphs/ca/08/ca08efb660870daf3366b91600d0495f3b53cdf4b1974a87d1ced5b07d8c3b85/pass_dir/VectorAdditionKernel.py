import torch
import triton
import triton.language as tl

# Pattern matching function - match a simple computation pattern
def pattern(x):
    tmp_0 = x + 1.0
    tmp_1 = tmp_0 * 2.0
    return tmp_1

# Argument extraction function
def replacement_args(x):
    return (x,)

# Direct Triton kernel for vector addition and multiplication
@triton.jit
def simple_computation_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input element
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Perform computation: (x + 1.0) * 2.0
    result = (x + 1.0) * 2.0
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_titon_computation(x):
    """Simple computation using Triton kernel"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Launch Triton kernel
    simple_computation_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function
def replacement_func():
    return simple_titon_computation