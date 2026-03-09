import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.silu(in_1)
    tmp_1 = tmp_0 + in_0
    return tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def silu_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SiLU(y) = y * sigmoid(y) 
    sigmoid_y = tl.sigmoid(y)
    silu_y = y * sigmoid_y
    
    # Add x to SiLU(y)
    out = x + silu_y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_silu_add(in_0, in_1):
    # Determine tensor size
    n_elements = in_0.numel()
    
    # Choose optimal block size
    if n_elements < 1024:
        BLOCK_SIZE = 128
    elif n_elements < 1024 * 16:
        BLOCK_SIZE = 256  
    else:
        BLOCK_SIZE = 1024
        
    # Calculate grid size
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(in_0)
    
    # Launch kernel
    silu_add_kernel[(num_programs,)](
        x_ptr=in_0,
        y_ptr=in_1,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_silu_add