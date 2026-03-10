import torch
import triton
import triton.language as tl
import math

def pattern(x):
    # Match the tensor that cos() and sin() are applied to
    # This captures the tmp_4 in the computation
    cos_val = x.cos()
    sin_val = x.sin()
    # We return both values since they are used in the computation
    return cos_val, sin_val

def replacement_args(x):
    return (x,)

@triton.jit
def fused_cos_sin_kernel(
    x_ptr,
    out_cos_ptr,
    out_sin_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute both cos and sin simultaneously
    cos_val = tl.cos(x)
    sin_val = tl.sin(x)
    
    # Store both results
    tl.store(out_cos_ptr + offsets, cos_val, mask=mask)
    tl.store(out_sin_ptr + offsets, sin_val, mask=mask)

@torch.fx.wrap
def fused_cos_sin(x):
    # Get input shape and compute total elements
    n_elements = x.numel()
    
    # Set optimal block size based on GPU architecture
    BLOCK_SIZE = 1024
    
    # Calculate number of programs needed
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensors
    cos_out = torch.empty_like(x)
    sin_out = torch.empty_like(x)
    
    # Launch the fused kernel
    fused_cos_sin_kernel[(num_programs,)](
        x_ptr=x,
        out_cos_ptr=cos_out,
        out_sin_ptr=sin_out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return cos_out, sin_out

def replacement_func():
    return fused_cos_sin