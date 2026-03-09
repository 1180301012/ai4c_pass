import torch
import triton
import triton.language as tl

# Pattern matching for dropout with training=False
def pattern(x):
    tmp_2 = torch.nn.functional.dropout(x, 0.1, False, False)
    return tmp_2

# Extract arguments for the replacement function  
def replacement_args(x):
    return (x,)

# Triton kernel for scaling operation (dropout inference = multiply by 0.9)
@triton.jit
def dropout_scale_kernel(
    x_ptr,      # Input tensor
    out_ptr,    # Output tensor  
    n_elements, # Total number of elements
    scale_val,  # Scaling factor (0.9 for p=0.1 dropout)
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply scaling (equivalent to dropout with training=False, p=0.1)
    out = x * scale_val
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper (must be decorated with @torch.fx.wrap)
@torch.fx.wrap
def dropout_inference_scale(x):
    # Get tensor properties
    N = x.numel()
    
    # Scaling factor for dropout with p=0.1: (1 - 0.1) = 0.9
    scale_val = 0.9
    
    # Optimal block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    dropout_scale_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scale_val=scale_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return dropout_inference_scale