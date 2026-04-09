import torch
import triton
import triton.language as tl

# Pattern matching function - matches addition + ReLU sequence
def pattern(in_1, tmp_2):
    tmp_3 = in_1 + tmp_2
    tmp_4 = torch.relu_(tmp_3)
    return tmp_4

# Argument extraction function
def replacement_args(in_1, tmp_2):
    return (in_1, tmp_2)

# Triton kernel for fused addition and ReLU
@triton.jit
def fused_add_relu_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements  # Mask to ensure we don't go out of bounds
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform fused addition and ReLU
    result = x + y
    result = tl.maximum(result, 0.0)
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_add_relu(x, y):
    # Calculate total elements
    n_elements = x.numel()
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate block size and number of programs  
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_relu_kernel[(num_programs,)](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_add_relu