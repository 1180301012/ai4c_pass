import torch
import triton
import triton.language as tl

def pattern(tmp_4):
    """Fuse the activation sequence: sigmoid → subtract 0.25 → multiply by pi"""
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7

def replacement_args(tmp_4):
    return (tmp_4,)

@triton.jit
def fused_activation_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for sigmoid(x) - 0.25) * pi"""
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # fused activation: (sigmoid(x) - 0.25) * pi
    # Using fast approximate sigmoid for better performance
    x_fast = tl.where(x > 0, x / (1 + tl.exp(-x)), 1 - 1 / (1 + tl.exp(x)))
    result = (x_fast - 0.25) * 3.141592653589793
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_activation(tmp_4):
    """Function wrapper for the fused activation kernel"""
    n_elements = tmp_4.numel()
    output = torch.empty_like(tmp_4)
    
    # Calculate optimal grid size
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_activation_kernel[grid_size](
        tmp_4,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

def replacement_func():
    return fused_activation