import torch
import triton
import triton.language as tl

def pattern(in_1, sigmoid_expanded, in_0):
    """
    Pattern: element-wise multiplication + addition
    This matches: in_1 * sigmoid_expanded + in_0
    """
    tmp_3 = in_1 * sigmoid_expanded
    tmp_4 = tmp_3
    tmp_4 += in_0
    return tmp_4

def replacement_args(in_1, sigmoid_expanded, in_0):
    return (in_1, sigmoid_expanded, in_0)

@triton.jit
def fused_mul_add_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    batch,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * channels * height * width
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load all three operands
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    
    # Fused multiply-add operation
    out = x * y + z
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_mul_add(in_1, sigmoid_expanded, in_0):
    """Fused element-wise multiplication and addition"""
    # Get tensor shapes (should all be the same)
    shape = in_1.shape
    
    # Create output tensor
    out = torch.empty_like(in_1)
    
    # Get dimensions
    batch, channels, height, width = shape
    
    # Calculate total elements
    total_elements = batch * channels * height * width
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_mul_add_kernel[grid_size](
        in_1,
        sigmoid_expanded,
        in_0,
        out,
        batch,
        channels,
        height,
        width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_mul_add